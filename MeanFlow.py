import os
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
# pip install einops>=0.4
# pip install einops-exts


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def norm_layer(channels):
    return nn.GroupNorm(32, channels)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        """
        Attention block with shortcut

        Args:
            channels (int): channels
            num_heads (int, optional): attention heads. Defaults to 1.
        """
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t, y):
        """
        Apply the module to `x` given `t` timestep embeddings, `y` conditional embedding same shape as t.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x, t, y):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t, y)
            else:
                x = layer(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class LayerNorm(nn.Module):
    def __init__(self, feats, stable=True, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim
        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups=8,
        norm=True,
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        norm_context=False,
        cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = dim if context_dim is None else context_dim

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        context = self.norm_context(context)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)
        # add null key / value for classifier free guidance in prior net
        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads,  b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q = q * self.scale
        # similarities
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale
        # masking
        max_neg_value = -torch.finfo(sim.dtype).max
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = torch.einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, use_global_context=False, groups=8):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, groups=groups)

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels*2)
        )

        self.block2 = Block(out_channels, out_channels, groups=groups)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        cond_dim = time_channels
        self.gca = GlobalContext(dim_in=out_channels, dim_out=out_channels) if use_global_context else Always(1)
        self.cross_attn = CrossAttention(dim=out_channels, context_dim=cond_dim)

    def forward(self, x, t, y):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `y` has shape `[batch_size, num_time_tokens, cond_dim]`
        """
        h = self.block1(x)

        # Apply condition on hidden state via cross attention
        context = y
        size = h.size(-2)
        hidden = rearrange(h, 'b c h w -> b (h w) c')
        attn = self.cross_attn(hidden, context)
        attn = rearrange(attn, 'b (h w) c -> b c h w', h=size)
        h += attn

        # Add time step embeddings
        t = self.time_emb(t)
        t = rearrange(t, 'b c -> b c 1 1')
        scale_shift = t.chunk(2, dim=1)
        h = self.block2(h, scale_shift=scale_shift)

        h *= self.gca(h)
        return h + self.shortcut(x)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        label_num=10,
        num_time_tokens=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # condition embedding
        cond_dim = time_embed_dim
        self.label_embedding = nn.Embedding(label_num + 1, time_embed_dim)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_embed_dim, num_time_tokens * cond_dim),
            Rearrange('b (r d) -> b r d', r=num_time_tokens)
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor):
        """Apply the model to an input batch.

        Args:
            x (Tensor): [N x C x H x W]
            t (Tensor): [N,] a 1-D batch of timesteps.
            y (Tensor): [N,] LongTensor conditional labels.

        Returns:
            Tensor: [N x C x ...]
        """
        # time step embedding
        t = self.time_embed(timestep_embedding(t, self.model_channels))
        y = self.label_embedding(y)
        y = self.to_time_tokens(y)

        hs = []
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, t, y)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t, y)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t, y)
        return self.out(h)


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ"""
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (w.detach() * loss).mean()


class MeanFlow:
    def __init__(self, flow_ratio=0.50, time_dist='lognorm',
                 cfg_ratio=0.10, cfg_scale=2.0, num_classes=10):
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale
        self.num_classes = num_classes

    def sample_t_r(self, batch_size, device):
        """Sample t and r ensuring r <= t."""
        if self.time_dist == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        else:  # lognorm
            mu, sigma = -0.4, 1.0
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r


    @torch.no_grad()
    def sample(self, model: nn.Module, y: torch.LongTensor, image_size, batch_size=8, channels=3):
        """Sample images using MeanFlow."""
        shape = (batch_size, channels, image_size, image_size)
        device = next(model.parameters()).device
        z = torch.randn(shape, device=device)

        t = torch.ones(batch_size, device=device)
        r = torch.zeros(batch_size, device=device)

        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        v = model(z, t, y)
        z = z - (t_ - r_) * v

        z = (z.clip(-1, 1) + 1) * 0.5  # denormalize
        return z


    def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor,
                     r: torch.LongTensor, y: torch.LongTensor):
        # compute train losses
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        e = torch.randn_like(x_start)
        x_start_norm = x_start * 2 - 1  # normalize to [-1, 1]
        z = (1 - t_) * x_start_norm + t_ * e
        v = e - x_start_norm

        # CFG
        uncond = torch.ones_like(y) * self.num_classes
        cfg_mask = torch.rand_like(y.float()) < self.cfg_ratio
        y_input = torch.where(cfg_mask, uncond, y)

        if self.w is not None:
            with torch.no_grad():
                u_t = model(z, t, uncond)
            v_hat = self.w * v + (1 - self.w) * u_t
            cfg_mask_img = rearrange(cfg_mask, "b -> b 1 1 1").bool()
            v_hat = torch.where(cfg_mask_img, v, v_hat)
        else:
            v_hat = v

        # JVP computation
        def model_fn(z, t, r):
            return model(z, t, y_input)

        jvp_args = (
            model_fn,
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )
        u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)

        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - u_tgt.detach()
        loss = adaptive_l2_loss(error)

        mse_val = (error.detach() ** 2).mean()
        return loss, mse_val


batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# use MNIST dataset
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# define model and diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=1,
    model_channels=96,
    out_channels=1,
    channel_mult=(1, 2, 2),
    attention_resolutions=[],
    label_num=len(dataset.classes)
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
meanflow = MeanFlow(num_classes=len(dataset.classes))

# train
epochs = 10
for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        batch_size = images.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        # sample t and r for MeanFlow
        t, r = meanflow.sample_t_r(batch_size, device)

        loss, mse_val = meanflow.train_losses(model, images, t, r, labels)

        if step % 200 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}, MSE: {mse_val.item():.6f}")

        loss.backward()
        optimizer.step()

batch_size = 32
y = torch.randint(0, 10, (batch_size,)).cuda()
generated_images = meanflow.sample(model, y, 28, batch_size=batch_size, channels=1)
# generated_images: [batch_size=64, channels=1, height=28, width=28]

# generate new images
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        img = generated_images[n_row*8+n_col, 0].cpu().numpy()
        f_ax.imshow(img * 255, cmap="gray")
        f_ax.axis("off")
        plt.title(f"condition: {y[n_row*8+n_col].item()}")
