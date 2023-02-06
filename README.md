# PyTorch-DDPM

500 行代码用 PyTorch 实现降噪扩散模型 DDPM

如何实现？可参考个人精简后的公式： https://timecat.notion.site/DDPM-b8e2a91927d249fdbcf7c82f2eb6f846

> 建议使用 codelab 打开 notebook，可以不用自己配环境了
> 
> [codelab: DDPM](https://colab.research.google.com/github/LinXueyuanStdio/PyTorch-DDPM/blob/master/DDPM.ipynb)
> 
> [codelab: Classifier-Free DDPM](https://colab.research.google.com/github/LinXueyuanStdio/PyTorch-DDPM/blob/master/ClassifierFreeDDPM.ipynb)

## DDPM

从随机噪声中降噪生成图片

运行 `python ddpm.py` 或者打开 notebook `DDPM.ipynb`

![](assets/DDPM_each_step.png)

## Classifier-Free DDPM

条件控制 DDPM：给定一个数字，根据数字生成图片

运行 `python classifier_free_ddpm.py` 或者打开 notebook `ClassifierFreeDDPM.ipynb`

![](assets/classifier_free_DDPM_all.png)
![](assets/classifier_free_DDPM_each_step.png)
