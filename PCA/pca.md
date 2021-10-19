![image-20211009153115786](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211009153118.png)

## PCA原理概述

PCA是无监督的数据降维方法，其主要思路是将原始数据通过投影直线投影后尽可能的分散，即拥有最大方差。

假设现在有**n**个**m**维数据，其中$x_i$表示第$i$个数据，其是一个$m*1$的向量。$X$表示整个数据集，其维数为$m*n$。

与平常的认知不同的是，通常数据集的格式是$n*m$，这边需要转置一下。我们的目标是最大化投影距离：
$$
\begin{align}
max \quad \frac{1}{n}\sum_{i=1}^{n}\parallel\vec{x_i}\vec{u}\parallel \\
st: \quad u^Tu=1 
\end{align}
$$
经过平方化，向量内积，可化成
$$
max \quad \frac{1}{n}\sum_{i=1}^{n}u^Tx_ix_i^Tu
$$
由于
$$
\sum_{i=1}^{n}x_ix_i^T=XX^T
$$


原式可以化简成
$$
max \quad \frac{1}{n}u^TXX^Tu
$$
使用拉格朗日乘子法求解该问题：
$$
\begin{align}
min \quad -u^TXX^Tu \\
st: \quad u^Tu=1 
\end{align}
$$
引入$\lambda$
$$
-u^TXX^Tu+\lambda(u^Tu-1)
$$
对$u$求导，可得
$$
XX^Tu = \lambda u
$$
此时，最大化目标
$$
\frac{1}{n}u^TXX^Tu = \frac{1}{n}u^T\lambda u=\frac{1}{n}\lambda
$$
可以明显的看出，$\lambda$和$u$分别是$XX^T$的特征值和特征向量，要想原式最大；则只需要取到最大的特征值以及其对应的特征向量就行了。

如果将维数从n维降到k维，则只需要将特征值排序后取得前k个最大的特征值，以及其对应的特征向量。将特征向量组合形成最终的投影矩阵，我们可以通过以下公式计算降维后的信息保有量:
$$
\sqrt{\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^n\lambda_i}}
$$

> [csdn主成分分析（PCA）原理详解](https://blog.csdn.net/zhongkelee/article/details/44064401)
>
> [刘建平-[主成分分析（PCA）原理总结](https://www.cnblogs.com/pinard/p/6239403.html)

## PCA代码与实例

```python
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

def my_pca(x,k):
	#step1 所有样本区中心化
	avg = np.average(x,axis=0)
	x = x - avg
	#step2 计算x的协方差矩阵
	mat = np.cov(x.T)
	print(mat.shape)
	#step3 计算协方差矩阵的特征值和特征向量,并选取前k个特征向量
	a,b = np.linalg.eig(mat)
	np.linalg.eig(b)
	index = np.argsort(-a) #降序排序
	p = b[index][:,:k]
	#step4 Y=PX得到降维后的数据
	return x@p


if __name__ == "__main__":

	data = datasets.load_boston()["data"]
	print(data.shape)
	
	pca = PCA(n_components=2)
	pca.fit(data)
	print(pca.transform(data))

	print(my_pca(data,2))

```

