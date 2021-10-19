> [线性判别分析LDA原理总结](https://www.cnblogs.com/pinard/p/6244265.html)

## LDA笔记

共轭就是矩阵每个元素都取共轭（实部不变，虚部取负）。

转置就是把矩阵的每个元素按左上到右下的所有元素对称调换过来。

## LDA推导过程

![image-20211009190925530](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211009190927.png)

## LDA算法流程

1. 计算类内散度矩阵$S_w$
2. 计算类间散度矩阵$S_b$
3. 计算矩阵$S_w^{-1}S_b$
4. 计算$S_w^{-1}S_b$的最大的d个特征值和对应的d个特征向量$(W_1,W_2,W_3……,W_d)$，得到投影矩阵$W$
5. 对样本集中的每一个样本特征$x_i$,转换为新的样本$Z_i=W^Tx_i$

### LDA算法实例

```python
import numpy as np
from sklearn import datasets

#导入数据集
cancer = datasets.load_breast_cancer() 
data = cancer["data"]     #shape:[569,30]
labels = cancer["target"] #shape:[30,]

#IDA算法核心

#求类间距离
X0 = data[labels==0].T  #[30,212]
X1 = data[labels==1].T  #[30,357]
mu0 = np.average(X0,axis=1)[:,np.newaxis] #[30,1]
mu1 = np.average(X1,axis=1)[:,np.newaxis] #[30,1]
sb = (mu0-mu1) @ (mu0-mu1).T #[30,30]

#求类内距离
sigma0 = (X0-mu0) @ (X0-mu0).T
sigma1 = (X1-mu1) @ (X1-mu1).T
sw = sigma0 + sigma1 #[30,30]

#求特征向量和特征值
mat = np.linalg.inv(sw)@sb #[30,30]
a, b = np.linalg.eig(mat)
#去掉虚数部分
a = np.real(a)
b = np.real(b)

#求出投影矩阵
index = np.argsort(-a) #降序排序
p = b[index][:,:2] #[30,2]

#完成数据转换
result = data @ p
```



