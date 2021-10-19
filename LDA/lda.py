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