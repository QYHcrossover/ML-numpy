import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

def my_pca(x,k):
	#step1 所有样本区中心化
	avg = np.average(x,axis=0)
	x = x - avg
	#step2 计算x的协方差矩阵
	mat = np.cov(x.T)
	#step3 计算协方差矩阵的特征值和特征向量,并选取前k个特征向量
	a,b = np.linalg.eig(mat)
	np.linalg.eig(b)
	index = np.argsort(-a) #降序排序
	p = b[index][:,:k]
	#step4 Y=PX得到降维后的数据
	return x@p


if __name__ == "__main__":

	data = datasets.load_boston()["data"]
	
	pca = PCA(n_components=2)
	pca.fit(data)
	print(pca.transform(data))

	print(my_pca(data,2))
