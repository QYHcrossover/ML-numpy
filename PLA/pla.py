import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class PLA:
	def __init__(self,la=1):
		self.theta = None
		self.la = la

	def fit(self,X,y):
		self.theta = np.zeros([X.shape[-1]+1,1])  # theta 定义为列向量
		X_ = np.ones([X.shape[0],X.shape[1]+1])   # 扩展 X
		X_[:,1:] = X
		X_y = X_ * y  # 将X与ｙ相乘得到
		while True:
			R = X_y @ self.theta # 得到矩阵
			B = R > 0
			if np.alltrue(B):
				break
			for xy in X_y[B.squeeze()==False]:
				self.theta += self.la * xy.reshape(-1,1)

	def predict(self,X):
		X_ = np.ones([X.shape[0],X.shape[1]+1])   # 扩展 X
		X_[:,1:] = X
		predicts = X_@self.theta
		index1 = predicts<=0
		index2 = predicts>0
		predicts[index1] = -1
		predicts[index2] = 1
		return predicts

if __name__ == "__main__":
	X,y = make_blobs(200,2,2,random_state=222)
	plt.scatter(X[y==0][:,0],X[y==0][:,1],color="red")
	plt.scatter(X[y==1][:,0],X[y==1][:,1],color="blue")
	y[y==0] = -1
	
	pla = PLA()
	pla.fit(X,y.reshape(-1,1))
	k = - pla.theta[1][0] / pla.theta[2][0]
	b = - pla.theta[0][0] / pla.theta[2][0]
	minx,maxx = np.min(X,axis=0)[0],np.max(X,axis=0)[0]
	print(minx,maxx)
	plt.plot([minx,maxx],[k*minx+b,k*maxx+b],"m-")
	plt.show()