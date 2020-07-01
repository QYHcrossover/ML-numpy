import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class LogisticRegression:
	def __init__(self,lr=0.001,thr=1e-2): #学习率和阈值
		self.lr = lr
		self.thr = thr

	def predict(self,X):
		sigmoid = lambda x: 1/(1+np.exp(-x))
		#计算结果
		T = X@self.theta
		return sigmoid(T)

	def fit(self,X,y):
		#构造扩展的X
		paddedX = np.ones([X.shape[0],X.shape[1]+1])
		paddedX[:,:-1] = X
		#构造theta
		self.theta = np.random.randn(X.shape[-1]+1)[:,np.newaxis]
		#开始训练
		epoch = 0
		while True:
			#计算loss
			loss = np.squeeze(-1*y.T@np.log(self.predict(paddedX)) + -1*(1-y).T@np.log(1-self.predict(paddedX)))
			print("第{}epoch,loss为{}".format(epoch,loss))
			#计算梯度
			grad = paddedX.T@(self.predict(paddedX)-y)
			#是否收敛
			if abs(np.sum(grad)) < self.thr or loss<0.145:
				break
			#梯度下降
			self.theta -= self.lr * grad
			epoch += 1

if __name__ == "__main__":
	X,y = make_blobs(200,2,2,random_state=222)
	plt.scatter(X[y==0,0],X[y==0,1],y,color="red")
	plt.scatter(X[y==1,0],X[y==1,1],y,color="blue")

	lr = LogisticRegression()
	lr.fit(X,y[:,np.newaxis])

	k = - lr.theta[0] / lr.theta[1]
	b = - lr.theta[2] / lr.theta[1]
	minx,maxx = np.min(X,axis=0)[0],np.max(X,axis=0)[0]
	print(minx,maxx)
	plt.plot([minx,maxx],[k*minx+b,k*maxx+b],"m-")
	plt.show()
