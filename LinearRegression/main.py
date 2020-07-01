import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
	def __init__(self,lr=0.0001,thr=1e-6):
		self.lr = lr
		self.thr = thr

	def fit(self,X,y):
		#构造theta
		self.theta = np.random.randn(X.shape[-1]+1)[:,np.newaxis]
		#构造扩展的X
		paddedX = np.ones([X.shape[0],X.shape[1]+1])
		paddedX[:,:-1] = X
		#开始进行梯度下降
		epoch = 0
		while True:
			#计算loss
			loss = 0.5*np.mean((paddedX@self.theta - y)**2)
			print("第{}次训练，loss大小为{}".format(epoch,loss))
			#计算梯度
			grad = paddedX.T@(paddedX@self.theta-y)
			#是否收敛
			if abs(np.sum(grad))<self.thr:
				break
			#梯度下降 
			self.theta -= self.lr*grad
			epoch += 1


if __name__ == "__main__":
	data_size = 100
	x = np.random.uniform(low=1.0, high=10.0, size=data_size)
	y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)
	lr = LinearRegression()
	lr.fit(x[:,np.newaxis],y[:,np.newaxis])
	plt.scatter(x, y, marker='.')
	plt.plot([1,10],[lr.theta[0]*1+lr.theta[1],lr.theta[0]*10+lr.theta[1]],"m-")
	plt.show()