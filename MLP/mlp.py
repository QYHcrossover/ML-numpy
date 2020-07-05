import numpy as np
import math

class MLP:
	def __init__(self,units,activs):
		self.units = units
		self.length = len(units)
		self.activations = activs
		assert len(units)-1 == len(activs) and set(activs).issubset(set(["noactiv","relu","sigmoid","softmax","tanh"])) and "softmax" not in activs[:-1]
		#构造激活函数和激活函数的导数
		activDict,derivDict = MLP.Activations()
		#！默认第0除了A[0]也就是X外，其他W,b,g都是None
		self.activs = [None]+[activDict[i] for i in activs]
		self.derivs = [None]+[derivDict[i] if i!="softmax" else None for i in activs]
		#随机初始化W和b
		self.Ws = [None]+[2*np.random.random([units[i+1],units[i]])-1 for i in range(0,len(units)-1)]
		self.bs = [None]+[np.zeros([units[i+1],1]) for i in range(0,len(units)-1)]

	#激活函数与其导数
	def Activations():
		#激活函数
		noactiv = lambda x:x
		sigmoid = lambda x: 1/(1+np.exp(-x))
		relu = lambda x: 1*(x>0)*x
		softmax = lambda x:np.exp(x)/np.sum(np.exp(x),axis=0)
		tanh = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		activations = {"noactiv":noactiv,"sigmoid":sigmoid,"relu":relu,"softmax":softmax,"tanh":tanh}
		#激活函数的导数
		noactiv_d = lambda x:np.ones_like(x)
		sigmoid_d = lambda x: sigmoid(x)*(1-sigmoid(x))
		relu_d = lambda x: 1*(x>0)
		tanh_d = lambda x: 1-tanh(x)**2
		'''定义的softmax的导数，在实际中常与交叉熵函数结合起来求导
		且下面的公式只适合单个样本，多个样本下涉及矩阵对矩阵的求导比较麻烦
		def softmax_d(x):
			y = MLP.softmax(x)
			temp =  -1*y@y.T
			print(temp)
			for i in range(x.shape[0]):
				temp[i][i] += y[i].squeeze()
			return temp
		'''
		derivatives = {"noactiv":noactiv_d,"sigmoid":sigmoid_d,"relu":relu_d,"tanh":tanh_d}
		return activations,derivatives


	#前向传播
	def forward(self,X):
		#同时记录Z和A
		Zs,As = [None]*self.length,[None]*self.length
		#初始化A[0]为X
		As[0] = X
		#逐层计算
		for i in range(1,self.length):
			Zs[i] = self.Ws[i]@As[i-1] + self.bs[i]
			As[i] = self.activs[i](Zs[i])
		return Zs,As

	#反向传播
	def backward(self,X,Y,lr,loss_function):
		amount = X.shape[-1] #样本数量
		Zs,As = self.forward(X)
		loss = loss_function.calLoss(As[-1],Y)
		if self.activations[-1] == "softmax":
			dZ = As[-1] - Y
		else:
			dA = loss_function.calDeriv(As[-1],Y)
		for l in range(self.length-1,0,-1):
			#分别计算dZ,dW,db,dA
			if self.activations[-1]!="softmax" or l<self.length-1:
				dZ = dA * self.derivs[l](Zs[l])
			dW = 1/amount * dZ @ As[l-1].T
			db = 1/amount * np.sum(dZ,axis=1,keepdims=True)
			dA = self.Ws[l].T @ dZ
			#梯度下降
			self.Ws[l] -= lr*dW
			self.bs[l] -= lr*db
		return loss

	#主函数
	def fit(self,X,Y,lr,max_iters,loss_function,batch_size=None):
	 	#X,y,units是神经网络个数，activations为激活函数列表，loss为损失函数以及导数计算方式
	 	assert X.shape[-1] == self.units[0] and Y.shape[-1] == self.units[-1] #第一维和最后一维需要匹配
	 	#为了不将转置施加于W和b上，故将X和Y转置
	 	X,Y = X.T,Y.T
	 	amount = X.shape[-1] #样本数量
	 	#开始迭代
	 	for epoch in range(max_iters):
	 		#batch梯度下降或minibatch梯度下降
	 		if not batch_size:
	 			loss_avg = self.backward(X,Y,lr,loss_function)
	 		else:
	 			loss_avg = 0
	 			for i in range(math.ceil(amount/batch_size)):
	 				loss = self.backward(X[:,i*batch_size:i*batch_size+batch_size],Y[:,i*batch_size:i*batch_size+batch_size],lr,loss_function)
	 				loss_avg = (i/(i+1))*loss_avg +(1/(i+1))*loss
	 		print("第{}轮训练，loss大小为{}".format(epoch,loss_avg))

#损失函数的模板类，该类定义了两个方法分别是
#计算loss和根据loss计算导数、该类必须被实体类继承
class LossFunc:
	def calLoss(y,y_h):
		pass
	def calDeriv(y,y_h):
		pass

if __name__ == "__main__":
	# ##例子一、分类模型，mnist的例子
	# #载入数据集
	# X_train = np.load(r"..\SVM\之前\X_train.npy")
	# y_train = np.load(r"..\SVM\之前\y_train.npy")
	# #将y_train进行one-hot编码
	# def one_hot(y,feature_size):
	#     LB = np.zeros([len(y), feature_size], dtype=np.int32)
	#     for i in range(len(y)):
	#         LB[i][y[i]] = 1
	#     return LB
	# y_train = one_hot(y_train,10)
	# #构建交叉熵损失函数
	# class CrossEntropy(LossFunc):
	# 	def calLoss(y,y_h):
	# 		return np.mean(np.sum(-1*y_h*np.log(y),axis=0))
	# #构建模型
	# mlp = MLP([784,50,50,10],["tanh","tanh","softmax"])
	# mlp.fit(X_train,y_train,0.1,10000,CrossEntropy,batch_size=None)
	# '''

	#例子二，线性回归的例子，波士顿房价预测
	import sklearn.datasets as datasets
	boston = datasets.load_boston()
	#构建均方误差损失函数
	class MSE(LossFunc):
		def calLoss(y,y_h):
			return np.mean(np.sum((y - y_h)**2,axis=0))
		def calDeriv(y,y_h):
			return 2*(y-y_h)
	mlp = MLP([13,5,5,1],["relu","relu","noactiv"])
	mlp.fit(boston.data,boston.target[:,np.newaxis],0.01,3000,MSE,batch_size=None)
