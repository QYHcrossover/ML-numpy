import numpy as np
from utils import *
import math

# relu函数的实现和求导
relu_forward = lambda x: 1*(x>0)*x
relu_backward = lambda x: 1*(x>0)

# 展开与折叠
flatten = lambda x: x.reshape([x.shape[0],-1])
flod = lambda x: x.reshape([x.shape[0],4,4,16])

# softmax函数
def softmax(x):
	result = np.exp(x)/np.sum(np.exp(x),axis=1)[:,np.newaxis]
	return result

# cross_entropy
def cross_entropy(x,y):
	return np.sum(-1*y*log(x),axis=1)

# softmax_with_crossEntropy_backward
def softmax_with_crossEntropy_backward(y,y_h):
	return y-y_h

# fc实现与求导
def fc_forward(A,W,b):
	cache = (A,W,b)
	return A@W.T+b.T,cache

def fc_backward(dZ,cache):
	A,W,b = cache
	dA = dZ @ W
	dW = 1/A.shape[0]*dZ.T @ A
	db = np.mean(dZ,axis=1)
	return dA,dW,db


class CNN:
	def __init__(self):
		#定义卷积层和全连接的权重和偏置
		self.conv1_W = 0.25*np.random.normal(size=[5,5,1,8])
		self.conv1_b = np.zeros([1,1,1,8])
		self.conv2_W = 0.25*np.random.normal(size=[5,5,8,16])
		self.conv2_b = np.zeros([1,1,1,16])
		self.fc1_W = 0.25*np.random.normal(size=[128,256])
		self.fc1_b = np.zeros([128,1])
		self.fc2_W = 0.25*np.random.normal(size=[10,128])
		self.fc2_b = np.zeros([10,1])

		#pad方式为valid，池化层为max_pool间隔为2
		self.hparameters_conv = {"stride":1,"pad":0}
		self.hparameters_pool = {"stride":2,"f":2}
	
	def forward(self,X):
		conv1 , cache_conv1 = conv_forward(X,self.conv1_W,self.conv1_b,self.hparameters_conv)
		relu1 = relu_forward(conv1)
		pool1 , cache_pool1 = pool_forward(relu1,self.hparameters_pool,mode="max")
		conv2 , cache_conv2 = conv_forward(pool1,self.conv2_W,self.conv2_b,self.hparameters_conv)
		relu2 = relu_forward(conv2)
		pool2 , cache_pool2 = pool_forward(relu2,self.hparameters_pool,mode="max")
		flatn = flatten(pool2)
		fc1 , cache_fc1 = fc_forward(flatn,self.fc1_W,self.fc1_b)
		relu3 = relu_forward(fc1)
		fc2 , cache_fc2= fc_forward(relu3,self.fc2_W,self.fc2_b)
		softx = softmax(fc2)
		cache = (cache_conv1,cache_pool1,cache_conv2,cache_pool2,cache_fc1,cache_fc2)
		return softx,cache

	def backward(self,X,y_h,lr):
		#前向传播
		y,cache = self.forward(X)
		(cache_conv1,cache_pool1,cache_conv2,cache_pool2,cache_fc1,cache_fc2) = cache
		#计算loss
		loss = np.mean(cross_entropy(y,y_h))
		#反向传播
		d_fc2 = softmax_with_crossEntropy_backward(y,y_h)
		d_relu3,d_fc2_W,d_fc2_b = fc_backward(d_fc2,cache_fc2)
		d_fc1 = relu_backward(d_relu3)
		d_flatn,d_fc1_W,d_fc1_b = fc_backward(d_fc1,cache_fc1)
		d_pool2 = flod(d_flatn)
		d_relu2 = pool_backward(d_pool2,cache_pool2)
		d_conv2 = relu_backward(d_relu2)
		d_pool1,d_conv2_W,d_conv2_b = conv_backward(d_conv2,cache_conv2)
		d_relu1 = pool_backward(d_pool1,cache_pool1)
		d_conv1 = relu_backward(d_relu1)
		dX,d_conv1_W,d_conv1_b = conv_backward(d_conv1,cache_conv1)
		#梯度下降
		self.conv1_W -= lr*d_conv1_W
		self.conv1_b -= lr*d_conv1_b
		self.conv2_W -= lr*d_conv2_W
		self.conv2_b -= lr*d_conv2_b
		self.fc1_W -= lr*d_fc1_W
		self.fc1_b -= lr*d_fc1_b
		self.fc2_W -= lr*d_fc2_W
		self.fc2_b -= lr*d_fc2_b
		return loss

	def fit(self,X,y,lr,max_iters,batch_size=8):
		amount = X.shape[0]
		for epoch in range(max_iters):
	 		#batch梯度下降或minibatch梯度下降
	 		loss_avg = 0
 			for i in range(math.ceil(amount/batch_size)):
 				loss = self.backward(X[:,i*batch_size:i*batch_size+batch_size],y[:,i*batch_size:i*batch_size+batch_size],lr)
 				loss_avg = (i/(i+1))*loss_avg +(1/(i+1))*loss
	 		print("第{}轮训练，loss大小为{}".format(epoch,loss_avg))

if __name__ == "__main__":
	X_train,y_train,X_test,y_test = load_mnist("./data")
	X_train = X_train.reshape([X_train.shape[0],28,28,1])*(1/255)
	y_train = labelBinarizer(y_train)

	cnn = CNN()
	cnn.fit(X_train,y_train,0.01,100)