#针对原版本做了改进
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets

class LogisticRegression:
    def __init__(self,lr=0.001,thr=1e-2,max_epoch=100000): #学习率和阈值
        self.lr = lr
        self.thr = thr
        self.max_epoch = max_epoch
        self.best_loss = np.inf
        self.best_theta = None

    def predict(self,X):
#         sigmoid = lambda x: 1/(1+np.exp(-x))
        #计算结果
        T = X@self.theta
        data = np.array(T,copy=True)
        pos_index = data>=0
        neg_index = data<0
        pos = 1.0/(1+np.exp(-data))
        data[pos_index] = pos[pos_index]
        neg = np.exp(data)/(1+np.exp(data))
        data[neg_index] = neg[neg_index]
#         print("data",data)
        return data

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
            loss = 1/X.shape[0]*np.squeeze(-1*y.T@np.log(self.predict(paddedX)+1e-6) + -1*(1-y).T@np.log(1-self.predict(paddedX)+1e-6))
            if loss<self.best_loss:
                self.best_loss = loss
                self.best_theta = self.theta
            print("第{}epoch,loss为{}".format(epoch,loss))
            #计算梯度
            grad = paddedX.T@(self.predict(paddedX)-y)
#             print(grad)
            #是否收敛
            if abs(np.sum(grad)) < self.thr or epoch>self.max_epoch:
                break
            #梯度下降
            self.theta -= self.lr * grad
            epoch += 1
    
    def accuracy(self,test_X ,test_y):
        self.theta = self.best_theta
        paddedX = np.ones([test_X.shape[0],test_X.shape[1]+1])
        paddedX[:,:-1] = test_X
        predict_y = np.squeeze(self.predict(paddedX).astype(np.int))
        return np.sum(predict_y==test_y)/len(test_y)

if __name__ == "__main__":
	#导入数据集
	dataset = datasets.load_breast_cancer()
	X,y = dataset.data,dataset.target

	#数据标准化
	X_std = (X-X.mean(axis=0)) / X.std(axis=0)
	print(X_std)

	#划分为训练集和测试集
	import random
	random_index = list(range(len(y)))
	random.shuffle(random_index)
	train_index = random_index[:-100]
	test_index = random_index[-100:]
	train_X,train_y = X[train_index,:],y[train_index]
	print("X_train shape:",train_X.shape,"train_y shape:",train_y.shape)
	test_X,test_y = X[test_index,:],y[test_index]
	print("test_X shape:",test_X.shape,"test_y shape:",test_y.shape)

	#逻辑回归训练
	lr = LogisticRegression(lr=0.01,max_epoch=100000)
	lr.fit(train_X,train_y[:,np.newaxis])

	print("best loss:{}".format(lr.best_loss))
	print("ACC:{}".format(lr.accuracy(test_X,test_y)))