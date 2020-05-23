import numpy as np 
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class SVM:
	def __init__(self):
		self.W = None
		self.b = None

	def fit(self,X,y):
		#计算P
		P = np.identity(X.shape[-1]+1)
		P[0,0] = 0
		#计算q
		q = np.zeros(X.shape[-1]+1).reshape(-1,1)
		#计算G
		G = np.ones([X.shape[0],X.shape[-1]+1])
		G[:,1:] = X
		G = - y * G
		#计算h
		h = - np.ones(X.shape[0]).reshape(-1,1)
		#批量转换
		P,q,G,h = matrix(P),matrix(q),matrix(G),matrix(h)
		sol = solvers.qp(P,q,G,h)

		self.b = np.array(sol["x"])[0]
		self.W = np.array(sol["x"])[1:]

if __name__ == "__main__":
	X,y = make_blobs(200,2,2)

	plt.scatter(X[y==0,0],X[y==0,1],y,color="red")
	plt.scatter(X[y==1,0],X[y==1,1],y,color="blue")
	
	y[y==0] = -1
	svc = SVM()
	svc.fit(X,y.reshape(-1,1))

	k = - svc.W[0][0] / svc.W[1][0]
	b = - svc.b[0] / svc.W[1][0]
	minx,maxx = np.min(X,axis=0)[0],np.max(X,axis=0)[0]
	print(minx,maxx)
	plt.plot([minx,maxx],[k*minx+b,k*maxx+b],"m-")
	plt.show()
