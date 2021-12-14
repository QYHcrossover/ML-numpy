import numpy as np 
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pla import PLA

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
	X,y = make_blobs(200,2,2,random_state=222)
	plt.scatter(X[y==0][:,0],X[y==0][:,1],color="red")
	plt.scatter(X[y==1][:,0],X[y==1][:,1],color="blue")
	
	y[y==0] = -1
	svc = SVM()
	svc.fit(X,y.reshape(-1,1))

	k_svc = - svc.W[0][0] / svc.W[1][0]
	b_svc = - svc.b[0] / svc.W[1][0]

	pla = PLA()
	pla.fit(X,y.reshape(-1,1))
	k_pla = - pla.theta[1][0] / pla.theta[2][0]
	b_pla = - pla.theta[0][0] / pla.theta[2][0]

	# minx,maxx = np.min(X,axis=0)[0],np.max(X,axis=0)[0]
	# print(minx,maxx)
	plt.plot([3.8,3.98],[k_svc*3.8+b_svc,k_svc*3.98+b_svc],"m-",label="svc")
	plt.plot([0.3,6],[k_pla*0.3+b_pla,k_pla*6+b_pla],"y-",label="pla")
	plt.legend()
	plt.show()
