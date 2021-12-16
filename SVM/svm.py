import numpy as np
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from tqdm import tqdm
from sklearn.datasets import make_gaussian_quantiles


def plot_clf(X,y,cls):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = cls.predict(points).reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()

class SVM:
    def __init__(self,sigma=1,C=1,kind="linear"):
        assert kind in ["linear","gaussian"]
        self.sigma = sigma
        self.C = C
        gaussian = lambda x,z: np.exp(-0.5*np.sum((x-z)**2)/(self.sigma**2))
        linear = lambda x,z: np.sum(x*z)
        self.kernel = linear if kind == "linear" else gaussian
    
    def fit(self,X,y):
        mat = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i,X.shape[0]):
                result = self.kernel(X[i],X[j])
                mat[i,j] = result
                mat[j,i] = result
        P = mat * (y.reshape(-1,1) @ y.reshape(1,-1))
        q = -1*np.ones(X.shape[0]).reshape(-1,1)
        
        G = np.zeros((2*X.shape[0],X.shape[0]))
        G[0:X.shape[0]] = - np.identity(X.shape[0])
        G[X.shape[0]:] = np.identity(X.shape[0])
        h = np.zeros(2*X.shape[0])
        h[X.shape[0]:] = self.C
        h = h.reshape(-1,1)
        
        A = y.reshape(1,-1)
        b = np.zeros(1).reshape(-1,1)
        
        [P,q,G,h,A,b] = [matrix(i,i.shape,"d")for i in [P,q,G,h,A,b]]
        result = solvers.qp(P,q,G,h,A,b)
        self.A = np.array(result["x"])
        support_vector_index = np.where(self.A > 1e-4)[0]
        self.support_vectors = X[support_vector_index]
        self.support_vector_as = self.A[support_vector_index,0]
        self.support_vector_ys = y[support_vector_index]
        for i,a in enumerate(self.A):
            if a>0+1e-4 and a<self.C-1e-4:
                self.b = y[i] - np.sum(self.A.ravel()*y*mat[i])
                break
    
    def predict(self,X):
        preds = []
        for x in tqdm(X):
            Ks = [self.kernel(x,support_vector) for support_vector in self.support_vectors]
            pred = np.sum(self.support_vector_as * self.support_vector_ys * Ks) + self.b
            pred = 1 if pred >=0 else -1
            preds.append(pred)
        return np.array(preds)

    def score(self,X,y):
        return np.sum(self.predict(X)==y) / len(y)

if __name__ == "__main__":
    X, y = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=2, mean=[1,2],cov=2,random_state=222)
    y[y==0] = -1

    svc = SVM(kind="linear")
    svc.fit(X,y)
    plot_clf(X,y,svc)