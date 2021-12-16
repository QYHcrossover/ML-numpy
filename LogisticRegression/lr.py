import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.001,thr=1e-2,max_epoch=600): #学习率和阈值
        self.lr = lr
        self.thr = thr
        self.max_epoch = max_epoch
        self.best_loss = np.inf
        self.best_theta = None
        self.losses_process = []

    def _predict(self,X,theta):
        sigmoid = lambda x: 1/(1+np.exp(-x)) if x>=0 else np.exp(x)/(1+np.exp(x))
        pX = np.ones([X.shape[0],X.shape[1]+1])
        pX[:,:-1] = X
        preds = [sigmoid(i) for i in pX@theta.ravel()]
        return np.array(preds).reshape(-1,1)
    
    def predict(self,X):
        preds = self._predict(X,self.best_theta).ravel()
        preds[preds<=0.5] = 0
        preds[preds>0.5] = 1
        return preds
    
    def score(self,X,y):
        return np.sum(self.predict(X)==y) / len(y)
        
    def fit(self,X,y):
        y = y.reshape(-1,1)
        pX = np.ones([X.shape[0],X.shape[1]+1])
        pX[:,:-1] = X
        
        #构造theta
        theta = np.random.randn(X.shape[-1]+1).reshape(-1,1)      
        #开始训练
        epoch = 0
        while True:
            #计算loss
            loss = -1/X.shape[0]*np.squeeze(y.T@np.log(self._predict(X,theta)+1e-6) + (1-y).T@np.log(1-self._predict(X,theta)+1e-6))
            self.losses_process.append(loss)
            if loss<self.best_loss:
                self.best_loss = loss
                self.best_theta = theta
#             print("第{}epoch,loss为{}".format(epoch,loss))
            #计算梯度
            grad = pX.T@(self._predict(X,theta)-y)
            #是否收敛
            if np.sum(np.abs(grad)) < self.thr or epoch>self.max_epoch: break
            #梯度下降
            theta -= self.lr * grad
            epoch += 1

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection  import train_test_split
    import matplotlib.pyplot as plt

    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    #数据标准化
    X_std = (X-X.mean(axis=0)) / X.std(axis=0)
    #划分为训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    #逻辑回归训练
    lr = LogisticRegression(max_epoch=1000)
    lr.fit(X_train,y_train)
    print(f"train score {lr.score(X_train,y_train):.2f}")
    print(f"test  score {lr.score(X_test, y_test):.2f}")

    plt.plot(lr.losses_process)
