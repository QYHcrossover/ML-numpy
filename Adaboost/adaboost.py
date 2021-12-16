from sklearn.datasets import load_breast_cancer
from sklearn.model_selection  import train_test_split
import numpy as np
from tqdm import tqdm

# 构建Adaboost类
class MyAdaboost:
    def __init__(self,n_estimators):
        self.n_estimators = n_estimators
        self.clfs = [lambda x:0 for i in range(self.n_estimators)]
        self.alphas = [0 for i in range(self.n_estimators)]
        self.weights = None
        
    # 构造弱分类器的决策函数g(X)
    def _G(self,fi,fv,direct):
        assert direct in ["positive","nagetive"]
        def _g(X):
            if direct  == "positive":
                predict = (X[:,fi] <= fv) * -1 # which <= value assign -1 else 0
            else:
                predict = (X[:,fi] > fv) * -1 # which > value assign 0 else -1
            predict[predict == 0] = 1
            return predict 
        return _g
    
    #选择最佳的划分点,即求出fi和fv
    def _best_split(self,X,y,w):
        best_err = 1e10
        best_fi = None
        best_fv = None
        best_direct = None
        for fi in range(X.shape[1]):
            series = X[:,fi]
            for fv in np.sort(series):
                predict = np.zeros_like(series,dtype=np.int32)
                # direct = postive
                predict[series <= fv] = -1
                predict[series > fv] = 1
                err = np.sum((predict != y)* 1 * w)
#                 print("err = {} ,fi={},fv={},direct={}".format(err,fi,fv,"postive"))
                if err < best_err:
                    best_err = err
                    best_fi = fi
                    best_fv = fv
                    best_direct = "positive"

                # direct = nagetive
                predict = predict * -1
                err = np.sum((predict != y) * 1 * w)
                if err < best_err:
                    best_err = err
                    best_fi = fi
                    best_fv = fv
                    best_direct = "nagetive"
#                 print("err = {} ,fi={},fv={},direct={}".format(err,fi,fv,"nagetive"))
        return best_err,best_fi,best_fv,best_direct
    
    def fit(self,X_train,y_train):
        self.weights = np.ones_like(y_train) / len(y_train)
        for i in tqdm(range(self.n_estimators)):
            err,fi,fv,direct = self._best_split(X_train,y_train,self.weights)
#             print(i,err,fi,fv,direct)
            
            #计算G(x)的系数alpha
            alpha = 0.5 * np.log((1-err)/err) if err !=0 else 1
#             print("alpha:",alpha)
            self.alphas[i] = alpha
            
            #求出G
            g = self._G(fi,fv,direct)
            self.clfs[i] = g
            
            if err == 0: break
            
            #更新weights
            self.weights = self.weights * np.exp(-1 * alpha * y_train * g(X_train))
            self.weights = self.weights / np.sum(self.weights)
#             print("weights :",self.weights)
    
    def predict(self,X_test):
        y_p = np.array([self.alphas[i] * self.clfs[i](X_test) for i in range(self.n_estimators)])
        y_p = np.sum(y_p,axis=0)
        y_predict = np.zeros_like(y_p,dtype=np.int32)
        y_predict[y_p>=0] = 1
        y_predict[y_p<0] = -1
        return y_predict
    
    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict == y_test)/len(y_predict)

if __name__ == "__main__":
	breast_cancer = load_breast_cancer()
	X = breast_cancer.data
	y = breast_cancer.target
	y[y==0] = -1

	# 划分数据
	X_train,X_test,y_train,y_test = train_test_split(X,y)
	print(X_train.shape,X_test.shape)

	clf = MyAdaboost(100)
	clf.fit(X_train,y_train)
	print(clf.score(X_test,y_test))