import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNNclassifier:
    def __init__(self,k):
        assert k>=1,"k必须是大于1的整数"
        self.k=k
        self.X_train=None
        self.y_train=None

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        return self

    def _predict(self,X):
        distance = [np.sum((X_i-X)**2) for X_i in self.X_train]
        arg = np.argsort(distance)
        top_k = [self.y_train[i] for i in arg[:self.k]]
        # c=Counter(top_k)
        # return c.most_common(1)[0][0]
        counter={}
        for i in top_k:
            counter.setdefault(i,0)
            counter[i]+=1
        sort=sorted(counter.items(),key=lambda x:x[1])
        return sort[0][0]

    def predict(self, X_test):
        y_predict = [self._predict(i) for i in X_test]
        return np.array(y_predict)

    def score(self, X_test ,y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict==y_test)/len(X_test)

if __name__ ==  "__main__":
    iris = datasets.load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=10,random_state=22)
    my_KNN=KNNclassifier(6)
    my_KNN.fit(X_train,y_train)
    y_predict=my_KNN.predict(X_test)
    print("y_test:","\n",y_test)
    print("y_predict","\n",y_predict)
    print("accuracy:",my_KNN.accuracy(X_test, y_test))
    print("accuracy:",accuracy_score(y_test,y_predict))

