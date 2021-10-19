import numpy as np

class Bayes:    
    @staticmethod
    def mean(X):
        return np.mean(X,axis=0)
    
    @staticmethod
    def variance(X):
        return np.mean((X-Bayes.mean(X))**2,axis=0)
    
    def gaussian(self,x,avg,var): 
        return (1./np.sqrt(2*np.pi*var)) * np.exp(-0.5*((x-avg)**2)/var)
    
    def fit(self,X,y):
        c = len(set(y))
        self.c = c
        Xs = [X[y==i] for i in range(c)]
        
        #各个类别的均值、方差、所占比率
        self.avgs = [Bayes.mean(X) for X in Xs]
        self.vars = [Bayes.variance(X) for X in Xs]
        self.percs = [len(y[y==i])/len(y) for i in range(c)]
        
    def predict(self,x):
        if len(x.shape) == 1:
            result = np.array(self.percs)
            for i in range(self.c):
                gaus = self.gaussian(x,self.avgs[i],self.vars[i])
                for j in range(len(x)):
                    result[i] *= gaus[j]
            return np.argmax(result)
        results = np.array([self.predict(x[i]) for i in range(len(x))])
        return results
    
    def score(self,X,y):
        y_pred = self.predict(X)
        return np.sum(y_pred==y)/len(y)

if __name__=="__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X,y = iris["data"],iris["target"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=22)
    # print(len(X_train))
    # print(len(X_test))
    bayes = Bayes()
    bayes.fit(X_train,y_train)
    # print(bayes.avgs)
    # print(bayes.vars)
    print(bayes.score(X_test,y_test))

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))