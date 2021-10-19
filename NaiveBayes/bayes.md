## 朴素贝叶斯概念

![image-20211013093849744](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211013093851.png)

### 参数估计

![image-20211013093906900](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20211013093908.png)

## 代码实现

```python
class Bayes:    
    @staticmethod
    def mean(X):
        return np.mean(X,axis=0)
    
    @staticmethod
    def variance(X):
        return np.mean((X-Bayes.mean(X))**2,axis=0)
    
    def gaussian(self,x,avg,var): 
        return (1./np.sqrt(2*math.pi*var)) * np.exp(-0.5*((x-avg)**2)/var)
    
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
```

