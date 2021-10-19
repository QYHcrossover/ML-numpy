from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

class ID3DecisionTree:
    @staticmethod
    def entropy(y):
        precs = np.array(list(Counter(y).values()))/len(y)
        ent = np.sum(-1 * precs * np.log(precs))
        return ent
    
    def decide_feature(self,X,y,feature_order):
        n_features = X.shape[-1]
        ents = (feature_order != -1).astype(np.float64)
        for i in range(n_features):
            if feature_order[i] >= 0:
                continue
            for feature,size in Counter(X[:,i]).items():
                index = (X[:,i] == feature)
                splity = y[index]
                ent = ID3DecisionTree.entropy(splity)
                ents[i] += ent*size/len(X)
        fi = np.argmin(ents)
        return fi,ents[fi]
    
    def build_tree(self,X,y,feature_order):
        curent = ID3DecisionTree.entropy(y)
        counts = dict(Counter(y))
        if len(counts) == 1 or min(feature_order) == 0:
            result = max(counts,key=counts.get)
            return {"counts":counts,"result":result}
        fi,ent = self.decide_feature(X,y,feature_order)
        feature_order[fi] = max(feature_order)+1 
        result = None
        next_ = {}
        for value,_ in Counter(X[:,fi]).items():
            next_[value] = self.build_tree(X[X[:,fi]==value],y[X[:,fi]==value],feature_order)
        return {"feature":fi,"entgain":curent-ent,"counts":counts,"result":result,"next":next_}
    
    def fit(self,X,y):
        feature_order = -1 * np.ones(X.shape[-1])
        self.tree = self.build_tree(X,y,feature_order)
        
    def predict(self,X):
        y = []
        for i in range(len(X)):
            x_test = X[i]
            tree = self.tree
            while tree["result"] == None:
                feature = tree["feature"]
                nexttree = tree["next"][x_test[feature]]
                tree = nexttree
            y.append(tree["result"])
        return y

if __name__ == "__main__":
	def create_data():
	    datasets = [['青年', '否', '否', '一般', '否'],
	               ['青年', '否', '否', '好', '否'],
	               ['青年', '是', '否', '好', '是'],
	               ['青年', '是', '是', '一般', '是'],
	               ['青年', '否', '否', '一般', '否'],
	               ['中年', '否', '否', '一般', '否'],
	               ['中年', '否', '否', '好', '否'],
	               ['中年', '是', '是', '好', '是'],
	               ['中年', '否', '是', '非常好', '是'],
	               ['中年', '否', '是', '非常好', '是'],
	               ['老年', '否', '是', '非常好', '是'],
	               ['老年', '否', '是', '好', '是'],
	               ['老年', '是', '否', '好', '是'],
	               ['老年', '是', '否', '非常好', '是'],
	               ['老年', '否', '否', '一般', '否'],
	               ]
	    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
	    # 返回数据集和每个维度的名称
	    return datasets, labels
	
	dataset,columns  = create_data()
	X,y = np.array(dataset)[:,:-1],np.array(dataset)[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

	dt = ID3DecisionTree()
	dt.fit(X_train,y_train)
	print(dt.tree)

	print(dt.predict(X_test))