from sklearn.datasets import load_breast_cancer
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

# 整合所有的东西
class CartDT:
    @staticmethod
    def gini(y):
        if type(y) == list or type(y) == np.ndarray:
            y = dict(Counter(y))
        precs = np.array(list(y.values())) / sum(y.values())
        return 1-np.sum(precs**2)
    
    def best_split(self,X,y):
        best_gini = 1e10
        best_fi = None
        best_fv = None
        for fi in range(X.shape[1]):
            #该特征仅有一个取值无法再分
            if len(set(X[:,fi])) == 1:
                continue
            for fv in sorted(set(X[:,fi]))[:-1]:
                y_left = y[X[:,fi] <= fv]
                gini_left = CartDT.gini(y_left)
                y_right = y[X[:,fi] > fv]
                gini_right = CartDT.gini(y_right)
                gini = len(y_left)/len(y)*gini_left + len(y_right)/len(y)*gini_right
    #             print(f"fi={fi:.2f} fv={fv:.2f} gini={gini:.2f}")
                if gini < best_gini:
                    best_gini = gini
                    best_fi = fi
                    best_fv = fv
        return best_gini,best_fi,best_fv

    def build_tree(self,X,y):
        #叶子节点的条件1,仅有一个类别
        counts = dict(Counter(y))
        result = max(counts,key=counts.get)
        if len(counts) == 1:
            return {"counts":counts,"result":result}

        #叶子节点的条件2，所有特征仅有一个取值
        fcs = [len(Counter(X[:,fi])) for fi in range(X.shape[-1])]
        if sum(fcs) == X.shape[-1]:
            return {"counts":counts,"result":result}

        gini,fi,fv = self.best_split(X,y)
        index_left,index_right = X[:,fi]<=fv,X[:,fi]>fv
        left = self.build_tree(X[index_left],y[index_left])
        right = self.build_tree(X[index_right],y[index_right])
        return {"counts":counts,"result":None,"left":left,"right":right,"fi":fi,"fv":fv}
    
    def fit(self,X,y):
        self.tree = self.build_tree(X,y)
    
    def _C(self,tree):
        leafs = []
        count = 0
        def dfs(tree):
            nonlocal leafs,count
            count += 1
            if tree["result"] != None:
                leafs.append(tree["counts"])
                return
            dfs(tree["left"])
            dfs(tree["right"])
            return
        dfs(tree)
        percs = np.array([sum(leaf.values()) for leaf in leafs])
        percs = percs / percs.sum()
        ginis = np.array([CartDT.gini(leaf) for leaf in leafs])
        c = np.sum(percs * ginis)
        return c,count
    
    def _add_alpha(self,tree):
        if tree["result"] != None:
            return tree
        gini_one = CartDT.gini(tree["counts"])
        gini_whole,counts = self._C(tree)
        alpha = (gini_one - gini_whole)/(counts-1)
        self.alphas.append(alpha)
        tree["alpha"] = alpha
        tree["left"] = self._add_alpha(tree["left"])
        tree["right"] = self._add_alpha(tree["right"])
        return tree
    
    def _inactivity(self,tree,alpha):
        if tree["result"] != None:
            return tree
        if tree["alpha"] <= alpha:
            tree["result"] = max(tree["counts"],key=tree["counts"].get)
        tree["left"] = self._inactivity(tree["left"],alpha)
        tree["right"] = self._inactivity(tree["right"],alpha)
        return tree
    
    def post_pruning(self):
        self.alphas = []
        self.tree = self._add_alpha(self.tree)
        self.subtrees = [self.tree.copy() for _ in range(len(set(self.alphas)))]
        for i,alpha in enumerate(sorted(set(self.alphas))):
            self.subtrees[i] = self._inactivity(self.subtrees[i],alpha)
        
    def _predict(self,X,tree):
        y_pred = []
        for x in X:
            cur = tree
            while cur["result"] == None:
                fi,fv = cur["fi"],cur["fv"]
                cur = cur["left"] if x[fi] <= fv else cur["right"]
            y_pred.append(cur["result"])
        return np.array(y_pred)
    
    def _score(self,X,y,tree):
        return np.sum(self._predict(X,tree)==y) / len(y)
    
    def predict(self,X):
        return self._predict(X,self.tree)
    
    def score(self,X,y):
        return np.sum(self.predict(X)==y) / len(y)

if __name__ == "__main__":
    data = load_breast_cancer()
    X,y = data["data"],data["target"]
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    cart_clf = CartDT()
    cart_clf.fit(X_train,y_train)
    print(cart_clf.score(X_test,y_test))
    cart_clf.post_pruning()
    for subtree in cart_clf.subtrees:
        print(cart_clf._score(X_test,y_test,subtree))