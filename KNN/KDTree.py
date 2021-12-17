import numpy as np
from collections import Counter
from tqdm import tqdm

class TreeNode:
    def __init__(self,data=None,label=None,fi=None,fv=None,left=None,right=None):
        self.data = data
        self.label = label
        self.fi = fi
        self.fv = fv
        self.left = left
        self.right = right

class KDTreeKNN:
    def __init__(self,k=3):
        self.k = k
    
    def buildTree(self,X,y,depth):
        n_size,n_feature = X.shape
        #递归终止条件
        if n_size == 1:
            tree = TreeNode(data=X[0],label=y[0])
            return tree

        fi = depth % n_feature
        argsort = np.argsort(X[:,fi])
        middle_idx = argsort[n_size // 2]
        left_idxs,right_idxs = argsort[:n_size//2],argsort[n_size//2+1:]

        fv = X[middle_idx,fi]
        data,label = X[middle_idx],y[middle_idx]
        left,right = None,None
        if len(left_idxs) > 0:
            left = self.buildTree(X[left_idxs],y[left_idxs],depth+1)
        if len(right_idxs) > 0:
            right = self.buildTree(X[right_idxs],y[right_idxs],depth+1)
        tree = TreeNode(data,label,fi,fv,left,right)
        return tree
    
    def fit(self,X,y):
        self.tree = self.buildTree(X,y,0)
        
    def _predict(self,x):
        finded = []
        labels = []
        for i in range(self.k):
            nearest_point,nearest_dis,nearest_label = self.find_nearest(x,finded)
            finded.append(nearest_point)
            labels.append(nearest_label)
        
        counter={}
        for i in labels:
            counter.setdefault(i,0)
            counter[i]+=1
        sort=sorted(counter.items(),key=lambda x:x[1])
        return sort[0][0]
    
    def predict(self,X):
        return np.array([self._predict(x) for x in X])

    def score(self,X,y):
    	return np.sum(self.predict(X)==y) / len(y)
    
    def _isin(self,x,finded):
        for f in finded:
            if KDTreeKNN.distance(x,f) < 1e-6: return True
        return False
        
    @staticmethod
    def distance(a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def find_nearest(self,x,finded):
        nearest_point = None
        nearest_dis = np.inf
        nearest_label = None
        def travel(kdtree,x):
            nonlocal nearest_dis,nearest_point,nearest_label
            if kdtree == None:
                return

            #如果根节点到目标点的距离小于最近距离，则更新nearest_point和nearest_dis
            if KDTreeKNN.distance(kdtree.data,x) < nearest_dis and not self._isin(kdtree.data,finded) :
                nearest_dis = KDTreeKNN.distance(kdtree.data,x)
                nearest_point = kdtree.data
                nearest_label = kdtree.label

            if kdtree.fi == None or kdtree.fv == None:
                return

            #进入下一个相应的子节点
            if x[kdtree.fi] < kdtree.fv:
                travel(kdtree.left,x)
                if x[kdtree.fi] + nearest_dis > kdtree.fv:
                    travel(kdtree.right,x)
            elif x[kdtree.fi] > kdtree.fv:
                travel(kdtree.right,x)
                if x[kdtree.fi] - nearest_dis < kdtree.fv:
                    travel(kdtree.left,x)
            else:
                travel(kdtree.left,x)
                travel(kdtree.right,x)
        travel(self.tree,x)
        return nearest_point,nearest_dis,nearest_label