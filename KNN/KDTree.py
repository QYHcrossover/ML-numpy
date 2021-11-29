import numpy as np

class TreeNode:
    def __init__(self,data=None,fi=None,fv=None,left=None,right=None):
        self.data = data
        self.fi = fi
        self.fv = fv
        self.left = left
        self.right = right

class KDTree:
    def buildTree(self,X,depth):
        n_size,n_feature = X.shape
        #递归终止条件
        if n_size == 1:
            tree = TreeNode(data=X[0])
            return tree

        fi = depth % n_feature
        argsort = np.argsort(X[:,fi])
        middle_idx = argsort[n_size // 2]
        left_idxs,right_idxs = argsort[:n_size//2],argsort[n_size//2+1:]

        fv = X[middle_idx,fi]
        data = X[middle_idx]
        left,right = None,None
        if len(left_idxs) > 0:
            left = self.buildTree(X[left_idxs],depth+1)
        if len(right_idxs) > 0:
            right = self.buildTree(X[right_idxs],depth+1)
        tree = TreeNode(data,fi,fv,left,right)
        return tree
    
    def fit(self,X):
        self.tree = self.buildTree(X,0)
        
    @staticmethod
    def distance(a,b):
        return np.sqrt(((a-b)**2).sum())
    
    def find_nearest(self,x):
        nearest_point = self.tree.data
        nearest_dis = KDTree.distance(self.tree.data,x)
        def travel(kdtree,x):
            nonlocal nearest_dis,nearest_point
            if kdtree == None:
                return

            #如果根节点到目标点的距离小于最近距离，则更新nearest_point和nearest_dis
            if KDTree.distance(kdtree.data,x) < nearest_dis:
                nearest_dis = KDTree.distance(kdtree.data,x)
                nearest_point = kdtree.data

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
        return nearest_point,nearest_dis

if __name__ == "__main__":
    X = np.array([(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)])
    x = np.array([2,4.5])

    kdtree = KDTree()
    kdtree.fit(X)
    nearest_point,nearest_dis = kdtree.find_nearest(x)
    print(nearest_point)
    print(nearest_dis)