from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def gen_data(mean,std,size):
    data = np.array(mean) + np.random.randn(size,2)*std
    return data

X1 = gen_data(mean=(6,7),std=1,size=200)
y1 = np.zeros(X1.shape[0])
X2 = gen_data(mean=(3,3),std=1,size=200)
y2 = np.zeros(X2.shape[0])
X = np.vstack([X1,X2])
y = np.hstack([y1,y2])
print(X.shape,y.shape)

class MeanShift:
    def __init__(self,bandwidth,eps):
        self.bandwidth = bandwidth #小领域的距离长度
        self.cluster_locs = [] #每类中心点坐标
        self.visit_times = [] #每个样本点被各类访问次数
        self.eps = eps
    
    def fit(self,X):
        size = X.shape[0]
        is_visit = np.zeros(size,dtype=np.bool)
        while min(is_visit) == False: #所有数据点均访问过则结束
            #随机选择一个未访问的数据
            idx = np.random.choice(np.where(is_visit!=True)[0]) 
            x = X[idx]
            #此类中各个数据点被访问的次数
            cur_times = np.zeros(size,dtype=np.int)
            while True:
                #计算距离
                dis = np.sqrt(np.sum((X-x)**2,axis=1))
                #更新访问次数和是否访问过
                index = dis < self.bandwidth
                cur_times[index] += 1
                is_visit[index] = True
                #计算mean_shift
                mean_shift = np.mean((X-x)[dis < self.bandwidth],axis=0)
                if np.sum(mean_shift**2)< self.eps:
                    break
                x = x+mean_shift
            #类间合并
            has_merge = False
            for i in range(len(self.cluster_locs)):
                if np.sum((self.cluster_locs[i]-x)**2)<self.eps:
                    self.visit_times[i] += cur_times
                    has_merge = True
                    break
            if has_merge:
                self.cluster_locs.append(x)

ms = MeanShift(1.5,0.001)
ms.fit(X)