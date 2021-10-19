import numpy as np

def kmeans(data,k):
    ci = np.random.choice(len(dataset),k,replace=False)
    centers = data[ci]
#     print("centers : {}".format(centers.shape))
    it = 0
    
    while True:
        #计算所有点到聚类中心的距离
        distances = np.hstack([np.sum((data-center)**2,axis=1)[:,np.newaxis] for center in centers])
        mink = np.argmin(distances,axis=1)

        #确定下一轮聚类中心
        newcenters = np.array([np.mean(data[mink==i],axis=0) for i in range(k)])
#         print("new centers :{}".format(newcenters.shape))

        #判断是否需要下次迭代
        delta = np.sum(np.abs(newcenters - centers))
        if delta < 1e-5 or it >10000:
            return centers,mink,it

        #下次迭代
        centers = newcenters
        it += 1

if __name__ == "__main__":
	#读取数据
	data = []
	with open("data.txt") as f:
	    for line in f:
	        x,y = line.strip().split()
	        data.append((float(x),float(y)))
	data = np.array(dataset)

	#kmeans聚类
	centers,index,it = kmeans(data,2)
	print(centers)
	print(index)
	print(it)