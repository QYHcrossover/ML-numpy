from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
centers =[[-2,2],[2,2],[0,4]]
c = np.array(centers)
X,y = make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.60)
print(X)
print(y)
from sklearn.neighbors import KNeighborsClassifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
X_sample=[0,2]
X_sample=np.atleast_2d(X_sample)
print(X_sample)
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample,return_distance=False)
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap="cool")
plt.scatter(c[:,0],c[:,1],c="k",marker="^")
plt.scatter(X_sample[:,0],X_sample[:,1],marker="x",c=y_sample,s=100,cmap="cool")
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[:,0]],[X[i][1],X_sample[:,1]],"k--",linewidth=0.6)
plt.show() 