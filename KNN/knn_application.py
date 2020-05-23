import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


iris = datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
# indices=np.random.permutation(len(X))
# X_train = X[indices[:-10]]
# X_test =  X[indices[-10:]]
# y_train = y[indices[:-10]]
# y_test =  y[indices[-10:]]
knn = KNeighborsClassifier()
param_grid=[
    {
        "weights": ["uniform"],
        "n_neighbors":[i for i in range(1, 6)]
    },
    {
        "weights": ["distance"],
        "n_neighbors":[i for i in range(1, 6)],
        "p":[i for i in range(1, 6)]
    }
]
grid_search=GridSearchCV(knn, param_grid)
grid_search.fit(X_train,y_train)
print("best_estimator",grid_search.best_estimator_)
print("best_params",grid_search.best_params_)
print("best_score",grid_search.best_score_)

# print("y_test:",y_test)
# print("y_predict:",y_predict)
# print("accuracy:%s"%score)
# print("probility:%s"%probility)
# print("classification:",classification_report(y_test,y_predict,target_names=["class0","class1","class2"]))