仿照“sklearn"的API格式，自己写了一个小型的”sklearn“。

## 基本用法

```python
# 模型训练
clf.fit(X_train,y_train)
# 模型预测
clf.predict(X)
# 测试集score
clf.score(X_test,y_test)
```

## 算法列表


|算法 | 代码实现 | 博客 |
| --- | --- | --- |
| KNN | 排序法实现：[knn.py](./KNN/knn.py)<br>KDTree实现:  [KDTree.ipynb](./KNN/KDTree.ipynb) | [KNN原理及代码实现——以irs为例](https://zhuanlan.zhihu.com/p/611231868)<br>[KDTree实现KNN算法](https://zhuanlan.zhihu.com/p/611301589) |
| PLA | 详见[pla.py](./PLA) | [感知机原理及代码实现](https://zhuanlan.zhihu.com/p/611500346) |
| Logistic Regression | 详见[lr.py](./LogisticRegression/lr.py) | [逻辑回归算法原理以及代码实现](https://zhuanlan.zhihu.com/p/611305829) |
| Decision Tree | ID3实现: [MyDecisionTree.ipynb](DecisionTree/MyDecisionTree.ipynb)<br>C4.5实现: [C4.5DecisionTree.ipynb](DecisionTree/C4.5DecisionTree.ipynb)<br>Cart实现: [CART-DT.ipynb](DecisionTree/CART-DT.ipynb) | [决策树算法原理以及ID3算法代码实现](https://zhuanlan.zhihu.com/p/611309188)<br>[决策树算法原理以及cart代码实现](https://zhuanlan.zhihu.com/p/611311582) |
| Naive Bayes | [mybayes.ipynb](./NaiveBayes/mybayes.ipynb) | [朴素贝叶斯原理以及代码实现](https://zhuanlan.zhihu.com/p/611304041) |
| SVM | 采用cvxopt工具求解: [SVC.ipynb](./SVM/SVC.ipynb)  [svm.py](./SVM/svm.py)<br>SMO算法实现：[SMO.ipynb](./SVM/SMO.ipynb) | [支持向量机原理以及代码实现](https://zhuanlan.zhihu.com/p/611475806)<br>[支持向量机SMO代码实现](https://zhuanlan.zhihu.com/p/611483233) |
| MLP | 支持自定层数，自定义损失函数等特性：[mlp](./MLP) | [多层感知机及代码实现](https://zhuanlan.zhihu.com/p/611500502) |
| Adaboost | 元模型为某个特征上”切一刀“，详见 [myAdaboost.ipynb](./Adaboost/myAdaboost.ipynb) | [Adaboost原理以及代码实现](https://zhuanlan.zhihu.com/p/611312201) |
| CNN | [cnn.py](./CNN) |  |
| Linear Regression | 详见 [linear regression](./LinearRegression) | [线性回归原理以及代码实现](https://zhuanlan.zhihu.com/p/611485530) |
| PCA | 详见 [PCA](./PCA) | [PCA算法以及代码实现](https://zhuanlan.zhihu.com/p/611486901) |


## 一些例子

### breast-cancer数据集

```python
def create_clfs():
    clfs = {
        "pla":PLA(),
        "lr":LogisticRegression(),
        "knn":KNNclassifier(5),
        "bayes":Bayes(),
        "cartdt":CartDT(),
        "adaboost":MyAdaboost(100),
        "svm-linear":SVM(kind="linear"),
        "svm-gaussian":SVM(kind="gaussian")
    }
    return clfs
clfs = create_clfs()
results = {"name":[],"train-score":[],"test-score":[]}
for name,clf in clfs.items():
    # logistic—regression 的 label 是 0和1，其余都是-1和1
    (train_y,test_y) = (y_train,y_test) if name in ["lr","bayes"] else (y_train_neg,y_test_neg)
    clf.fit(X_train,train_y)
    train_score = clf.score(X_train,train_y)
    test_score = clf.score(X_test,test_y)
    results["name"].append(name)
    results["train-score"].append(train_score)
    results["test-score"].append(test_score)
    print(f"{name}-{train_score:.2f}-{test_score:.2f}")
```

结果

![image-20211216220606457](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/image-20211216220606457.png)

详见: [summary-test.ipynb](summary-test.ipynb)

### 二维数据模型效果可视化



![image-20211216220606457](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/1_1.png)

![image-20211216220606457](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/1_2png.png)

详见:[summary-test.ipynb](summary-test.ipynb)

