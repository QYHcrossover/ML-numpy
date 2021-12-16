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

### 监督学习-分类算法

- **KNN** K最近邻算法
  - 排序法实现：[knn.py](./KNN/knn.py)
  - KDTree实现:  [KDTree.ipynb](./KNN/KDTree.ipynb)
- **PLA** 感知机模型
  - 详见: [pla.py](./PLA)
- **logistic regression** 逻辑回归算法
  - 梯度下降法实现: [lr.py](./LogisticRegression/lr.py)
- **decision tree** 决策树算法
  - ID3实现: [MyDecisionTree.ipynb](DecisionTree/MyDecisionTree.ipynb)
  - C4.5实现: [C4.5DecisionTree.ipynb](DecisionTree/C4.5DecisionTree.ipynb)
  - Cart实现: [CART-DT.ipynb](DecisionTree/CART-DT.ipynb)

- **Naive Bayes** 朴素贝叶斯算法

  - 针对连续变量采用正态分布作为参数估计:  [mybayes.ipynb](./NaiveBayes/mybayes.ipynb)

- **SVM** 支持向量机

  - 采用cvxopt工具求解: [SVC.ipynb](./SVM/SVC.ipynb)  [svm.py](./SVM/svm.py)

  - SMO算法实现：[SMO.ipynb](./SVM/SMO.ipynb)

- **MLP** 多层感知机

  - 支持自定层数，自定义损失函数等特性，详情： [mlp](./MLP)
  - 实现: [mlp.py](./MLP/mlp.py)

- **adaboost**
  - 元模型为某个特征上”切一刀“，详见 [myAdaboost.ipynb](./Adaboost/myAdaboost.ipynb)

- **CNN** 卷积神经网络
  - 详见: [cnn.py](./CNN)

### 监督学习-回归算法

- **linear regression** 线性回归
  - 详见: [linear regression](./LinearRegression)

### 无监督学习算法

- **PCA** 主城成分分析
  - 详见: [](./PCA)

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

![image-20211216220606457](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/1.png)

![image-20211216220606457](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed//blogimg/2.png)

详见:[summary-test.ipynb](summary-test.ipynb)

