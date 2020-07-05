## 项目概况

自己纯用numpy搭建的神经网络

## 功能特性

1. **自定义神经网络结构**，网络结构以list传入MLP类中

2. 自定义激活函数，支持sigmoid、tanh、softmax、relu激活函数以及不激活noactiv；**支持每一层使用不同的激活函数**
3. **自定义损失函数**，继承LossFunc类，重写两个求loss和求导的函数；对于softmax特殊处理，一般和交叉熵结合起来直接求导
4. 支持**分类问题**和**回归问题**，具体使用方式见例子
5. 支持batch梯度下降和mini_batch梯度下降两种方式

## 使用例子

### mnist数据集

```python
#载入数据集
X_train = np.load(r"..\SVM\之前\X_train.npy")
y_train = np.load(r"..\SVM\之前\y_train.npy")
#将y_train进行one-hot编码
def one_hot(y,feature_size):
    LB = np.zeros([len(y), feature_size], dtype=np.int32)
    for i in range(len(y)):
        LB[i][y[i]] = 1
        return LB
y_train = one_hot(y_train,10)
#构建交叉熵损失函数
class CrossEntropy(LossFunc):
    def calLoss(y,y_h):
        return np.mean(np.sum(-1*y_h*np.log(y),axis=0))
#构建模型
mlp = MLP([784,50,50,10],["tanh","tanh","softmax"])
mlp.fit(X_train,y_train,0.1,10000,CrossEntropy,batch_size=None)
# '''
```

### 结果

![image-20200705154923802](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20200705155609.png)

### boston房价预测

```python
#例子二，线性回归的例子，波士顿房价预测
import sklearn.datasets as datasets
boston = datasets.load_boston()
#构建均方误差损失函数
class MSE(LossFunc):
	def calLoss(y,y_h):
		return np.mean(np.sum((y - y_h)**2,axis=0))
	def calDeriv(y,y_h):
		return 2*(y-y_h)
mlp = MLP([13,5,5,1],["relu","relu","noactiv"])
mlp.fit(boston.data,boston.target[:,np.newaxis],0.01,3000,MSE,batch_size=None)
```

### 结果

![image-20200705155320217](https://cdn.jsdelivr.net/gh/QYHcrossover/blog-imgbed/blogimg/20200705155616.png)