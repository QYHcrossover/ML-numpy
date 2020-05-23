import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

n_dots = 200
x = np.linspace(-2*np.pi,2*np.pi,n_dots)
y = np.sin(x) + 0.2 * np.random.rand(n_dots) - 0.1
x = x.reshape(-1,1)
y = y.reshape(-1,1)

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    return pipeline

degrees = [2,3,5,10]
results = []
for i in degrees:
    model = polynomial_model(degree=i)
    model.fit(x,y)
    score = model.score(x,y)
    mse = mean_squared_error(y,model.predict(x))
    results.append({"model":model,"degree":i,"score":score,"mse":mse})

for i in range(len(results)):
    print("lr_degree={};score={};mse={}".format(results[i]["degree"],results[i]["score"],results[i]["mse"]))

fig=plt.figure(figsize=(16,10),dpi=200)
for i in range(len(results)):
    plt.subplot(2,2,i+1)
    plt.scatter(x,y,c="orange",s=5)
    plt.plot(x,results[i]["model"].predict(x))
plt.show()

