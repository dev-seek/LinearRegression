# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# %%
bias = 100
X,y1,coef = make_regression(n_samples=200,n_features=1, noise=100 , coef="True" , random_state=42 , bias=bias)

# %%
y_a = X*coef+bias
y = y_a.reshape(-1,1)
coef

# %%
class LinearRegression:
    def __init__(self,lr):
        self.lr = lr
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.coef_ = np.random.random()
        self.intercept_ = np.random.random()
        error = []
        for i in range(50):
            self.gradient();
            error.append(self.error())
        return error
    
    def gradient(self,):
        coef_des , inter_des = self.gradient_des()
        self.coef_ -= coef_des*self.lr
        self.intercept_ -= inter_des*self.lr

    def gradient_des(self):
        yh = self.predict(x1 = self.X)
        coef_des = ((yh-self.y)*self.X).mean()
        inter_des = (yh-self.y).mean()
        return coef_des, inter_des
    def predict(self,x1):
        yh = x1*self.coef_+self.intercept_
        return yh
    def error(self):
        y_h = self.predict(x1=self.X)
        return ((y_h-self.y)**2).sum()

# %%
model = LinearRegression(0.1)
model.fit(X,y)
y_train = model.predict(X)


# %%
plt.scatter(X,y1)
plt.plot(X,y_a ,label = "y_actual")
plt.plot(X,y_train,label = "y_generated")
plt.legend()

# %%
model.coef_ , model.intercept_


