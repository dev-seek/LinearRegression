# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# %%
bias=100
X,y,coef = make_regression(100,1,bias=bias,noise=30,random_state=42,coef="True")

# %%
y_gen = X*coef+bias
plt.figure(figsize=(5,5))
plt.scatter(X,y)
plt.plot(X,y_gen)

# %%
model = LinearRegression()
model.fit(X,y)
y_train = X*model.coef_+model.intercept_

# %%
model.coef_ , model.intercept_ 

# %%
plt.figure(figsize=(5,5))
plt.scatter(X,y)
plt.plot(X,y_gen)
plt.plot(X,y_train)


