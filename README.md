# EX-03  Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard python libraries for Gradient design.<br>
2.Introduce the variables needed to execute the function.<br>
3.Use function for the representation of the graph.<br>
4.Using for loop apply the concept using the formulae.<br>
5.Execute the program and plot the graph. <br>
6.Predict and execute the values for the given conditions. <br>

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SRI SAI PRIYA.S
RegisterNumber: 212222240103
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
  data=pd.read_csv("/content/50_Startups.csv")
  data.head()
```
```
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
```
## Output:

data.head() 

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475702/abfdd69d-91eb-4bee-8eed-1d656f8a4750)

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475702/21af59b9-e77b-4c68-bb8d-7bf2496e93ca)

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475702/d16bc62b-9b4e-4317-8083-bacac49714ac)

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475702/0e05923c-1cab-45e0-b203-f4f18295efb6)

![image](https://github.com/SriSaiPriyaSenthilvel/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475702/56afb4b2-cdb8-49a2-98f0-f217718368d9)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
