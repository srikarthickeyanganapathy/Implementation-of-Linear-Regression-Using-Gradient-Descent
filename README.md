# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Sri Karthickeyan Ganapathy
RegisterNumber:  212222240102

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X, y , theta and generate the cost fuction in a linear regression model
  """
  m=len(y)
  h=X.dot(theta)  # length of training data
  square_err=(h-y)**2 

  return 1/(2*m) * np.sum(square_err)  
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)   # function call

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
 predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### PROFIT PREDICTION:
![out](https://user-images.githubusercontent.com/115707860/229810205-e3213d36-1d63-4814-9403-c6443c0f8d34.png)
### COST FUNCTION:
![out](https://user-images.githubusercontent.com/115707860/229810399-a24dc2e0-3767-4363-a598-442a9c08b9cb.png)
### GRADIENT DESCENT:
![out](https://user-images.githubusercontent.com/115707860/229810569-1349a4dc-1e6b-4ecb-a97e-4c664e1a649f.png)
### COST FUNCTION USING GRADIENT DESCENT:
![out](https://user-images.githubusercontent.com/115707860/229810861-8bb4ce4e-4e44-4cea-a773-b175a93c7bc6.png)
### GRAPH WITH BEST FIT LINE (PROFIT PREDICTION):
![out](https://user-images.githubusercontent.com/115707860/229811328-621bb11e-0b69-4642-a370-678aaf263177.png)
### PROFIT PREDICTION FOR A POPULATION OF 35,000:
![out](https://user-images.githubusercontent.com/115707860/229811762-82adce62-6abf-49bd-b9f8-2eca2fb80725.png)
### PROFIT PREDICTION FOR A POPULATION OF 70,000:
![out](https://user-images.githubusercontent.com/115707860/229811946-3177730d-a45c-4243-8f4c-0f3d81813120.png)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
