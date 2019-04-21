# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 20:36:34 2017

@author: Faaiz
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:44:48 2017

@author: Faaiz
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#creating column names for wine quality data
nc1=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

wqred=pd.read_csv("winequality-red.csv",sep=';', header=None, names=nc1)
print(wqred)

wqwhite=pd.read_csv("winequality-white.csv",sep=';', header=None, names=nc1)
print(wqwhite)

# finding rows with null values in both the datasets

null_data1 = wqred[wqred.isnull().any(axis=1)]
print(null_data1)

null_data2 = wqwhite[wqwhite.isnull().any(axis=1)]
print(null_data2)

#splitting both the data sets

splitdata=np.random.rand(len(wqred))<0.8
wqredtrain=wqred[splitdata]
wqredtest=wqred[~splitdata]
print(wqredtrain,wqredtest)

splitdata=np.random.rand(len(wqwhite))<0.8
wqwhitetrain=wqwhite[splitdata]
wqwhitetest=wqwhite[~splitdata]
print(wqwhitetrain,wqwhitetest)


# Implementing the Gradient Descent Algoithm
def gradient_descent(Beta,XB,Y,Alpha,iteration,z):
     XT= np.transpose(X)
     la=[] 
     rmse=[]
     #creating a column of biased values 1
     bias= np.ones((len(z), 1))
     z.insert(0,'bias',bias)
     X_test=z.iloc[:,0:12]
    
     for i in range(0,iteration):  
         
       
         #∂f/∂βˆ(βˆ) = −2(XTy − XT Xβˆ)= −2XT(y − Xβˆ)
         XB=np.dot(X,Beta)
         H=np.subtract(Y,XB)
         gradient=-2*(np.dot(XT,H))
         Beta=Beta - Alpha* gradient 
         l=lf(X,Y,Beta)
         la.append(l)
         
         rmse1=(np.sqrt(l)/len(X_test))
         rmse.append(rmse1)

     plt.plot(rmse)
     plt.show()
         
     plt.plot(la)
     plt.show()

# The values of test data is used in the function defined.
# Outside the block, we insert bias column and take the integer values of the trained dataset.     

def lf(X,Y,Beta):   
     predY=np.dot(X,Beta) # Predicted value of Y
     loss=np.subtract(Y,predY)
     loss=np.sum(loss*loss) # calculating the sum of the squares of residuals
     return loss

# calculating gradient descent for red wine

bias= np.ones((len(wqredtrain), 1))
wqredtrain.insert(0,'bias',bias)
X=wqredtrain.iloc[:,0:12] # works on the positions in the index 
print(X)
Y=wqredtrain.iloc[:,12]
XB= np.column_stack((bias, X))
iteration=1000
Alpha=0.0000002 # The values of step length (learning rate) are kept low as keeping it higher was not producing the correct graph.
Beta=np.zeros(len(wqredtrain.iloc[0:12]),float) # adds a column of 0s
gradient_descent(Beta,XB,Y,Alpha,iteration,wqredtest)

# calculating gradient descent for white wine

bias= np.ones((len(wqwhitetrain), 1))
wqwhitetrain.insert(0,'bias',bias)
X=wqwhitetrain.iloc[:,0:12]
Y=wqwhitetrain.iloc[:,12]
XB= np.column_stack((bias, X))
iteration=1000
Alpha=0.00000001
Beta=np.zeros(len(wqwhitetrain.iloc[0:12]),float)
gradient_descent(Beta,XB,Y,Alpha,iteration,wqwhitetest)

# The same procedure is used for both the wine quality data set and is implemented by the ‘gradient_descent’ function.
