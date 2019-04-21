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

#Ex 1

#creating column names for airq402 data
nc=["City1","City2","Average Fair","Distance","Avg Weekly Passengers","Market Leading Airline","Market Share","Average Fair","Low Price Airline","Market Share","Price"] #header names

airq402=pd.read_csv("http://www.stat.ufl.edu/~winner/data/airq402.dat", names=nc, sep="\s+", header=None)
print(airq402)

#creating column names for wine quality data
nc1=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

wqred=pd.read_csv("winequality-red.csv",sep=';', header=None, names=nc1)
print(wqred)

wqwhite=pd.read_csv("winequality-white.csv",sep=';', header=None, names=nc1)
print(wqwhite)

dframe=pd.DataFrame(airq402)

# this function is converting categorical values into numerical values
def handle_non_numeric_data(dframes):
    columns=dframes.columns.values
    
    for column in columns:
        text_digit_values={}
        def convert_to_int(val):
            return text_digit_values[val]
        
        if dframes[column].dtype != np.int64 and dframes[column].dtype != np.float64:
            column_content=dframes[column].values.tolist()
            unique_elements=set(column_content)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_values:
                    text_digit_values[unique]=x
                    x+=1
            dframes[column]=list(map(convert_to_int,dframes[column]))
            
    return dframes

dframe=handle_non_numeric_data(dframe)
print(dframe.head())


# finding rows with null values in all three datasets
null_data = airq402[airq402.isnull().any(axis=1)]
print(null_data)

null_data1 = wqred[wqred.isnull().any(axis=1)]
print(null_data1)

null_data2 = wqwhite[wqwhite.isnull().any(axis=1)]
print(null_data2)

#splitting all the three data sets
splitdata=np.random.rand(len(dframe))<0.8
aqtrain=dframe[splitdata]
aqtest=dframe[~splitdata]
print(aqtrain,aqtest)

splitdata=np.random.rand(len(wqred))<0.8
wqredtrain=wqred[splitdata]
wqredtest=wqred[~splitdata]
print(wqredtrain,wqredtest)

splitdata=np.random.rand(len(wqwhite))<0.8
wqwhitetrain=wqwhite[splitdata]
wqwhitetest=wqwhite[~splitdata]
print(wqwhitetrain,wqwhitetest)



# =============================================================================
#Ex 2
# =============================================================================

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
Alpha=0.0000002
Beta=np.zeros(len(wqredtrain.iloc[0:12]),float)
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

# =============================================================================
# bias= np.ones((len(aqtrain), 1))
# aqtrain.insert(0,'bias',bias)
# X=aqtrain[:,0:12]
# Y=aqtrain[:,12]
# XB= np.column_stack((bias, X))
# iteration=1000
# Alpha=0.00000001
# Beta=np.zeros(len(aqtrain[0:12]),float)
# gradient_descent(Beta,XB,Y,Alpha,iteration,aqtest)
# =============================================================================

