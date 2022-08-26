#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Downloads/HousingData.csv")
data = data.dropna()
data1 = data[['ZN','INDUS','NOX','RM']]
vector = data1.to_numpy()
data1


# In[3]:


x1= vector[:,0]
x2= vector[:,1]
x3= vector[:,2]
x4= vector[:,3]
Y = data["MEDV"].to_numpy()

L = 0.0001 #Learning rate
iterations = 1000
n = 394 #rows
b0 = 0 #beta values
b1 = 0
b2 = 0
b3 = 0
b4 = 0
b0_list = list()
b1to4_list = list()
Y_pred = 0
Y_preds_list = list()
loss_list = []

for i in range(iterations):
    b0 = b0 - L * ((-2/n)*sum(Y - Y_pred))
    b1 = b1 - L * ((-2/n)*sum(x1*(Y-Y_pred)))
    b2 = b2 - L * ((-2/n)*sum(x2*(Y-Y_pred)))
    b3 = b3 - L * ((-2/n)*sum(x3*(Y-Y_pred)))
    b4 = b4 - L * ((-2/n)*sum(x4*(Y-Y_pred)))
    
    Y_pred = b0 + (b1*x1) + (b2*x2) + (b3*x3) + (b4*x4)
    Y_preds_list.append(Y_pred)
    
    loss = np.sqrt((np.sum(np.square(Y-Y_pred)))/n)
    loss_list.append(loss)
    
    b1to4_list.append([b1, b2, b3, b4])
    b0_list.append(b0) 
print(b1to4_list[-1])
print(b0_list[-1])


# In[4]:


plt.plot(list(range(iterations)), loss_list, '-r')


# ## Function format with Learning rate and iterations as parameters

# In[5]:


x1= vector[:,0]
x2= vector[:,1]
x3= vector[:,2]
x4= vector[:,3]
Y = data["MEDV"].to_numpy()

def Grad_descent(L, iterations):
    n = 394 #rows
    b0 = 0 #beta values
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b0_list = list()
    b1to4_list = list()
    Y_pred = 0
    Y_preds_list = list()
    loss_list = []
    
    for i in range(iterations):
        b0 = b0 - L * ((-2/n)*sum(Y - Y_pred))
        b1 = b1 - L * ((-2/n)*sum(x1*(Y-Y_pred)))
        b2 = b2 - L * ((-2/n)*sum(x2*(Y-Y_pred)))
        b3 = b3 - L * ((-2/n)*sum(x3*(Y-Y_pred)))
        b4 = b4 - L * ((-2/n)*sum(x4*(Y-Y_pred)))

        Y_pred = b0 + (b1*x1) + (b2*x2) + (b3*x3) + (b4*x4)
        Y_preds_list.append(Y_pred)

        loss = np.sqrt((np.sum(np.square(Y-Y_pred)))/n)
        loss_list.append(loss)

        b1to4_list.append([b1, b2, b3, b4])
        b0_list.append(b0) 
    
    plt.plot(list(range(iterations)), loss_list, '-r')
    return ((b1to4_list[-1]),(b0_list[-1]))


# In[6]:


test1 = Grad_descent(0.0001, 1000)
print(test1)


# In[7]:


print(loss)

