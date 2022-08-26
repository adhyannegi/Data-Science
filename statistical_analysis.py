#!/usr/bin/env python
# coding: utf-8

# ## y hat = xB
# ## y hat = py

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("Downloads/HousingData.csv")
data = data.dropna()
X = data.iloc[:,0:4]
Y = data.iloc[:,-1]


# In[3]:


X.shape


# In[4]:


X.head(50)


# # Computing p

# ## p = X(transpose(x)X)^(-1)transpose(X)

# In[5]:


XTX = X.T.dot(X)


# In[6]:


XTX.shape


# In[7]:


XTX = np.linalg.inv(XTX)


# In[8]:


p1 = (X.dot(XTX))


# In[9]:


p1.shape


# In[10]:


P = np.dot(np.dot(X,XTX),X.T)


# In[11]:


P.shape


# In[12]:


Y_pred = P.dot(Y)


# In[13]:


Y_pred.shape


# In[14]:


df = pd.DataFrame(Y_pred,Y)


# In[15]:


df.head()


# In[16]:


df = df.reset_index()


# In[17]:


df.head()


# In[18]:


df.columns = ['Y', 'Y_pred']


# In[19]:


df.head()


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(Y, Y_pred)


# In[21]:


def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(y_test, y_pred, color='red')
    plt.plot(range(my_range), range(my_range), 'o')
    plt.title(regressorName)
    plt.show()
    return

plotGraph(Y, Y_pred, 'alt')


# In[22]:


B = np.linalg.inv(X.T.dot(X)).dot(X.T)
B = B.dot(Y)
B


# In[ ]:




