#!/usr/bin/env python
# coding: utf-8

# In[11]:


# LSTM implementation
get_ipython().system('pip install tensorflow ')
 # fro matrix multiplication
get_ipython().system('pip install numpy ')
# define data structure
get_ipython().system('pip install pandas  ')
#for visualization
get_ipython().system('pip install matplotlib  ')
#for normalizing data
get_ipython().system('pip install scikit-learn  ')


# In[12]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as StandardScaler
import io
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


PowerData = pd.read_csv(io.StringIO(uploaded['filename.csv']))
# selecting a column that we going to use
data_to_use=PowerData['1'].values
data_to_use

#data preprocessing(scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1,1))


# In[15]:


# plot the data
import matplotlib.pyplot as plt
plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('signal ID 1')
plt.xlable('Measurements')
plt.ylable('Amplitude')
plt.plot(scaled_data,label='signal 1')
plt.legend()
plt.show()

#features and label dataset
def window_data(PowerDate,window_size):
    X = []
    y = []
    
    i = 0
    while (i+window_size) <=len(PowerData) - 1:
        X.append(PowerData[i:i+window_size])
        y.append(PowerData[i+window_size])
        
        i += 1
    assert len(X) == len(y)
    returen X,y
#windowing the data with window_data function
#need disscuse the size 7?
X, y = window_data(scaled_data,7)


#Spliting data into train and test
# Paul: Keeping every 3,125 columns (800,000/256=3,125).
import numpy as np
X_train = np.array(X[:3215])
y_train = np.array(y[:3215])

X_test = np.array(X[3215:])
y_test = np.array(y[3215:])

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))

#batch_size: This is the number of windows of data we are passing at once.
#window_size: The number of days we consider to predict the bitcoin price for our case.
#hidden_layers: This is the number of units we use in our LSTM cell.
#clip_margin: This is to prevent exploding the gradient — we use clipper to clip gradients below above this margin.
#learning_rate: This is a an optimization method that aims to reduce the loss function.
#epochs:This is the number of iterations (forward and back propagation) our model needs to make.

batch_size = 7
window_size = 7
hidden_layer = 256
clip_margin = 4
learning_rate = 0.001
epochs = 200





