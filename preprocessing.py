
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pyarrow.parquet as pq
from sklearn import preprocessing #used for scaling #pip install sklearn
import numpy as np #Importing numpy
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as StandardScaler
import io
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.metrics import matthews_corrcoef
from keras import backend as K
from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split


#print working dir
cwd = os.getcwd()
print(cwd)

#Read in denoised data (800000, 8712)
df = pq.read_table('/projects/jbaldo/jlartey/denoisedData.parquet')
df = df.to_pandas()
print("Original DataFrame : ")
print(df.shape)
print(df.shape)
print(df.head())


#Transpose datafarme (8712, 800000)
print("Transposed DataFrame : ")
df = df.transpose()
print(df.shape)
print(df.head())


#Read in metadata
metadata = pd.read_csv('/home/jlartey/metadata_train.csv')
print("metadata:")
print(metadata.shape)
print(metadata.head())



#Create a target dataframe and add it to the entire df
print("Create a signal_id dataframe")
target = metadata["target"]
print(target.shape)
print(target.head())




#Preprocess function for normalizing data, #Scaling the data to be between zero and one
def preprocess_df(df):
    #normalizing data
    df = preprocessing.normalize(df, norm='l2', axis=1, copy=True, return_norm=False)
    df = pd.DataFrame(df)   
    #Scaling the data to be between zero and one
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)
    scaled = pd.DataFrame(scaled)
    return scaled


df = preprocess_df(df)
print(df.head())

print("df.isnull().values.any()")
print(df.isnull().values.any())




# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg




#Split data into test train

y = target
x = df

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 0)

#convert to values
train_X = train_X.values
test_X = test_X.values
train_y = train_y.values
test_y = test_y.values

#Change datatype to float32
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_y = train_y.astype('float32')
test_y = test_y.astype('float32')


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(train_X[0].shape)


#Convert dataframe into values
values = df.values

# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')



# frame as supervised learning
reframed = series_to_supervised(values, 1, 1)
print(reframed.head())

values = reframed.values



def matthews_correlation(inv_y, inv_yhat):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    inv_yhat = tf.convert_to_tensor(inv_yhat, np.float32)
    inv_y = tf.convert_to_tensor(inv_y, np.float32)
    
    y_pred_pos = K.round(K.clip(inv_yhat, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(inv_y, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
    
    
    
    
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=[matthews_correlation])
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history



# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(yhat)

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler1.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]



# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler1.inverse_transform(inv_y)
inv_y = inv_y[:,0]




# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)




# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(inv_y, inv_yhat):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = K.eval(matthews_correlation(inv_y.astype(np.float32), (inv_yhat > threshold).astype(np.float32)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result





best_threshold = threshold_search(inv_yhat, inv_y)['threshold']









# Now load the test data
# This first part is the meta data, not the main data, the measurements
meta_test = pd.read_csv ('/projects/jbaldo/jlartey/metadata_test.csv')






meta_test = meta_test.set_index(['signal_id'])
meta_test.head()









#Read in denoised data (800000, 20337)
X_test = pq.read_table('/projects/jbaldo/jlartey/denoised_test_data.parquet')
X_test = X_test.to_pandas()
print("denoisedData.: ")
print(X_test.shape)
print(X_test.head())





#Drop last row (799999, 20337)

# X_test = X_test[:-1]
# print("Dropped last row :")
# print(X_test.shape)

#Transpose dataframe from (800000, 20337) to (20337, 800000)
print("Transposed DataFrame : ")
X_test = X_test.transpose()
print(X_test.shape)



#Convert dataframe into values
X_test_1 = X_test.values



# integer encode direction
encoder = LabelEncoder()
X_test_1[:,4] = encoder.fit_transform(X_test_1[:,4])
# ensure all data is float
X_test_1 = X_test_1.astype('float32')





#preprocess data (normalize/and scale)

#Scaling the data to be between zero and one
scaler = MinMaxScaler()
X_test_1 = scaler.fit_transform(X_test_1)
print("X_test_1 = scaler.fit_transform(X_test_1)")
X_test_1.shape



# This is supposed to turn the data set from two dimensions into three (preps the data for the model)
# Before data prep, the data looks like this (x,y)
# After the data prep, the data looks like this (a,b,c)
# The data for the model needs to be 3D (samples, timesteps, features)
X_test_1 = X_test_1.reshape((X_test_1.shape[0], 1, X_test_1.shape[1]))
print("X_test_1 = X_test_1.reshape((X_test_1.shape[0], 1, X_test_1.shape[1]))")
print(X_test_1.shape)
#np.save("X_test.npy",X_test_1)
#X_test_1.shape

#submission Data shape (20337, 2)
submission = pd.read_csv ('/projects/jbaldo/jlartey/sample_submission.csv')
print(len(submission))
print(submission.shape)
submission.head()



# How many times we split our data
N_SPLITS = 5


# Predicting the output for all of the inputs
preds_test = []
for i in range(N_SPLITS):
    pred = model.predict(X_test_1, batch_size=300, verbose=1)
    pred_3 = []
    for pred_scalar in pred:
        for i in range(1):
            pred_3.append(pred_scalar)
    preds_test.append(pred_3)
    
    


# Creating a binary. 1 = partial discharge (preds_test is all of the partial discharges)
preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
preds_test.shape




# Updates the target column in the submission file 
submission['target'] = preds_test
submission.to_csv('/projects/jbaldo/jlartey/submission.csv', index=False)
submission.head()

