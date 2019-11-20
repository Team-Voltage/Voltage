
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
signal_id = metadata["signal_id"]
print(signal_id.shape)
print(signal_id.head())




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
df['signal_id'] = signal_id
idx = df.groupby('signal_id').apply(lambda x: x.sample(frac=0.2, random_state = 0)).index.get_level_values(1)
test_1 = df[df['signal_id'].isin(idx)]
train_1 = df[~df['signal_id'].isin(idx)]


#drop signal id
train_1=train_1.drop(['signal_id'], axis=1)
test_1=test_1.drop(['signal_id'], axis=1)
df=df.drop(['signal_id'], axis=1)




#Convert dataframe into values
values = df.values

train = train_1.values
test = test_1.values


# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
train = train.astype('float32')
test = test.astype('float32')



# frame as supervised learning
reframed = series_to_supervised(values, 1, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())



values = reframed.values


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)




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
