# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Open train.parquet

import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pywt
from scipy import signal
from statsmodels.robust import mad




train = pq.read_table('/home/jlartey/train.parquet')
train = train.to_pandas()

temp = train.head()

print(temp)


#Visualization of first 3 phases with target 0

fig=plt.figure(figsize=(8, 8), dpi= 100, facecolor='w', edgecolor='k')
plot_labels = ['Phase 1', 'Phase 2', 'Phase 3']
colores = ["#3D9140", "#FF6103", "#8B2323"]
plt.plot(list(range(len(train))), train['0'], '-', label=plot_labels[0
], color=colores[0])
plt.plot(list(range(len(train))), train['1'], '-', label=plot_labels[1
], color=colores[1])
plt.plot(list(range(len(train))), train['2'], '-', label=plot_labels[2
], color=colores[2])
plt.ylim((-30, 30))
plt.legend(loc='lower right')
plt.title('Raw Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


#Box plot of first 3 phases with target 0
 
value1 = train['0']
value2 = train['1']
value3 = train['2']
 
box_plot_data=[value1,value2,value3]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Phase A','Phase B','Phase C'])
print(plt.show())


#Visualization of second 3 phases with target 1
## Columns with partial discharge

fig=plt.figure(figsize=(8, 8), dpi= 100, facecolor='w', edgecolor='k')
plot_labels = ['Phase 1', 'Phase 2', 'Phase 3']
colores = ["#3D9140", "#FF6103", "#8B2323"]
plt.plot(list(range(len(train))), train['3'], '-', label=plot_labels[0
], color=colores[0])
plt.plot(list(range(len(train))), train['4'], '-', label=plot_labels[1
], color=colores[1])
plt.plot(list(range(len(train))), train['5'], '-', label=plot_labels[2
], color=colores[2])
plt.ylim((-30, 30))
plt.legend(loc='lower right')
plt.title('Raw Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


#Box plot of second 3 phases with target 1
 
value1 = train['3']
value2 = train['4']
value3 = train['5']
 
box_plot_data=[value1,value2,value3]
plt.boxplot(box_plot_data,patch_artist=True,labels=['Phase A','Phase B','Phase C'])
plt.show()


# Step 2: Using Pywavelets to remove noise (High-frequency noise)

# References
# Why using this aproach (Discrete Wavelet Transfrom)? Reading:Links provided in the reference section [8] & [9] Project report
# I used and modified the following code (Kernel-link below) to get the wavelet smoothing parameters and de-noise the dataset. 
# http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/ 
# To find a deep explanation how the Python code works: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
# To explore deeply into Wavelet : "A guide for using the Wavelet Transform in Machine Learning"
#http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

try:
    def waveletSmooth( x, wavelet="db4", level=1, title=None ):
        coeff = pywt.wavedec( x, wavelet, mode="per" )
        sigma = mad( coeff[-level] )
        uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
        coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )

        y = pywt.waverec( coeff, wavelet, mode="per" )
        f, ax = plt.subplots(figsize=(8, 4), dpi= 100, facecolor='w', edgecolor='k')
        colores = ["#3D9140", "#FF6103", "#8B2323"]
        plt.plot( x, color="#FF6103", alpha=0.5, label="Original")
        plt.plot( y, color="#8B2323", label="Transformed" )
        plt.ylim((-50, 50))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend(loc='lower right')
        if title:
            ax.set_title(title)
        ax.set_xlim((0,len(y)))
        return y
except Exception as e: print(e)

try:
    #Creat list to store output of denoised data
    mylist = [] #noisy data
    zlist = [] #denoised data
    for j in train:
        title1 = 'Discrete Wavelet Transformed Signal: ' + str(j)
        signal_n = waveletSmooth(train[j], wavelet="db4", level=1, title=title1)
        mylist.append(signal_n)
        z = (train[j])-signal_n # z= noise
        zlist.append(z)
        print(signal_n) 
except Exception as e: print(e)

try:
    # Tranform noisy data list to dataframe 
    dfList = pd.DataFrame(mylist)
    dfList = dfList.transpose()
    print(dfList.transpose().head())
except Exception as e: print(e)

try:
    # Tranform list of "original data - noisy data" to dataframe
    zdf = pd.DataFrame(zlist)
    zdf = zdf.transpose()
    print(zdf.transpose().head())
    zdf.to_csv('/scratch/jlartey/denoisedData.csv')
except Exception as e: print(e)






