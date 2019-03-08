# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 14:51:22 2019

@author: Namish Kaushik
"""

import pandas as pd


import statsmodels.api as sm
import numpy as np
import warnings
import matplotlib
#import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")
matplotlib.rcParams['axes.labelsize']=14
matplotlib.rcParams['xtick.labelsize']=12
matplotlib.rcParams['ytick.labelsize']=12
matplotlib.rcParams['text.color']='k'
mydata= pd.read_csv("aus.csv")

mydata['Years'] = pd.to_datetime(mydata['Year']).dt.year
mydata = mydata.set_index('Years')

passengers = mydata[['Passengers']]
passengers.plot(figsize=(10,8),linewidth=5,fontsize=20)
plt.xlabel('Year',fontsize= 20)
plt.ylabel('average passenger per year')
plt.title('plotting yuearly average')
plt.legend()


# creating train and test

train = passengers[0:28]
test = passengers[28:]
train.Passengers.plot(figsize=(10,8),title= "monthly passenger",fontsize=14)
test.Passengers.plot(figsize=(10,8),title= "monthly passenger",fontsize=14)
plt.show()



# douvble exponentialj

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing,Holt
y_hat_avg = test.copy()
fit1= Holt(np.asarray(train['Passengers'])).fit(optimized= True)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
plt.figure(figsize = (10,8))
plt.plot(train['Passengers'],label = 'Train')
plt.plot(test['Passengers'],label = 'Test')
plt.plot(y_hat_avg['Holt_linear'],label = 'Holt_linear')
plt.legend(loc= 'best')
plt.show()


# computing error value

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Passengers,y_hat_avg.Holt_linear))
print(rms)