# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:59:13 2017

@author: PedroAlberto
"""
#==============================================================================
# 
# for i in range(8):
#     print (i)
#==============================================================================

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




data = [
        [[1,2,3,4],[5,6,7,8]],
        [[1,2,3,4],[5,6,7,8]]
        ]


model = Sequential()
model.add(LSTM(7, return_sequences=True, input_dim=4))
#==============================================================================
# model.add(LSTM(7, return_sequences=True))
#==============================================================================
model.add(LSTM(7))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data, [4,4], nb_epoch=100, batch_size=1, verbose=0)






#==============================================================================
# from pylab import figure, axes, pie, title, show
# 
# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4],[0,1,3,1])
# plt.ylabel('some numbers')
# plt.show()
# 
# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()
# 
# plt.plot([1,2,3,4], [1,4,9,16], [1,2,3,4],[0,1,3,1])
# plt.axis([0, 6, 0, 20])
# plt.show()
# 
# 
# fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('to.png')   # save the figure to file
# plt.close(fig) 
# 
# 
# 
#==============================================================================



















