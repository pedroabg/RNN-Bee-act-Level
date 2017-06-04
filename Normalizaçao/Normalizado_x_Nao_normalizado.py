
# coding: utf-8

# In[1]:

import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import aux_func as f


# In[24]:

feature_dim = 6
look_back = 0
data_file = 'abelha-original-formatado.csv'
data_use_percent = 1
#row , col, square
net_arc = "square"

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv(data_file, engine='python')


# In[38]:


dataset = dataframe.values
#resizing dataset len for faster tests 
dataset = dataset[:int(len(dataset)*data_use_percent)]
dataset = dataset.astype('float32')


# In[50]:

dataframe.head(5)


# In[39]:

# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


trainX, trainY = f.create_dataset(train, feature_dim, look_back)
testX, testY = f.create_dataset(test, feature_dim, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the GRU network
model = f.baseline_model(feature_dim + look_back)
model.fit(trainX, trainY, nb_epoch=500, verbose=0)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print(' Train Score: %.6f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print(' Test Score: %.6f RMSE' % (testScore))
f.g_plot(dataset,trainPredict, testPredict, feature_dim, look_back)


# <h1>Agora vou normalizar</h1>

# In[47]:

dataframe.head(5)


# In[41]:

old_act = dataframe['act']


# In[42]:

#old_act


# In[43]:

dataframe2 = (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())


# In[44]:

dataframe2['act'] = old_act


# Normalização feita somente nos dados climáticos. Mantendo o target 

# In[46]:

dataframe2.head(5)


# In[48]:

dataset2 = dataframe2.values
#resizing dataset len for faster tests 
dataset2 = dataset2[:int(len(dataset2)*data_use_percent)]
dataset2 = dataset2.astype('float32')

# split into train and test sets
train_size = int(len(dataset2) * 0.70)
test_size = len(dataset2) - train_size
train, test = dataset2[0:train_size,:], dataset2[train_size:len(dataset),:]


trainX, trainY = f.create_dataset(train, feature_dim, look_back)
testX, testY = f.create_dataset(test, feature_dim, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = f.baseline_model(feature_dim + look_back)
model.fit(trainX, trainY, nb_epoch=500, verbose=0)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print(' Train Score: %.6f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print(' Test Score: %.6f RMSE' % (testScore))


# In[49]:

f.g_plot(dataset2,trainPredict, testPredict, feature_dim, look_back)


# Ficou bem diferente normalizado...

# In[ ]:



