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

feature_dim = 1
look_back = 1
data_file = 'abelha-prev-act.csv'
data_use_percent = 1
#row , col, square
net_arc = "square"

# fix random seed for reproducibility
#numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv(data_file, engine='python')


#==============================================================================
# dataframe = dataframe.sample(frac=1)
#==============================================================================
dataset = dataframe.values
#resizing dataset len for faster tests 
dataset = dataset[:int(len(dataset)*data_use_percent)]
dataset = dataset.astype('float32')


# split into train and test sets
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


trainX, trainY = f.create_dataset(train, feature_dim, look_back)
testX, testY = f.create_dataset(test, feature_dim, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = f.baseline_model(feature_dim + look_back)
model.fit(trainX, trainY, nb_epoch=100, verbose=1)


#==============================================================================
# for i in range(10):
#==============================================================================
#==============================================================================
# model = f.baseline_model(feature_dim + look_back)
# model.fit(trainX, trainY, nb_epoch=100, verbose=0)
#==============================================================================
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print(' Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print(' Test Score: %.2f RMSE' % (testScore))
a = pandas.DataFrame(testPredict)
b = pandas.DataFrame(testY)
c = pandas.concat([a, b], axis=1)
#==============================================================================
# csv_file = "resultados_csv/predicted_"+data_file+"_"+str(i)+".csv"
#==============================================================================
#==============================================================================
# csv_file = "resultados_csv/"+str(look_back)+"_predicted_LAST_gru_4.csv"
# pandas.DataFrame(c).to_csv(csv_file)
#==============================================================================


# make predictions







#==============================================================================
# f.g_plot(dataset,trainPredict, testPredict, feature_dim, look_back)
#==============================================================================






