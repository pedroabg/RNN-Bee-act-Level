# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix

feature_dim = 5
look_back = 0
data_file = 'abelha-sem-chuva.csv'
data_use_percent = 0.1

def create_dataset(dataset, look_back=0):
	dataX, dataY = [], []
 #len(dataset) tamanho do dataset
 #exemplo len(dataset) = 10, look_back = 1, entao range = 8, loop de 0 a 7
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), feature_dim]
		#print(dataset[i + look_back])
		#print(a)
		a =  numpy.concatenate((dataset[i + look_back][0:feature_dim] , a))
		dataX.append(a)
  #pegando o valor do label seguinte
		dataY.append(dataset[i + look_back, feature_dim])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
dataframe = pandas.read_csv(data_file, engine='python')
dataset = dataframe.values
#resizing dataset len for faster tests 
dataset = dataset[:int(len(dataset)*data_use_percent)]
dataset = dataset.astype('float32')


#==============================================================================
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
#==============================================================================


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#==============================================================================
# print("-----------------")
# print(trainX)
# print("-----------------")
# print(trainY)
#==============================================================================
ts = 7
#==============================================================================
# # reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0]//ts, ts, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0]//ts, ts, testX.shape[1]))
# 
#==============================================================================
# reshape input to be [samples, time steps, features]

# SÓ MUDA AQUI O TIME STEPS
#trainX = numpy.reshape(trainX, (1, trainX.shape[0], feature_dim ))
#testX = numpy.reshape(testX, (1, testX.shape[0], feature_dim ))



#==============================================================================
# data = data.reshape((1, 9, 11))
# target = target.reshape((1, 9, 11))
#==============================================================================

# create and fit the LSTM network
model = Sequential()
#==============================================================================
# model.add(LSTM(7, return_sequences=True, input_dim=(look_back+feature_dim)))
# #==============================================================================
# # model.add(LSTM(7, return_sequences=True))
# #==============================================================================
# model.add(LSTM(7))
#==============================================================================
model.add(LSTM(7, input_dim=feature_dim, return_sequences=True))
# model.add(LSTM(7, return_sequences=True, input_shape=trainX.shape))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, verbose=0)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)



trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

#==============================================================================
# ---------------------------PLOT-------------------------------------
#==============================================================================

# shift train predictions for plotting
#cria um array no formato da dataset porém todo vazio. a mesma shape cheio de nan
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[0:len(trainPredict), :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+look_back-1:len(trainPredict)+look_back-1+len(testPredict), :] = testPredict
# plot baseline and predictions
plt.plot(dataset[look_back:,feature_dim])
plt.plot(trainPredictPlot,c="g")

plt.plot(testPredictPlot, c='r')
plt.show()

#==============================================================================
# fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# ax.plot(dataset[:,6])
# ax.plot(trainPredictPlot,c="g")
# ax.plot(testPredictPlot, c='r')
# fig.savefig('to.png')   # save the figure to file
# plt.close(fig) 
# 
#==============================================================================



fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(dataset[352:600,feature_dim])
ax.plot(trainPredictPlot[350:500],c="g")
ax.plot(testPredictPlot[350:600], c='r')
fig.savefig('to100.png')   # save the figure to file
plt.close(fig) 









