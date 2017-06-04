# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# define base mode
def baseline_model(dim):
#==============================================================================
#     model = Sequential()
#     model.add(LSTM(2, input_dim=(dim),input_length=1, return_sequences=True))
#     model.add(LSTM(2, return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#==============================================================================
    model = Sequential()
    model.add(GRU(2, input_dim=(dim), return_sequences=True))
    model.add(GRU(2, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#==============================================================================
#     model = Sequential()
#     model.add(GRU(4, input_dim=(dim), return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#==============================================================================
#==============================================================================
#     model = Sequential()
#     model.add(LSTM(4, input_dim=(dim), return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#==============================================================================
#==============================================================================
#     model = Sequential()
#     model.add(LSTM(1, input_dim=(dim), input_length=1, return_sequences=True))
#     model.add(LSTM(1,  return_sequences=True))
#     model.add(LSTM(1, return_sequences=True))
#     model.add(LSTM(1, return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#==============================================================================
#==============================================================================
#     model = Sequential()
#     model.add(GRU(1, input_dim=(dim), input_length=1, return_sequences=True))
#     model.add(GRU(1,  return_sequences=True))
#     model.add(GRU(1, return_sequences=True))
#     model.add(GRU(1, return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#==============================================================================
    

def create_dataset(dataset, feature_dim, look_back=0):
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

 
#==============================================================================
# ---------------------------PLOT-------------------------------------
#==============================================================================
    
def g_plot(dataset, trainPredict, testPredict, feature_dim, look_back):
        

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
    return

