import csv
import sys
import os
import numpy as np 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import data_handler
import data_visualizer   
# define model

class Vanilla_LSTM():
    
    def __init__(self, input_shape):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.model.add(LSTM(8, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', metrics = ['accuracy'], loss='mae')
        self.n_features = 1
    
    def train(self, X, y, epochs, verbose):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def test(self, X, y):
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps,1))
            yhat = self.model.predict(input, verbose=1)
            print(y[i], yhat[0][0], np.mean(input[0]))
        # model.fit(X, y, epochs=200, verbose=1)
   
        
if __name__ == "__main__":
    # numpy.set_printoptions(threshold=sys.maxsize)
    time = data_handler.read_data("lob_datatrial0001.csv","TIME")
    prices = data_handler.read_data("lob_datatrial0001.csv", "MIC")

    # splitting data into chunks of 4
    steps = 59
    reshape = True
    # X, y = data_handler.split_data(prices, steps, reshape)
    
    # split_ratio = [9,1]
    # train_X, test_X = data_handler.split_train_test_data(X, split_ratio)
    
    # train_X = train_X.reshape((-1, steps, 1))
    # test_X = test_X.reshape((-1, steps, 1))
    
    # train_y, test_y = data_handler.split_train_test_data(y, split_ratio)

    model = Vanilla_LSTM((steps,1))
    
    for i in range(9):
        print("epoch " + str(i+1) + " out of 9")
        prices = data_handler.read_data("lob_datatrial000" + str(i+1) + ".csv", "MIC")
        X, y = data_handler.split_data(prices, steps, reshape)
        model.train(X, y, 200, 0)
    
    checkpoint_path = "./Models/vanilla.ckpt"
    
    # model.save_model(checkpoint_path)
    # model.test(test_X, test_y)

    prices = data_handler.read_data("lob_datatrial0010.csv", "MIC")

 


    
        
    
    
    
    
   
    


