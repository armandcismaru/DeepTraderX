
import csv
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.regularizers import l2
import data_handler
import data_visualizer
from NeuralNetwork import NeuralNetwork

# Univariate LSTM used to predict a single next step in time series data
class Vanilla_LSTM(NeuralNetwork):
    def __init__(self, input_shape, filename):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].

        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.model.add(LSTM(50, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[1]
        self.filename = filename

    def test(self, X, y, verbose):
        preds = np.array([])
        baseline = np.array([])
        
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps,1))
            yhat = self.model.predict(input, verbose=verbose)
            preds = np.append(preds, yhat[0][0])
            baseline = np.append(baseline, np.mean(input[0]))
            print(y[i], preds[i], baseline[i])

        # print(len(y), len(preds), len(baseline))
        data_visualizer.accuracy_plot(y, preds, baseline)
        
    def run_all(self):
       
        time=data_handler.read_data("./Data/lob_data.csv", "TIME")
        prices=data_handler.read_data("./Data/lob_data.csv", "MIC")
        X, y=data_handler.split_data(prices, self.steps)

        split_ratio=[9, 1]
        train_X, test_X=data_handler.split_train_test_data(X, split_ratio)
        train_X=train_X.reshape((-1, self.steps, 1))
        test_X=test_X.reshape((-1, self.steps, 1))
        train_y, test_y=data_handler.split_train_test_data(y, split_ratio)

        self.train(train_X, train_y, 200, verbose = 1)
        self.test(test_X, test_y, verbose = 1)
        self.save()

    def run_all2(self):
        
        for i in range(9):
            print("epoch " + str(i+1) + " out of 9")
            prices = data_handler.read_data("./Data/lob_datatrial000" + str(i+1) + ".csv", "MIC")
            X, y = data_handler.split_data(prices, self.steps)
            self.train(X, y, 100, 0)
        
        time = data_handler.read_data("./Data/lob_data.csv", "TIME")
        prices = data_handler.read_data("./Data/lob_data.csv", "MIC")
        X, y = data_handler.split_data(prices, self.steps)
        self.test(X, y, verbose=1)

# Univariate LSTM that takes in multiple steps to to predict a single next step in time series data
class MultiVanilla_LSTM(NeuralNetwork):
    def __init__(self, input_shape, out_steps, filename):
        # inputs: A 3D tensor with shape[batch, timesteps, feature].

        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[0]
        self.out_steps = out_steps
        self.model.add(LSTM(24,  activation='relu', input_shape=input_shape))
        # self.model.add(LSTM(8,  return_sequences=True, activation='relu'))
        # self.model.add(LSTM(6, activation='relu'))
        self.model.add(Dense(self.out_steps))
        self.model.compile(optimizer='adam', metrics=['accuracy'], loss='mae')
        self.n_features = self.input_shape[1]
        self.filename = filename

    def test(self, X, y, verbose):
        preds = np.array([])
        baseline = np.array([])
        actual = np.array([])

        for i in range(len(X)):
            input = X[i].reshape((1, self.steps, 1))
            yhat = self.model.predict(input, verbose=verbose)
            preds = np.append(preds, yhat[0][self.out_steps - 1])
            # baseline = np.append(baseline, np.mean(input[0]))
            actual = np.append(actual, y[i][self.out_steps - 1])
            
            print(actual[i], preds[i])

        # print(len(y), len(preds), len(baseline))
        data_visualizer.accuracy_plot(actual, preds)

    def run_all(self):
    
        time = data_handler.read_data("./Data/lob_datatrial0001.csv", "TIME")
        prices = data_handler.read_data("./Data/lob_datatrial0001.csv", "MIC")
        X, y = data_handler.multi_split_data(prices, self.steps, self.out_steps)

        split_ratio = [9, 1]
        train_X, test_X = data_handler.split_train_test_data(X, split_ratio)
        train_X = train_X.reshape((-1, self.steps, 1))
        test_X = test_X.reshape((-1, self.steps, 1))
        train_y, test_y = data_handler.split_train_test_data(y, split_ratio)
        train_y = train_y.reshape((-1, self.out_steps))
        test_y = test_y.reshape((-1, self.out_steps))
        self.train(train_X, train_y, 200, verbose=1)
        self.test(test_X, test_y, verbose=1)
        self.save()

class Multivariate_LSTM(NeuralNetwork):
    def __init__(self,  input_shape, filename):
        
        # setup
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[1]
        self.n_features = self.input_shape[2]
        self.filename = filename
        self.batch_size = input_shape[0]
        
        # self.max_vals = np.array([])
        # self.min_vals = np.array([])

        # architecture
        self.model.add(LSTM(10, activation='relu',  input_shape=(input_shape[1], input_shape[2])))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(3, activation='relu'))
        # self.model.add(Dense(3))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

        # compiling + options
        opt = Adam(learning_rate=1.5e-5)
        self.model.compile(optimizer=opt, metrics=['mae','msle','mse'], loss='mse')
        
        
    def create_model(self):
        pkl_path = "./normalized_data.pkl"
        train_data = data_handler.DataGenerator(
            pkl_path, self.batch_size, self.n_features)
        self.max_vals = train_data.train_max
        self.min_vals = train_data.train_min
        self.model.fit_generator(
            train_data, epochs=20, verbose=1, workers=16)
        self.save()
        tf.keras.utils.plot_model(
            self.model,
            to_file="model.png",
            show_shapes=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # single step vanilla LSTM
    # steps = 9
    # vanilla = Vanilla_LSTM((steps,1),  f"MIC_Predictor_{steps}.8")
    # vanilla.run_all2()

    # multiple step vanilla LSTM
    # in_steps = 9
    # out_steps = 1
    # mul = MultiVanilla_LSTM((in_steps,1), out_steps, f"MIC_MUL_Predictor")
    # mul.run_all()

    # multivariate LSTM
    batch_size = 16384
    no_features = 13
    no_steps = 1
    mv = Multivariate_LSTM( (batch_size, no_steps, no_features), f"DeepTrader1_6")
    mv.create_model()




    
    
