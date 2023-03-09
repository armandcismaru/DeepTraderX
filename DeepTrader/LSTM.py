from NeuralNetwork import NeuralNetwork
from DataGenerator import DataGenerator
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Multivariate_LSTM(NeuralNetwork):
    def __init__(self, input_shape, filename):

        # setup
        self.input_shape = input_shape
        self.model = Sequential()
        self.steps = input_shape[1]
        self.n_features = self.input_shape[2]
        self.filename = filename
        self.batch_size = input_shape[0]

        # architecture
        self.model.add(LSTM(10, activation='relu',
                            input_shape=(input_shape[1], input_shape[2])))
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dense(3, activation='relu'))
        self.model.add(Dense(1))

        # compiling + options
        opt = Adam(learning_rate=1.5e-5)
        self.model.compile(optimizer=opt, metrics=[
                           'mae', 'msle', 'mse'], loss='mse')

    def create_model(self):
        pkl_path = "./normalized_data.pkl"
        train_data = DataGenerator(
            pkl_path, self.batch_size, self.n_features)
    
        self.max_vals = train_data.train_max
        self.min_vals = train_data.train_min
        self.model.fit(
            train_data, epochs=20, verbose=1, workers=16)
        self.save()
        # tf.keras.utils.plot_model(
        #     self.model,
        #     to_file="model.png",
        #     show_shapes=False,
        #     show_layer_names=True,
        #     rankdir="TB",
        #     expand_nested=False,
        #     dpi=96,
        # )


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    
    # multivariate LSTM
    batch_size = 1638
    no_features = 13
    no_steps = 1
    mv = Multivariate_LSTM(
        (batch_size, no_steps, no_features), f"DeepTrader1_6")
    mv.create_model()
