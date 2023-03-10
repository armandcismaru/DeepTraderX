# pylint: skip-file
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import LambdaCallback
from keras.models import model_from_json
import csv


class NeuralNetwork():
    
    def __init__(self):
        pass
    
    def train(self, X, y, epochs, verbose=1):
        self.model.fit(X, y, epochs=epochs, verbose=verbose)
    
    # def train_debug(self, X, y, epochs, verbose=1):
    #     print_weights = LambdaCallback(
    #         on_epoch_end=lambda batch, logs: print(self.model.layers[0].get_weights()))
    #     self.model.fit(X, y, epochs=epochs, verbose=verbose,
    #                    callbacks=[print_weights])
    

    def test(self, X, y, verbose=1):
        for i in range(len(X)):
            input = X[i].reshape((1,self.steps, self.input_shape[1]))
            yhat = self.model.predict(input, verbose=verbose)
            print(y[i], yhat[0][0])
    
    def save(self):

        # create new directory if not already there
        path = "./Models/" + self.filename + "/"
        file = path + self.filename
        try: 
            os.system("mkdir " + path)
        except:
            print("hey")
            pass

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file + ".json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5 
        self.model.save_weights(file + ".h5")

        # saving normalization values to csv
        with open(file + '.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(self.max_vals)
            writer.writerow(self.min_vals)

        print("Saved model to disk.")

    @staticmethod
    def load_network(filename):
        
        # path directory variables
        path = "./Models/" + filename + "/"
        file = path + filename
        
        # load json and create model
        json_file = open(file + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(file + ".h5")

        # print("Loaded model from disk.")
        return loaded_model

    @staticmethod
    def normalization_values(filename):

        # path directory variables
        path = "./Models/" + filename + "/"
        file = path + filename

        # values used to normalize training data
        max_vals = np.array([])
        min_vals = np.array([])
        
        with open(file + '.csv', "r") as f:
            f_data = list(csv.reader(f))
            max_vals = np.array([float(f.strip()) for f in f_data[0]])
            min_vals = np.array([float(f.strip()) for f in f_data[1]])

        return max_vals, min_vals

        
        
        
        


