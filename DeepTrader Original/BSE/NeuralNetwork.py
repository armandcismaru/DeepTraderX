# pylint: skip-file
import numpy as np
import os
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
            input = X[i].reshape((1, self.steps, self.input_shape[1]))
            yhat = self.model.predict(input, verbose=verbose)
            print(y[i], yhat[0][0])

    def save(self):

        # create new directory if not already there
        path = "./Models/" + self.filename + "/"
        file = path + self.filename
        try:
            os.system('mkdir ' + path)
        except:
            pass

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(file + ".h5")
        max = [6.0000000e+02, 1.0000000e+00, 2.2400000e+02, 6.0950000e+02, 9.4987500e+02,
               1.0000000e+00, 1.1130000e+03, 2.2200000e+02, 1.1130000e+03, 2.3550000e+01,
               6.7000000e+01, 7.8813584e+01, 1.9753013e+02, 2.2400000e+02]
        min = [0.00000000e+000, 0.00000000e+000, 0.0, 0.00000000e+000,
               0.00000000e+000, -1.00000000e+000, 0.00000000e+000, 0.00000000e+000,
               0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
               0.00000000e+000, 1.0]
        # saving normalization values to csv
        with open(file + '.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',', dialect='unix')
            writer.writerow(max)
            writer.writerow(min)

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
