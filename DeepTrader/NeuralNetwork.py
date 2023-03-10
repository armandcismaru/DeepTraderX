# pylint: disable=invalid-name
"""
Module containing the base class for neural networks.
"""

import os
import csv
import numpy as np
from keras.models import model_from_json
from pickle_data import MAX_VALUES, MIN_VALUES


# pylint: disable=invalid-name,missing-function-docstring,no-member
class NeuralNetwork:
    """
    Base class for neural networks.
    """

    def __init__(self):
        pass

    def save(self):
        # create new directory if not already there
        path = "./Models/" + self.filename + "/"
        file = path + self.filename
        try:
            os.system("mkdir " + path)
        except FileExistsError:
            pass

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file + ".json", "w", encoding="utf-8") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(file + ".h5")

        # saving normalization values to csv
        with open(file + ".csv", "w", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, delimiter=",", dialect="unix")
            writer.writerow(MAX_VALUES)
            writer.writerow(MIN_VALUES)

        print("Saved model to disk.")

    @staticmethod
    def load_network(filename):
        # path directory variables
        path = "./Models/" + filename + "/"
        file = path + filename

        # load json and create model
        with open(file + ".json", "r", encoding="utf-8") as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(file + ".h5")

        print("Loaded model from disk.")
        return loaded_model

    @staticmethod
    def normalization_values(filename):
        # path directory variables
        path = "./Models/" + filename + "/"
        file = path + filename

        # values used to normalize training data
        max_vals = np.array([])
        min_vals = np.array([])

        with open(file + ".csv", "r", encoding="utf-8") as f:
            f_data = list(csv.reader(f))
            max_vals = np.array([float(f.strip()) for f in f_data[0]])
            min_vals = np.array([float(f.strip()) for f in f_data[1]])

        return max_vals, min_vals
