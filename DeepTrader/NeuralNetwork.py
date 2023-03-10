# pylint: disable=invalid-name
"""
Module containing the base class for neural networks.
"""

import os
import csv
import numpy as np
from keras.models import model_from_json


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
        maxValues = [
            6.0000000e02,
            1.0000000e00,
            2.2400000e02,
            6.0950000e02,
            9.4987500e02,
            1.0000000e00,
            1.1130000e03,
            2.2200000e02,
            1.1130000e03,
            2.3550000e01,
            6.7000000e01,
            7.8813584e01,
            1.9753013e02,
            2.2400000e02,
        ]
        minValues = [
            0.00000000e000,
            0.00000000e000,
            0.0,
            0.00000000e000,
            0.00000000e000,
            -1.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            0.00000000e000,
            1.0,
        ]

        # saving normalization values to csv
        with open(file + ".csv", "w", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, delimiter=",", dialect="unix")
            writer.writerow(maxValues)
            writer.writerow(minValues)

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
