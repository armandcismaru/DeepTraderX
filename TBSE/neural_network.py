# pylint: disable=invalid-name
"""
Module containing the base class for neural networks.
"""

import os
import csv
import numpy as np
from keras.models import model_from_json
from utils import MAX_VALUES, MIN_VALUES, read_data


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
    def test(model, X, y, verbose):
        """Test the model."""

        n_features = 13
        normalized_output = model.predict(X, verbose=verbose)

        yhat = (
            (normalized_output) * (MAX_VALUES[n_features] - MIN_VALUES[n_features])
        ) + MIN_VALUES[n_features]

        print(y, yhat)

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
    
    @staticmethod
    def test_models(no_files):
        """
        Test the models.
        """

        n_features = 13
        X, y = read_data(no_files)
        max_values, min_values = NeuralNetwork.normalization_values("DeepTrader1_6")

        # split_ratio = [9, 1]
        # train_X, test_X = split_train_test_data(X, split_ratio)
        # train_X = train_X.reshape((-1, 1, 1))
        # test_X = test_X.reshape((-1, 1, 1))
        # _ , test_y = split_train_test_data(y, split_ratio)

        model = NeuralNetwork.load_network("DeepTrader1_6")
        for i in range(X.shape[0]):
            normalized_input = (X[i] - min_values[:n_features]) / (
                    max_values[:n_features] - min_values[:n_features]
                )
            normalized_input = np.reshape(normalized_input, (1, 1, -1))

            print(normalized_input)
            print(normalized_input.shape)
            NeuralNetwork.test(model, normalized_input, y[i], verbose=1)


if __name__ == "__main__":
    NeuralNetwork.test_models(30)

