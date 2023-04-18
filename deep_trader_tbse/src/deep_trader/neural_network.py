"""
Module containing the base class for neural networks.
"""

import os
import csv
import numpy as np
from keras.models import model_from_json
from .utils import MAX_VALUES, MIN_VALUES, read_data


class NeuralNetwork:
    """Base class for neural networks."""

    def __init__(self):
        self.filename = None
        self.model = None

    def save(self):
        """Save the model to disk."""

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
        """Load network from disk."""

        # path directory variables
        path = "./src/deep_trader/Models/" + filename + "/"
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
        """Load normalization values from CSV file."""

        # path directory variables
        path = "./src/deep_trader/Models/" + filename + "/"
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
    def test(model, x, y, verbose):
        """Test the model."""

        n_features = 13
        normalized_output = model.predict(x, verbose=verbose)

        yhat = (
            (normalized_output) * (MAX_VALUES[n_features] - MIN_VALUES[n_features])
        ) + MIN_VALUES[n_features]

        print(y, yhat)

    @staticmethod
    def test_models(no_files):
        """Test the models."""

        n_features = 13
        x, y = read_data(no_files)
        max_values, min_values = NeuralNetwork.normalization_values("DeepTrader2_2")

        model = NeuralNetwork.load_network("DeepTrader2_2")
        for i in range(x.shape[0]):
            normalized_input = (x[i] - min_values[:n_features]) / (
                max_values[:n_features] - min_values[:n_features]
            )
            normalized_input = np.reshape(normalized_input, (1, 1, -1))
            NeuralNetwork.test(model, normalized_input, y[i], verbose=1)


if __name__ == "__main__":
    NeuralNetwork.test_models(30)
    # import visualkeras
    # model_to_see = NeuralNetwork.load_network("DeepTrader2_2")
    # visualkeras.layered_view(model_to_see, to_file='output.png', legend=True) # write to disk
