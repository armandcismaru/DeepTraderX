# pylint: disable=too-few-public-methods,E0401
"""
Module containing the architecture class of the model.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from neural_network import NeuralNetwork
from data_generator import DataGenerator
from utils import BATCHSIZE, NUMBER_OF_FEATURES, NUMBER_OF_STEPS


class MultivariateLSTM(NeuralNetwork):
    """
    Class for the LSTM model.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, input_shape, filename):
        """Define the model."""

        NeuralNetwork.__init__(filename, model=Sequential())
        self.input_shape = input_shape
        self.steps = input_shape[1]
        self.n_features = self.input_shape[2]
        self.filename = filename
        self.batch_size = input_shape[0]
        self.max_vals = None
        self.min_vals = None

        # architecture
        self.model.add(
            LSTM(
                10,
                activation="relu",
                input_shape=(input_shape[1], input_shape[2]),
                unroll=True,
            )
        )
        self.model.add(Dense(5, activation="relu"))
        self.model.add(Dense(3, activation="relu"))
        self.model.add(Dense(1))

        # compiling + options
        opt = Adam(learning_rate=1.5e-5)
        self.model.compile(optimizer=opt, metrics=["mae", "msle", "mse"], loss="mse")

    def create_model(self):
        """
        Create the model.
        """

        pkl_path = "./normalized_data.pkl"
        train_data = DataGenerator(pkl_path, self.batch_size, self.n_features)

        self.max_vals = train_data.train_max
        self.min_vals = train_data.train_min

        history = self.model.fit(train_data, epochs=20, verbose=1, workers=28)
        self.save()

        plt.plot(history.history["loss"])
        plt.title("Model loss/cost")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.xticks(np.arange(0, 22, 2))
        plt.legend(["Train"], loc="upper left")
        plt.savefig("loss_curve.png")


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)

    # Multivariate LSTM
    mv = MultivariateLSTM(
        (BATCHSIZE, NUMBER_OF_STEPS, NUMBER_OF_FEATURES), "DeepTrader2_2"
    )
    mv.create_model()
