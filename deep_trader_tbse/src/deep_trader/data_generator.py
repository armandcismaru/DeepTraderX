"""Generates data for feeding into the neural network."""

import pickle
import numpy as np
from keras.utils import Sequence


# pylint: disable=inconsistent-return-statements
class DataGenerator(Sequence):
    """Generates data for the neural network."""

    def __init__(self, dataset_path, batch_size, n_features):
        """Initialization."""

        self.no_items = 0
        self.dataset_path = dataset_path
        with open(self.dataset_path, "rb") as f:
            while 1:
                try:
                    self.no_items += len(pickle.load(f))
                except EOFError:
                    break  # no more data in the file
        print(self.no_items)

        self.batch_size = batch_size
        self.n_features = n_features

        self.train_max = np.empty((self.n_features + 1))
        self.train_min = np.empty((self.n_features + 1))

    def __getitem__(self, index):
        """Generate one batch of data."""

        # Generate indexes of the batch
        indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        x = np.empty((self.batch_size, 1, self.n_features))
        y = np.empty((self.batch_size, 1))

        with open(self.dataset_path, "rb") as f:
            count, number, i = 0, 0, 0
            while 1:
                try:
                    number = len(pickle.load(f)) + count
                    if number < indexes[0]:
                        count = number
                        break
                    if len(pickle.load(f)) == 0:
                        break

                    file = np.array(pickle.load(f))
                    for item in file:
                        item = item.astype(np.float)
                        if count in indexes:
                            x[i,] = np.reshape(item[: self.n_features], (1, -1))
                            y[i,] = np.reshape(item[self.n_features], (1, 1))
                        count += 1
                        i += 1
                        if i > self.batch_size - 1:
                            i = 0
                            return (x, y)

                except EOFError as e:
                    print(e)
                    break  # no more data in the file

    def __len__(self):
        return self.no_items // self.batch_size
