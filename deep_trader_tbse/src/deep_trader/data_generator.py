"""Generates data for feeding into the neural network."""

import pickle
import numpy as np
from keras.utils import Sequence


# pylint: disable=inconsistent-return-statements
class DataGenerator(Sequence):
    """Generates data for the neural network."""

    def __init__(self, dataset_path, batch_size, n_features):
        """Initialization."""
        super().__init__()
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
            count, i = 0, 0
            while 1:
                try:
                    chunk = pickle.load(f)
                    if not chunk:
                        continue
                    chunk_len = len(chunk)
                    # Skip chunks entirely before the first index
                    if count + chunk_len <= indexes[0]:
                        count += chunk_len
                        continue

                    file = np.array(chunk)
                    for item in file:
                        item = item.astype(np.float64)
                        if count in indexes:
                            x[i,] = np.reshape(item[: self.n_features], (1, -1))
                            y[i,] = np.reshape(item[self.n_features], (1, 1))
                            i += 1
                            if i >= self.batch_size:
                                return (x, y)
                        count += 1

                except EOFError as e:
                    print(e)
                    break  # no more data in the file

    def __len__(self):
        return self.no_items // self.batch_size
