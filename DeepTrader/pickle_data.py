"""
Utiliy to pickle the data from csv files to a single file.
"""

import os
import csv
import pickle
import numpy as np
from progress.bar import ShadyBar


def pickle_files(pkl_path, no_files):
    """
    Pickle the data from csv files to a single file.
    """

    loading_bar = ShadyBar("Pickling Data", max=no_files)

    # retrieving data from multiple files
    for i in range(no_files):
        filename = f"trial{(i+1):07d}.csv"
        file_list = []
        try:
            with open(filename, "r", encoding="utf-8") as file:
                f_data = csv.reader(file)
                for row in f_data:
                    file_list.append(row)

            with open(pkl_path, "ab") as fileobj:
                pickle.dump(file_list, fileobj)
        except FileNotFoundError:
            pass

        loading_bar.next()
    loading_bar.finish()


# pylint: disable=invalid-name
def normalize_data(X, max_values=0, min_values=0, train=True):
    """
    Normalize the data.
    """

    if train:
        max_values = np.max(X)
        min_values = np.min(X)

    normalized = (X - min_values) / (max_values - min_values)
    return normalized


def normalize_train():
    """
    Normalize the data in the train_data.pkl file.
    """

    pkl_path = "normalized_data.pkl"
    os.system("touch " + pkl_path)
    with open("train_data.pkl", "rb") as fileobj:
        max_values = [
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
        min_values = [
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

        while 1:
            try:
                file = np.array(pickle.load(fileobj)).astype(float)
                if file.shape[0] == 0:
                    continue
                for i, _ in enumerate(max_values):
                    file[:, i] = normalize_data(
                        file[:, i], max_values[i], min_values[i], False
                    )

                with open(pkl_path, "ab", encoding="utf-8") as fileobj:
                    pickle.dump(file, fileobj)

            except EOFError:
                break  # no more data in the file
            except ValueError:
                pass


if __name__ == "__main__":
    pickle_files("train_data.pkl", 30)
    normalize_train()
