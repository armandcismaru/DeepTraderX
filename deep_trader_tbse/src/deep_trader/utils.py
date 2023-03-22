# pylint: disable=invalid-name
"""
Utiliy to pickle the data from csv files to a single file.
"""

import os
import csv
import io
import pickle
import numpy as np
import pandas as pd
import boto3
from progress.bar import ShadyBar

MAX_VALUES = [
    3.6e3,
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

MIN_VALUES = [
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

BATCHSIZE = 1638
NUMBER_OF_FEATURES = 13
NUMBER_OF_STEPS = 1


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
                if sum(1 for _ in file) < 2:
                    continue
                for row in f_data:
                    file_list.append(row)

            with open(pkl_path, "ab") as fileobj:
                pickle.dump(file_list, fileobj)
        except FileNotFoundError as e:
            print(e)

        loading_bar.next()
    loading_bar.finish()


def pickle_s3_files(pkl_path):
    """
    Pickle the data from csv files to a single file.
    """

    # Set the S3 bucket and prefix
    bucket_name = "output-data-fz19792"

    # Initialize the S3 client
    s3 = boto3.client("s3")

    # List the objects in the bucket with the given prefix
    response = s3.list_objects_v2(Bucket=bucket_name)
    loading_bar = ShadyBar("Pickling Data", max=len(response["Contents"]))

    file_list = []
    for obj in response["Contents"]:
        if obj["Key"].endswith(".csv"):
            csv_obj = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
            csv_body = csv_obj["Body"].read().decode("utf-8")
            f_data = csv.reader(io.StringIO(csv_body))
            for row in f_data:
                file_list.append(row)
        loading_bar.next()

    loading_bar.finish()

    with open(pkl_path, "ab") as fileobj:
        pickle.dump(file_list, fileobj)


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


def normalize_train(data_path):
    """
    Normalize the data in the train_data.pkl file.
    """

    pkl_path = "deep_trader_tbse/src/deep_trader/normalized_data.pkl"
    os.system("touch " + pkl_path)
    with open(data_path, "rb") as f:
        while 1:
            try:
                file = np.array(pickle.load(f)).astype(float)
                if file.shape[0] == 0:
                    continue
                for i, _ in enumerate(MAX_VALUES):
                    file[:, i] = normalize_data(
                        file[:, i], MAX_VALUES[i], MIN_VALUES[i], False
                    )

                with open(pkl_path, "ab") as fileobj:
                    pickle.dump(file, fileobj)

            except EOFError:
                break  # no more data in the file
            except ValueError as e:
                print(e)


def read_data(no_files):
    """
    Read the data from the csv file.
    """

    X = np.empty((0, 13))
    y = np.empty((0, 1))
    for i in range(no_files):
        filename = f"trial{(i+1):07d}.csv"
        try:
            data = pd.read_csv(filename)
            X_file = data.iloc[:, :13].values
            y_file = data.iloc[:, 13].values.reshape(-1, 1)

            # Append to the arrays
            X = np.vstack((X, X_file))
            y = np.vstack((y, y_file))

        except FileNotFoundError as e:
            print(e)

    return X, y


def split_train_test_data(data, ratio):
    """
    Split the data into train and test data.
    """

    A = np.array([])
    B = np.array([])

    split_index = int(ratio[0] / (ratio[0] + ratio[1]) * len(data))

    A = np.append(A, data[:split_index])
    B = np.append(B, data[split_index:])

    return A, B


if __name__ == "__main__":
    train_data_path = "deep_trader_tbse/src/deep_trader/train_data.pkl"
    pickle_s3_files(train_data_path)
    normalize_train(train_data_path)
