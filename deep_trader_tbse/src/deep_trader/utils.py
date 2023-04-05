# pylint: disable=invalid-name,too-many-locals,broad-exception-caught
"""
Utiliy to pickle the data from csv files to a single file.
"""

import os
import csv
import io
import pickle
import sys
import numpy as np
import pandas as pd
import boto3
from progress.bar import ShadyBar

MAX_VALUES = [
    3612.903986,
    1.0,
    275.0,
    373.5,
    472.153846,
    1.0,
    500.0,
    267.0,
    500.0,
    589.662766,
    38.0,
    79.218286,
    275.0,
    275.0,
]

MIN_VALUES = [
    30.232129,
    0.0,
    1.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.00834,
    0.0,
    0.0,
    30.709072,
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

def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def pickle_s3_files(pkl_path):
    """
    Pickle the data from csv files to a single file.
    """

    # Set the S3 bucket and prefix
    bucket_name = "output-data-fz19792"

    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Get the list of objects
    paginator = s3.get_paginator("list_objects_v2")

    # Get the number of pages of objects
    pages = paginator.paginate(Bucket=bucket_name)

    no_objects = 0
    for page in pages:
        no_objects += len(page["Contents"])

    pages = paginator.paginate(Bucket=bucket_name)
    count = 0
    try:
        for page in pages:
            for obj in page["Contents"]:
                file_list = []
                if obj["Key"].endswith(".csv"):
                    csv_obj = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
                    csv_body = csv_obj["Body"].read().decode("utf-8")
                    f_data = csv.reader(io.StringIO(csv_body))

                    for row in f_data:
                        file_list.append(row)
                    count += 1
                    progressBar(count, no_objects)

                with open(pkl_path, "ab") as fileobj:
                    pickle.dump(file_list, fileobj)  
    except Exception as e:
        print(e)


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

    pkl_path = "normalized_data.pkl"
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
    train_data_path = "train_data.pkl"
    pickle_s3_files(train_data_path)
    normalize_train(train_data_path)
