# pylint: disable=too-many-locals,broad-except
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
    0.0,
    0.0,
    0.0,
    30.709072,
    1.0,
]

BATCHSIZE = 16384
NUMBER_OF_FEATURES = 13
NUMBER_OF_STEPS = 1


def pickle_files(pkl_path, no_files):
    """
    Pickle the data from csv files to a single file.
    """

    # retrieving data from multiple files
    for i in range(no_files):
        filename = f"trial{(i+1):07d}.csv"
        file_list = []
        try:
            progress_bar(i + 1, no_files, suffix="Pickle files")
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


def progress_bar(count_value, total, suffix="Pickle files"):
    """
    Progress bar to show the progress of the pickle files.
    """

    bar_length = 100
    filled_up_length = int(round(bar_length * count_value / float(total)))
    percentage = round(100.0 * count_value / float(total), 1)
    loading_bar = "=" * filled_up_length + "-" * (bar_length - filled_up_length)

    sys.stdout.write(f"[{loading_bar}] {percentage}% ...{suffix}\r")
    sys.stdout.flush()


def pickle_s3_files(pkl_path):
    """
    Pickle the data from CSV files in a S3 bucket to a single file.
    """

    # Set the S3 bucket and prefix
    bucket_name = "output-data-fz19792"

    # Initialize the S3 client

    # pylint: disable=line-too-long, invalid-name
    # Bad practice, needs to be changed. The keys below are expired.
    s3 = boto3.client(
        "s3",
        aws_access_key_id="ASIAV4Y55ZUXLRWGBDMC",
        aws_secret_access_key="XfFuMqJd/2N8Waq15TcPr+Oi6GqU3NQLd0PDGgVA",
        aws_session_token="FwoGZXIvYXdzECcaDEZrIeKmyAKMxDImLCLGAYxboLs1W2bfD5W3F1rqiSfwJHZtrM2wCpOd29NPmpOfuBEnmBX7P3bGVr6zKPv5UtNufuov+adpVnVUB2bFEXfLUhAastq5mRAzJxu4MlHjh3XPNJeD+1cIMDN0bJKGUJz3Cs5ATzlFBQIkqExfnJTfKmZ+LCeHEfN1eL76nPsycm8xAuapKK1HKD3JjNgNcnVu5wbHulApZpotf0R186fyrhGAlf/Em5SrCkTmnFFLp/wjxw1TfZz5eAwebhj8ckScrQNcuCjp0bChBjItH++mW1ynFCbXWObDbgRAXTp2SiwT084bHd3F+DKOk2o5gskqQcE1sJ5SdpEY",
    )

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
                    progress_bar(count, no_objects)
                with open(pkl_path, "ab") as fileobj:
                    pickle.dump(file_list, fileobj)
    except Exception as e:
        print(e)


def normalize_data(x, max_values=0, min_values=0, train=True):
    """Normalize the data."""

    if train:
        max_values = np.max(x)
        min_values = np.min(x)

    normalized = (x - min_values) / (max_values - min_values)
    return normalized


def normalize_train(data_path):
    """Normalize the data in the train_data.pkl file."""

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
    """Read the data from the csv file."""

    x = np.empty((0, 13))
    y = np.empty((0, 1))
    for i in range(no_files):
        filename = f"trial{(i+1):07d}.csv"
        try:
            data = pd.read_csv(filename)
            x_file = data.iloc[:, :13].values
            y_file = data.iloc[:, 13].values.reshape(-1, 1)

            # Append to the arrays
            x = np.vstack((x, x_file))
            y = np.vstack((y, y_file))

        except FileNotFoundError as e:
            print(e)

    return x, y


def split_train_test_data(data, ratio):
    """Split the data into train and test data."""

    a = np.array([])
    b = np.array([])

    split_index = int(ratio[0] / (ratio[0] + ratio[1]) * len(data))

    a = np.append(a, data[:split_index])
    b = np.append(b, data[split_index:])

    return a, b


if __name__ == "__main__":
    TRAIN_DATA_PATH = "train_data.pkl"
    pickle_s3_files(TRAIN_DATA_PATH)
    normalize_train(TRAIN_DATA_PATH)
