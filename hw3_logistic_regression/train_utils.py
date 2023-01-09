import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TRAIN_SET_PERCENTAGE = 0.75
CATEGORICAL_VALUES_THRESHOLD = 15


def train_test_split(X: pd.DataFrame, Y: pd.DataFrame):
    if len(X) != len(Y):
        raise ValueError("Dimensions between X and Y must match!")
    arr_rand = np.random.rand(X.shape[0])
    msk = arr_rand < np.percentile(arr_rand, TRAIN_SET_PERCENTAGE * 100)
    X_train = X[msk]
    X_test = X[~msk]

    Y_train = Y[msk]
    Y_test = Y[~msk]
    if TRAIN_SET_PERCENTAGE == 1.0:
        X_test = X_train
        Y_test = Y_train
    return X_train, Y_train, X_test, Y_test


def compute_accuracy(predictions, actual_y):
    total_predictions = len(predictions)
    if total_predictions != len(actual_y):
        raise ValueError(
            f"Dimensions between predictions and actual_y must match! Predictions is {total_predictions} and actual_y is {len(actual_y)}"
        )
    score = 0
    for i in range(total_predictions):
        score += 1 if predictions[i] == actual_y[i] else 0
    return score / total_predictions


def one_hot_encode(X: pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown="ignore")
    return pd.DataFrame(encoder.fit_transform(X).toarray())


def find_categorical_vars(X: pd.DataFrame):
    categorical_vars = {}
    for col in X.columns:
        if len(X[col].value_counts()) < CATEGORICAL_VALUES_THRESHOLD:
            categorical_vars[col] = len(X[col].value_counts())
    return categorical_vars


def k_percent_first_rows(X: pd.DataFrame, k):
    index = int(np.floor(len(X) * k))
    return X.head(index)
