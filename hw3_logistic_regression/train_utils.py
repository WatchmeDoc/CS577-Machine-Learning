import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

TRAIN_SET_PERCENTAGE = 0.75


def train_test_split(X: pd.DataFrame, Y: pd.DataFrame):
    if len(X) != len(Y):
        raise ValueError("Dimensions between X and Y must match!")
    msk = np.random.rand(len(X)) < TRAIN_SET_PERCENTAGE
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
            f"Dimensions between predictions and actual_y must match! Predictions is {total_predictions} and actual_y is {len(actual_y)}")
    score = 0
    for i in range(total_predictions):
        score += 1 if predictions[i] == actual_y[i] else 0
    return score / total_predictions


def count_categories_per_feature(X: pd.DataFrame):
    counts = []
    for index in X.columns:
        counts.append(len(X[index].value_counts()))
    return pd.DataFrame(counts)


def one_hot_encode(X: pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown='ignore')
    return pd.DataFrame(encoder.fit_transform(X).toarray())
