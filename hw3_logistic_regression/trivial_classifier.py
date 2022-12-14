import numpy as np
import pandas as pd


def trivial_train(X: pd.DataFrame, y: pd.DataFrame):
    return np.argmax(y.value_counts())


def trivial_predict(X, model):
    predictions = []
    for _ in X.iterrows():
        predictions.append(model)
    return predictions
