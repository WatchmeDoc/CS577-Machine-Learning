import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X: pd.DataFrame, y: pd.DataFrame, C=1.0, regularization='l2'):
    return LogisticRegression(penalty=regularization, C=C).fit(X=X, y=y.values.ravel())


def predict_logistic_regression(X, model):
    return model.predict(X)
