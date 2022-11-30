from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import numpy as np
import pandas as pd

Y_INDEX = 'Y is house valuable'
TRAIN_SET_PERCENTAGE = 0.7


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
            f"Dimensions between predictions and actual_y must match! Predictions is {total_predictions} and actual_y is {len(actual_y)}")
    score = 0
    for i in range(total_predictions):
        score += 1 if predictions[i] == actual_y[i] else 0
    return score / total_predictions


def TrainRF(X: pd.DataFrame, Y: pd.Series, n_trees, min_samples_leaf):
    """
    Trains a Random Forest classifier using the provided hyperparameters.
    :param X: The sample data (samples along rows, features along columns)
    :param Y: The label vector. Y is the class variable we want to predict.
    :param n_trees: The number of trees to grow
    :param min_samples_leaf: minimum number of leaf node observations
    :return model: A list of DecisionTreeClassifiers constructing the RandomForestClassifier
    """
    forest = []
    for _ in range(n_trees):
        X_bs, y_bs = resample(X, Y, replace=True)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_features="sqrt")
        model = model.fit(X_bs, y_bs)
        forest.append(model)
    return forest


def PredictRF(model, X):
    """
    Using the provided Random Forest model, this function classifies the provided data.
    :param model: A model previously trained using TrainRF.
    :param X: A matrix of data to be classified (samples along rows, features along columns)
    :return predictions: The vector of predicted classes for each of the input samples
    """
    predictions = []
    for tree in model:
        tree_predictions = tree.predict(X)
        predictions.append(tree_predictions)
    predictions = np.array(predictions).sum(axis=0)
    return list(map(lambda item: 0 if item < len(model) / 2 else 1, predictions))


if __name__ == "__main__":
    df = pd.read_csv('data/Dataset5_XY.csv')
    X = df[df.columns.drop(Y_INDEX)]
    y = df[Y_INDEX]
    X_train, y_train, X_test, y_test = train_test_split(X=X, Y=y)
    # Personal implementation
    model = TrainRF(X=X_train, Y=y_train, n_trees=10, min_samples_leaf=5)
    y_pred = PredictRF(model=model, X=X_test)
    print(compute_accuracy(predictions=y_pred, actual_y=y_test.values.tolist()))
    # VS SKLearn
    model = RandomForestClassifier(n_estimators=10, min_samples_leaf=5, max_features='sqrt')
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
