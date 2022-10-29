import pandas as pd
import numpy as np
from naive_bayes import train_NBC, predict_NBC, print_NBC_model
from train_utils import train_test_split, compute_accuracy, count_categories_per_feature, one_hot_encode
from trivial_classifier import trivial_train, trivial_predict
from logistic_regression import train_logistic_regression, predict_logistic_regression

# General Configs
CATEGORICAL_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_A_X.csv"
CATEGORICAL_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_A_Y.csv"
CONTINUOUS_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_B_X.csv"
CONTINUOUS_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_B_Y.csv"
MIX_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_C_X.csv"
MIX_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_C_Y.csv"

# Naive Bayes Configs
L = 1.0
CATEGORICAL = "categorical"
CONTINUOUS = "continuous"


# Logistic Regression Configs
L1 = 'l1'
L2 = 'l2'
REGULARIZATION = L2
C = 1 / L

if __name__ == "__main__":
    X = pd.read_csv(CATEGORICAL_DATA_X_FP, header=None)
    y = pd.read_csv(CATEGORICAL_DATA_Y_FP, header=None)
    X_train, y_train, X_test, y_test = train_test_split(X=X, Y=y)
    D_categorical = count_categories_per_feature(X=X)
    # Trivial Train
    model = trivial_train(X=X_train, y=y_train)
    predictions = trivial_predict(X=X_test, model=model)
    accuracy = compute_accuracy(predictions=predictions, actual_y=y_test.values.tolist())
    print('Trivial Accuracy:', accuracy)
    # NBC Train
    # model = train_NBC(X=X_train, Y=y_train, X_dtype=CATEGORICAL, L=L, D_categorical=D_categorical)
    # Logistic Regression Train
    X_encoded = one_hot_encode(X)
    X_train, y_train, X_test, y_test = train_test_split(X=X_encoded, Y=y)
    model = train_logistic_regression(X=X_train, y=y_train, C=C, regularization=REGULARIZATION)
    predictions = predict_logistic_regression(X=X_test, model=model)
    accuracy = compute_accuracy(predictions=predictions, actual_y=y_test.values.tolist())
    print('Logistic Regression Accuracy:', accuracy)
