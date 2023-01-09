import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logistic_regression import predict_logistic_regression, train_logistic_regression
from naive_bayes import predict_NBC, print_NBC_model, train_NBC
from tqdm import tqdm
from train_utils import (
    compute_accuracy,
    find_categorical_vars,
    k_percent_first_rows,
    one_hot_encode,
    train_test_split,
)
from trivial_classifier import trivial_predict, trivial_train

# General Configs
CATEGORICAL_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_A_X.csv"
CATEGORICAL_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_A_Y.csv"
CONTINUOUS_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_B_X.csv"
CONTINUOUS_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_B_Y.csv"
MIX_DATA_X_FP = "Assignment3_DataCode/Dataset3.2_C_X.csv"
MIX_DATA_Y_FP = "Assignment3_DataCode/Dataset3.2_C_Y.csv"
_DATASET_MAP = {
    CATEGORICAL_DATA_X_FP: "categorical",
    CONTINUOUS_DATA_X_FP: "continuous",
    MIX_DATA_X_FP: "mixed",
}
_TRIVIAL_KEY = "trivial_classifier"
_NB_KEY = "naive_bayes"
_LR_KEY = "logistic_regression"
_CLASSIFIERS = [_TRIVIAL_KEY, _NB_KEY, _LR_KEY]
K_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REPEATS = 100

# Naive Bayes Configs
L = 1.0

# Logistic Regression Configs
L1 = "l1"
L2 = "l2"
REGULARIZATION = L2
C = 1 / L


def print_acc_dict(accuracy_per_classifier):
    for dataset_x, _ in datasets:
        print(f"{_DATASET_MAP[dataset_x]} : {'{'}")
        for classifier in _CLASSIFIERS:
            print(f"\t{classifier} : {'{'}")
            for K in K_values:
                mean_acc = np.mean(accuracy_per_classifier[dataset_x][classifier][K])
                print(f"\t\t{K} : {mean_acc}")
            print("\t}")
        print("}")


def plot_acc_dict(accuracy_per_classifier):
    for dataset_x, _ in datasets:
        plt.figure()
        plt.title(_DATASET_MAP[dataset_x])
        for classifier in _CLASSIFIERS:
            acc_per_k = []
            for K in K_values:
                mean_acc = np.mean(accuracy_per_classifier[dataset_x][classifier][K])
                acc_per_k.append(mean_acc)
            plt.scatter(K_values, acc_per_k)
            plt.plot(K_values, acc_per_k, label=classifier)
        plt.xlabel("K percentage")
        plt.ylabel("Mean Accuracy")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    datasets = [
        (CATEGORICAL_DATA_X_FP, CATEGORICAL_DATA_Y_FP),
        (CONTINUOUS_DATA_X_FP, CONTINUOUS_DATA_Y_FP),
        (MIX_DATA_X_FP, MIX_DATA_Y_FP),
    ]
    accuracy_per_classifier = {}
    for dataset_x, _ in datasets:
        accuracy_per_classifier[dataset_x] = {}
        for classifier in _CLASSIFIERS:
            accuracy_per_classifier[dataset_x][classifier] = {}
            for K in K_values:
                accuracy_per_classifier[dataset_x][classifier][K] = []

    for dataset_x, dataset_y in datasets:

        X = pd.read_csv(dataset_x, header=None)
        y = pd.read_csv(dataset_y, header=None)
        if dataset_x == CATEGORICAL_DATA_X_FP:
            X_encoded = one_hot_encode(X=X)
        else:
            X_encoded = X
        categorical_vars = find_categorical_vars(X=X)
        for _ in tqdm(range(REPEATS)):
            X_train_full, y_train_full, X_test, y_test = train_test_split(X=X, Y=y)
            (
                X_train_encoded_full,
                y_train_encoded_full,
                X_test_encoded,
                y_test_encoded,
            ) = train_test_split(X=X_encoded, Y=y)
            for K in K_values:
                X_train = k_percent_first_rows(X=X_train_full, k=K)
                y_train = k_percent_first_rows(X=y_train_full, k=K)
                # Trivial Train
                model = trivial_train(X=X_train, y=y_train)
                predictions = trivial_predict(X=X_test, model=model)
                accuracy = compute_accuracy(
                    predictions=predictions, actual_y=y_test.values.tolist()
                )
                accuracy_per_classifier[dataset_x][_TRIVIAL_KEY][K].append(accuracy)

                # NBC Train
                model = train_NBC(
                    X=X_train, Y=y_train, L=L, categorical_variables=categorical_vars
                )
                predictions = predict_NBC(model=model, X=X_test)
                # print_NBC_model(model=model)
                accuracy = compute_accuracy(
                    predictions=predictions, actual_y=y_test.values.tolist()
                )
                accuracy_per_classifier[dataset_x][_NB_KEY][K].append(accuracy)

                # Logistic Regression Train
                X_train = k_percent_first_rows(X=X_train_encoded_full, k=K)
                y_train = k_percent_first_rows(X=y_train_encoded_full, k=K)
                model = train_logistic_regression(
                    X=X_train, y=y_train, C=C, regularization=REGULARIZATION
                )
                predictions = predict_logistic_regression(X=X_test_encoded, model=model)
                accuracy = compute_accuracy(
                    predictions=predictions, actual_y=y_test_encoded.values.tolist()
                )
                accuracy_per_classifier[dataset_x][_LR_KEY][K].append(accuracy)

    print_acc_dict(accuracy_per_classifier=accuracy_per_classifier)
    plot_acc_dict(accuracy_per_classifier=accuracy_per_classifier)
