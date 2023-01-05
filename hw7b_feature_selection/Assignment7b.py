from sklearn.datasets import load_breast_cancer
from causallearn.utils.cit import CIT
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import matplotlib.pyplot as plt

TRAIN_SET_PERCENTAGE = 0.8
CLASSIFIERS_TO_TRAIN = [(True, SVC, [{'C': 1, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                                     {'C': 2, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                                     {'C': 5, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                                     {'C': 1, 'kernel': 'rbf', 'gamma': 0.1, 'probability': True},
                                     {'C': 2, 'kernel': 'rbf', 'gamma': 0.1, 'probability': True},
                                     {'C': 2, 'kernel': 'rbf', 'gamma': 1, 'probability': True},
                                     ]),
                        (True, DecisionTreeClassifier, [{'min_samples_leaf': 1, 'max_features': None},
                                                        {'min_samples_leaf': 4, 'max_features': None},
                                                        {'min_samples_leaf': 10, 'max_features': None},
                                                        {'min_samples_leaf': 10, 'max_features': "sqrt"}
                                                        ]
                         ),
                        (False, DecisionTreeClassifier, [{'min_samples_leaf': 1, 'max_features': None},
                                                         {'min_samples_leaf': 1, 'max_features': "sqrt"},
                                                         {'min_samples_leaf': 4, 'max_features': "sqrt"},
                                                         {'min_samples_leaf': 10, 'max_features': "sqrt"}
                                                         ]
                         )
                        ]


# SEE THE PARAMETERS' INFO FROM THE PDF
def stat_test(D, V_idx, T_idx, S=None):
    kci_obj = CIT(D, "kci")  # construct a CIT instance with data and method name
    pValue = kci_obj(V_idx, T_idx, S)
    return pValue


def forward_selection(D, V_indices, T_idx, S, a):
    """
    Forward Selection algorithm
    :param D: Full Dataset
    :param V_indices: Variable column indices
    :param T_idx: Target variable column index
    :param S: Previously selected Variables
    :param a: Significance Threshold
    :return S: Selected Variables
    """
    R = V_indices.drop(S)
    flag = True
    while flag:
        flag = False
        V_star, p_value = min([(V_idx, stat_test(D=D.values, V_idx=V_idx, T_idx=T_idx, S=S)) for V_idx in R],
                              key=lambda x: x[1])
        R = R.drop(V_star)
        if p_value <= a:
            flag = V_star not in S
            S = S.add(V_star)
    return S


def backward_selection(D, T_idx, S, a):
    """
    Backward Selection algorithm
    :param D: Full Dataset
    :param T_idx: Target variable index
    :param S: Previously Selected Variables
    :param a: Significance Threshold
    :return S: Selected Variables
    """
    flag = True
    while flag:
        flag = False
        V_star, p_value = max(
            [(V_idx, stat_test(D=D, V_idx=V_idx, T_idx=T_idx, S=S.drop(V_idx))) for V_idx in S],
            key=lambda x: x[1])
        if p_value > a:
            flag = V_star not in S
            S = S.drop(V_star)
    return S


def create_folds(data, k=5):
    """
    Splits the data into k stratified folds and returns a list of k arrays
    of indices that will be used for validation set in each fold.
    :param data: Input dataset
    :param k: Number of Splits
    :return: List of K-arrays, where each array contains the indices
    """
    y: pd.DataFrame = data[data.columns[-1]]
    X = data[data.columns[:-1]]
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    return [test_index for _, test_index in skf.split(X, y)]


def stratified_train_test_split(data: pd.DataFrame, T_idx):
    counts = data[T_idx].value_counts(normalize=True)
    print(counts)
    minor_class = counts.index[counts.argmin()]
    data_minor = data[data[T_idx] == minor_class]
    data_major = data[data[T_idx] != minor_class]

    def train_test_split(data):
        arr_rand = np.random.rand(data.shape[0])
        msk = arr_rand < np.percentile(arr_rand, TRAIN_SET_PERCENTAGE * 100)
        train = data[msk]
        test = data[~msk]
        return train, test

    train_minor, test_minor = train_test_split(data=data_minor)
    train_major, test_major = train_test_split(data=data_major)
    train = pd.concat([train_major, train_minor])
    test = pd.concat([test_major, test_minor])
    return train, test


def CV(data, validation_indices, configurations):
    """
    This function selects the best configuration and computes its performance
    via the cross validation procedure over the k validation_indices.
    :param data: Input dataset to apply Cross Validation model selection to
    :param validation_indices: List of k-arrays, where each array contains the validation set indices
    :param configurations: See above the structure of configurations list
    :return: scaler if applicable and model instance, built from the best configuration
    """
    # Init model perf data structure
    model_perf = {}
    for standardization, classifier, hyperparams in configurations:
        if model_perf.get(standardization) is None:
            model_perf[standardization] = {}
        if model_perf[standardization].get(classifier) is None:
            model_perf[standardization][classifier] = {}
        for index, kwargs in enumerate(hyperparams):
            if model_perf[standardization][classifier].get(index) is None:
                model_perf[standardization][classifier][index] = []
    x_cols = data.columns[:-1]
    y_col = data.columns[-1]
    # For each fold
    for test_indices in validation_indices:
        train_set = data.drop(test_indices)
        test_set = data.loc[test_indices]
        # X, Y split for train/test sets
        x_train = train_set[x_cols]
        y_train = train_set[y_col]
        x_test = test_set[x_cols]
        y_test = test_set[y_col]
        # for each configuration
        for standardization, classifier, hyperparams in configurations:
            if standardization is not None:
                scaler = StandardScaler()
                # Fit scaler ONLY on the train data (golden rule)
                scaler.fit(x_train)
                x_train = pd.DataFrame(scaler.transform(x_train))
                x_test = pd.DataFrame(scaler.transform(x_test))
            for index, kwargs in enumerate(hyperparams):
                # Train classifier and calculate AUC score
                clf = classifier(**kwargs)
                clf.fit(X=x_train, y=y_train)
                model_perf[standardization][classifier][index].append(
                    roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
    # Calculate mean AUC score for each classifier
    model_perf_list = []
    for standardization, classifier, hyperparams in configurations:
        for index, kwargs in enumerate(hyperparams):
            model_perf_list.append(
                (standardization, classifier, kwargs, np.mean(model_perf[standardization][classifier][index]))
            )
    # print(model_perf_list)
    # Select config with max mean AUC score
    best_config = max(model_perf_list, key=lambda x: x[3])
    print("Best configuration:")
    print(best_config)
    standardization, classifier, kwargs, _ = best_config
    if standardization is True:
        scaler = StandardScaler()
        scaler.fit(data[x_cols])
        data[x_cols] = scaler.transform(data[x_cols])
    else:
        scaler = None
    clf = classifier(**kwargs)
    clf.fit(X=data[x_cols], y=data[y_col])
    return scaler, clf


if __name__ == '__main__':
    D = pd.DataFrame(load_breast_cancer(as_frame=True).frame.values)
    V_indices = np.arange(D.shape[1] - 1)
    T_idx = D.shape[1] - 1

    # START WRITING YOUR CODE (YOU CAN CREATE AS MANY FUNCTIONS AS YOU WANT)
    train_set, hold_out_set = stratified_train_test_split(data=D, T_idx=T_idx)
    validation_set = create_folds(data=train_set)
    scaler, model_chosen = CV(data=train_set.reset_index(drop=True), validation_indices=validation_set, configurations=CLASSIFIERS_TO_TRAIN)

    x_test = hold_out_set[V_indices]
    y_test = hold_out_set[T_idx]
    if scaler is not None:
        x_test = scaler.transform(x_test)

    y_pred_proba = model_chosen.predict_proba(x_test)[::, 1]
    hold_out_auc = roc_auc_score(y_test, y_pred_proba)
    print("Hold-Out Set AUC for best config:", hold_out_auc)
