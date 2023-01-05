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
FS_ARGS = 'FS kwargs'
STANDARDIZATION_KEY = 'Standardization'
# Should probably make this a json file at this point, it is getting too large
# Note: SVC might not converge without standardized data. For speed purposes, I always standardize for this classifier.
CLASSIFIERS_TO_TRAIN = [([{STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.05}},
                          {STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.01}},
                          {STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.005}}],
                         SVC,
                         [{'C': 1, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                          {'C': 2, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                          {'C': 5, 'kernel': 'linear', 'gamma': 0.1, 'probability': True},
                          {'C': 1, 'kernel': 'rbf', 'gamma': 0.1, 'probability': True},
                          {'C': 2, 'kernel': 'rbf', 'gamma': 0.1, 'probability': True},
                          {'C': 2, 'kernel': 'rbf', 'gamma': 1, 'probability': True},
                          ]
                         ),
                        ([{STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.05}},
                          {STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.01}},
                          {STANDARDIZATION_KEY: True, FS_ARGS: {'a': 0.005}},
                          {STANDARDIZATION_KEY: False, FS_ARGS: {'a': 0.05}},
                          {STANDARDIZATION_KEY: False, FS_ARGS: {'a': 0.01}},
                          {STANDARDIZATION_KEY: False, FS_ARGS: {'a': 0.005}}
                          ],
                         DecisionTreeClassifier,
                         [{'min_samples_leaf': 1, 'max_features': None},
                          {'min_samples_leaf': 4, 'max_features': None},
                          {'min_samples_leaf': 10, 'max_features': None},
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


def select_features(**kwargs):
    """
    Applies the forward-backward feature selection algorithm
    :param kwargs:
    :return S: Set S of selected features
    """
    pass


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


def get_avg_auc_for_config(data, validation_indices, preprocessing, clf, clf_kwargs, skip_fs):
    model_scores = []
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
        standardization = preprocessing[STANDARDIZATION_KEY]
        if standardization is not None:
            scaler = StandardScaler()
            # Fit scaler ONLY on the train data (golden rule)
            scaler.fit(x_train)
            x_train = pd.DataFrame(scaler.transform(x_train))
            x_test = pd.DataFrame(scaler.transform(x_test))
        if skip_fs is not True:
            fs_kwargs = preprocessing[FS_ARGS]
            selected_features = select_features(fs_kwargs)
        else:
            selected_features = x_cols
        classifier = clf(**clf_kwargs)
        classifier.fit(X=x_train[selected_features], y=y_train)
        model_scores.append(roc_auc_score(y_test, classifier.predict_proba(x_test[selected_features])[:, 1]))
    return np.mean(model_scores)


def select_best_config(data, validation_indices, configurations, skip_fs=False):
    """
    This function selects the best configuration and computes its performance
    via the cross validation procedure over the k validation_indices.
    :param data: Input dataset to apply Cross Validation model selection to
    :param validation_indices: List of k-arrays, where each array contains the validation set indices
    :param configurations: See above the structure of configurations list
    :param skip_fs: Optional parameter, set true if you want to skip Feature Selection for all configs
    :return: scaler if applicable and model instance, built from the best configuration
    """
    x_cols = data.columns[:-1]
    y_col = data.columns[-1]
    model_perf_list = []
    for preprocessing_configs, classifier, hyperparams in configurations:
        for preprocessing in preprocessing_configs:
            for index, kwargs in enumerate(hyperparams):
                config_auc = get_avg_auc_for_config(data=data, validation_indices=validation_indices,
                                                    preprocessing=preprocessing, clf=classifier,
                                                    clf_kwargs=kwargs,
                                                    skip_fs=skip_fs)
                model_perf_list.append((config_auc, preprocessing, classifier, kwargs))
    # print(model_perf_list)
    # Select config with max mean AUC score
    best_config = max(model_perf_list, key=lambda x: x[0])
    print("Best configuration:")
    print(best_config)
    _, preprocessing, classifier, kwargs = best_config
    if preprocessing[STANDARDIZATION_KEY] is True:
        scaler = StandardScaler()
        scaler.fit(data[x_cols])
        data[x_cols] = scaler.transform(data[x_cols])
    else:
        scaler = None
    selected_features = x_cols
    if skip_fs is not True:
        selected_features = selected_features(preprocessing[FS_ARGS])
    clf = classifier(**kwargs)
    clf.fit(X=data[selected_features], y=data[y_col])
    return scaler, selected_features, clf


if __name__ == '__main__':
    D = pd.DataFrame(load_breast_cancer(as_frame=True).frame.values)
    V_indices = np.arange(D.shape[1] - 1)
    T_idx = D.shape[1] - 1

    # START WRITING YOUR CODE (YOU CAN CREATE AS MANY FUNCTIONS AS YOU WANT)
    train_set, hold_out_set = stratified_train_test_split(data=D, T_idx=T_idx)
    validation_set = create_folds(data=train_set)
    scaler, selected_features, model_chosen = select_best_config(data=train_set.reset_index(drop=True),
                                                                 validation_indices=validation_set,
                                                                 configurations=CLASSIFIERS_TO_TRAIN,
                                                                 skip_fs=True)

    x_test = hold_out_set[selected_features]
    y_test = hold_out_set[T_idx]
    if scaler is not None:
        x_test = scaler.transform(x_test)

    y_pred_proba = model_chosen.predict_proba(x_test)[::, 1]
    hold_out_auc = roc_auc_score(y_test, y_pred_proba)
    print("Hold-Out Set AUC for best config:", hold_out_auc)
