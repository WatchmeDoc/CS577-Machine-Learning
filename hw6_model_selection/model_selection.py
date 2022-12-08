import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Add more configs in a similar manner; First argument is whether to use StandardScaler on X or not,
# then the chosen classifier algorithm and finally a list of possible configurations for that classifier.
CLASSIFIERS_TO_TRAIN = [(True, LogisticRegression, [{'C': 1}, {'C': 3}, {'C': 5}]),
                        (False, LogisticRegression, [{'C': 1}, {'C': 3}, {'C': 5}]),
                        (True, DecisionTreeClassifier, [{'min_samples_leaf': 1, 'max_features': None},
                                                        {'min_samples_leaf': 4, 'max_features': None},
                                                        {'min_samples_leaf': 10, 'max_features': None},
                                                        {'min_samples_leaf': 1, 'max_features': "sqrt"},
                                                        {'min_samples_leaf': 4, 'max_features': "sqrt"},
                                                        {'min_samples_leaf': 10, 'max_features': "sqrt"}
                                                        ]
                         ),
                        (False, DecisionTreeClassifier, [{'min_samples_leaf': 1, 'max_features': None},
                                                        {'min_samples_leaf': 4, 'max_features': None},
                                                        {'min_samples_leaf': 10, 'max_features': None},
                                                        {'min_samples_leaf': 1, 'max_features': "sqrt"},
                                                        {'min_samples_leaf': 4, 'max_features': "sqrt"},
                                                        {'min_samples_leaf': 10, 'max_features': "sqrt"}
                                                        ]
                         )
                        ]


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
    validation_set = []
    for _, test_index in skf.split(X, y):
        validation_set.append(test_index)
    return validation_set


def CV(data, validation_indices, configurations):
    """
    This function selects the best configuration and computes its performance
    via the cross validation procedure over the k validation_indices.
    :param data: Input dataset to apply Cross Validation model selection to
    :param validation_indices: List of k-arrays, where each array contains the validation set indices
    :param configurations: See above the structure of configurations list
    :return: best configuration including the mean AUC score as resulted from CV protocol.
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
    return best_config


if __name__ == "__main__":
    df = pd.read_csv('data/Dataset6.A_XY.csv', header=None)
    validation_set = create_folds(data=df)
    model_chosen = CV(data=df, validation_indices=validation_set, configurations=CLASSIFIERS_TO_TRAIN)
    print(model_chosen)
