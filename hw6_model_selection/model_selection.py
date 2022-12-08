import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    df = pd.read_csv('data/Dataset6.A_XY.csv', header=None)
    x_cols = df.columns[:-1]
    y_col = df.columns[-1]
    x_train = df[x_cols]
    y_train = df[y_col]
    # For each fold
    validation_set = create_folds(data=df)
    scaler, model_chosen = CV(data=df, validation_indices=validation_set, configurations=CLASSIFIERS_TO_TRAIN)

    # Part B
    test_df = pd.read_csv('data/Dataset6.B_XY.csv', header=None)
    x_test = test_df[x_cols]
    y_test = test_df[y_col]
    if scaler is not None:
        x_test = scaler.transform(x_test)

    plt.figure()
    # AUC score and ROC for best classifier:
    y_pred_proba = model_chosen.predict_proba(x_test)[::, 1]
    hold_out_auc = roc_auc_score(y_test, y_pred_proba)
    print("Hold-Out Set AUC for best config:", hold_out_auc)
    label = "AUC={:0.3f}".format(hold_out_auc)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=2.0)
    plt.plot(fpr, tpr, label=label)

    # AUC and ROC for trivial classifier:
    y_pred_proba_trivial = np.ones(len(y_pred_proba)) * y_train.value_counts(normalize=True).max()
    trivial_auc = roc_auc_score(y_test, y_pred_proba_trivial)
    print("Hold-Out Set AUC for trivial classifier:", trivial_auc)
    label = "Trivial AUC={:0.3f}".format(trivial_auc)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_trivial, pos_label=2.0)
    plt.plot(fpr, tpr, label=label)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()

