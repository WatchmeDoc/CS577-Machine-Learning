import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import matplotlib.pyplot as plt

C_VALUES = [0.01, 1, 10]
KERNEL_VALUES = ['linear', 'rbf']
GAMMA_VALUES = [0.1, 1, 10]
TRAIN_SET_PERCENTAGE = 0.7
C_KEY = 'C'
KERNEL_KEY = 'kernel'
GAMMA_KEY = 'gamma'
Y_INDEX = 'Y is house valuable'


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


def train_test_split(data):
    arr_rand = np.random.rand(data.shape[0])
    msk = arr_rand < np.percentile(arr_rand, TRAIN_SET_PERCENTAGE * 100)
    train = data[msk]
    test = data[~msk]
    return train, test


def CV(data, validation_indices):
    """
    This function selects the best configuration and computes its performance
    via the cross validation procedure over the k validation_indices.
    :param data: Input dataset to apply Cross Validation model selection to
    :param validation_indices: List of k-arrays, where each array contains the validation set indices
    :return: scaler if applicable and model instance, built from the best configuration
    """
    x_cols = data.columns[:-1]
    y_col = data.columns[-1]
    config_performances = []
    sets = []
    for test_indices in validation_indices:
        train_set = data.drop(test_indices)
        test_set = data.loc[test_indices]
        x_train = train_set[x_cols]
        y_train = train_set[y_col]
        x_test = test_set[x_cols]
        y_test = test_set[y_col]
        sets.append((x_train, y_train, x_test, y_test))
    # For each fold
    for C in C_VALUES:
        for kernel in KERNEL_VALUES:
            for gamma in GAMMA_VALUES:
                config_performance = []
                kwargs = {C_KEY: C, KERNEL_KEY: kernel, GAMMA_KEY: gamma, 'probability': True}
                print("Trying config:", kwargs)
                for x_train, y_train, x_test, y_test in sets:
                    # Fit classifier with current config
                    scaler = StandardScaler().fit(x_train)
                    x_train = scaler.transform(x_train)
                    x_test = scaler.transform(x_test)
                    clf = SVC(**kwargs)
                    clf.fit(x_train, y_train)
                    # Estimate AUC on test fold
                    config_performance.append(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
                # Store mean AUC score for current config, and current config
                avg_auc = np.mean(config_performance)
                print("Average AUC for config:", avg_auc)
                config_performances.append((avg_auc, kwargs))
    # Select config with max mean AUC score
    best_config = max(config_performances, key=lambda x: x[0])
    print("Best configuration:")
    print(best_config)
    # Fit classifier with the best config on all Data
    _, kwargs = best_config
    scaler = StandardScaler().fit(data[x_cols])

    clf = SVC(**kwargs)
    clf.fit(X=scaler.transform(data[x_cols]), y=data[y_col])
    return scaler, clf


if __name__ == "__main__":
    df = pd.read_csv('dataset/Dataset7_XY.csv')
    X_INDEX = df.columns.drop(Y_INDEX)
    train_set, hold_out_set = train_test_split(data=df)

    validation_set = create_folds(data=train_set)
    start_time = time.time()
    scaler, model_chosen = CV(data=train_set.reset_index(drop=True), validation_indices=validation_set)
    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    # AUC score and ROC for best classifier:
    y_pred_proba = model_chosen.predict_proba(scaler.transform(hold_out_set[X_INDEX]))[::, 1]
    hold_out_auc = roc_auc_score(hold_out_set[Y_INDEX], y_pred_proba)
    print('Hold-Out set AUC:')
    print(hold_out_auc)

    # Plot ROC Curve
    plt.figure()
    label = "AUC={:0.3f}".format(hold_out_auc)
    fpr, tpr, _ = roc_curve(hold_out_set[Y_INDEX], y_pred_proba)
    plt.plot(fpr, tpr, label=label)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve for best config SVM')
    plt.legend()
    plt.show()
