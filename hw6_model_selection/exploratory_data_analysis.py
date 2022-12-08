import matplotlib.pyplot as plt
import pandas as pd

CORR_THRESHOLD = 0.29
CATEGORICAL_VALUES_THRESHOLD = 14
IMBALANCE_CV_THRESHOLD = 1.5
CV_THRESHOLD = 1.0


def find_categorical_vars(X: pd.DataFrame):
    categorical_vars = {}
    continuous_vars = {}
    for col in X.columns:
        if len(X[col].value_counts()) < CATEGORICAL_VALUES_THRESHOLD:
            categorical_vars[col] = len(X[col].value_counts())
        else:
            continuous_vars[col] = len(X[col].value_counts())
    return categorical_vars, continuous_vars


def plot_hist_features(X: pd.DataFrame, title= None):
    for col in X.columns:
        plt.figure()
        plt.hist(X[col])
        plt.xlabel(f"feature: {col}")
        plt.ylabel("count")
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('data/Dataset6.A_XY.csv', header=None)
    y = df[df.columns[-1]]
    X = df[df.columns[:-1]]
    categorical_vars, continuous_vars = find_categorical_vars(X=X)
    # Select high corr variables
    high_corr_vars = df.corr()[df.columns[-1]][:-1].sort_values().loc[lambda x: abs(x) >= CORR_THRESHOLD]
    print("Pearson Correlation Variables to Target Variable:")
    print(high_corr_vars)
    # Basic statistics
    mean = X[continuous_vars.keys()].mean()
    std = X[continuous_vars.keys()].std()
    # Coefficient of Variation
    CV = std / mean
    high_cv_vars = CV.loc[lambda x: x >= CV_THRESHOLD]
    print("\nContinuous Variables with High Coefficient of Variance:")
    print(high_cv_vars)

    # Percentage of values for categorical features
    # for col in categorical_vars.keys():
    #     print('\n------------------')
    #     print(f'{col}:\n', X[col].value_counts(normalize=True))
    imbalances = X[categorical_vars.keys()].apply(lambda x: x.value_counts(normalize=True).std() / x.value_counts(normalize=True).mean())
    high_imbalance_vars = imbalances.loc[lambda x: x >= IMBALANCE_CV_THRESHOLD]
    print("\nCategorical Variables with High Category Imbalance:")
    print(high_imbalance_vars)
    plot_hist_features(X=X[high_imbalance_vars.index], title="High Imbalance Variable")
    plot_hist_features(X=X[high_cv_vars.index], title="High coefficient of variation variable")
    plot_hist_features(X=X[high_corr_vars.index], title="High correlation to target variable")
