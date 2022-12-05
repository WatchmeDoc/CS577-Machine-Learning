import pandas as pd

CORR_THRESHOLD = 0.29
CATEGORICAL_VALUES_THRESHOLD = 14
IMBALANCE_STD_THRESHOLD = 0.5
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


if __name__ == "__main__":
    df = pd.read_csv('data/Dataset6.A_XY.csv', header=None)
    y = df[df.columns[-1]]
    X = df[df.columns[:-1]]
    categorical_vars, continuous_vars = find_categorical_vars(X=X)
    # Select high corr variables
    high_corr_vars = df.corr()[df.columns[-1]][:-1].sort_values().loc[lambda x: abs(x) >= CORR_THRESHOLD]
    # Basic statistics
    mean = X[continuous_vars.keys()].mean()
    std = X[continuous_vars.keys()].std()
    print('\nMean:\n', mean)
    print('\n------------------')
    print('\nstd:\n', std)
    # Coefficient of Variation
    CV = std / mean
    print('\n------------------')
    print('\nCV:\n', CV)
    high_cv_vars = CV.loc[lambda x: x >= CV_THRESHOLD]

    # Percentage of values for categorical features
    N = len(df)
    imbalances = []
    for col in categorical_vars.keys():
        imbalances.append(X[col].value_counts(normalize=True).std())
    imbalances = X[categorical_vars.keys()].apply(lambda x: x.value_counts(normalize=True).std())
    print(f"\n", imbalances.sort_values())
    high_imbalance_vars = imbalances.loc[lambda x: x >= IMBALANCE_STD_THRESHOLD]
    chosen_vars = high_imbalance_vars.index.union(high_cv_vars.index).union(high_corr_vars.index)
    print(chosen_vars)
