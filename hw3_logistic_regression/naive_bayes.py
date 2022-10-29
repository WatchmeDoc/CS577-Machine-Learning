import pandas as pd
import numpy as np

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
CLASS_PROBABILITIES_KEY = "class_probabilities"
LIKELIHOODS_KEY = "likelihoods"
MEAN_KEY = "mean"
STD_KEY = "std"


def train_NBC(X: pd.DataFrame, X_dtype, Y: pd.DataFrame, L, D_categorical):
    """
    Trains and returns a Naive Bayes classifier based on the Data provided.
    Model structure, a dictionary that contains "class_probabilities" and "likelihoods" keys.
    class_probabilities is a dictionary with class labels as keys and their corresponding probability.
    likelihoods is a dictionary with the following structure: likelihoods[x_variable_index][y_label][x_value]
    :param X: A IxM matrix of categorical variables. Rows correspond to samples
              and columns to variables
    :param X_dtype: A string describing the data type of X, which could be
                    either "categorical" or "continuous"
    :param Y: A Ix1 vector. Y is the class variable you want to predict.
    :param L: A positive scalar. L is the parameter referred to in the MAP estimates
              equation (for L = 0 you get MLE estimates)
    :param D_categorical: A 1xM vector. Each element D(m) contains the number of possible
                          different values that the categorical variable m can have. This
                          vector is ignored if X_dtype = "continuous"
    :return: a trained naive bayes classifier
    """
    sample_count = len(X)
    categories = np.unique(Y)
    class_probability = {}
    likelihoods = {}

    for label in categories:
        # 1. Calculate class prior probabilities
        label_count = len(Y[Y[0] == label])
        class_probability[label] = (label_count + L) / (sample_count + L * len(categories))
        class_x = X[Y[0] == label]

        # 2. Calculate frequency for each Variable
        # Dictionary structure: likelihoods[x_variable_index][y_label][x_value]
        # e.g. the frequency of 0 in X_2 given Y = 1 is likelihoods[2][1][0]
        if X_dtype is CATEGORICAL:
            for i, D_m in enumerate(D_categorical.iloc[0]):
                if likelihoods.get(i) is None:
                    likelihoods[i] = {}
                if likelihoods[i].get(label) is None:
                    likelihoods[i][label] = {}

                for value in range(D_m):
                    sample_freq = len(class_x[class_x[i] == value])
                    likelihoods[i][label][value] = (sample_freq + L) / (len(class_x[i]) + L * D_m)
        elif X_dtype is CONTINUOUS:
            for i in X.columns:
                if likelihoods.get(i) is None:
                    likelihoods[i] = {}
                if likelihoods[i].get(label) is None:
                    likelihoods[i][label] = {}
                X_i: pd.Series = class_x[i]
                mean = X_i.mean()
                std = X_i.std()
                likelihoods[i][label][MEAN_KEY] = mean
                likelihoods[i][label][STD_KEY] = std
        else:
            raise ValueError(f"Unknown Data type provided. Please choose either \"{CATEGORICAL}\" or \"{CONTINUOUS}\"")

    model = {CLASS_PROBABILITIES_KEY: class_probability,
             LIKELIHOODS_KEY: likelihoods}

    return model


def predict_NBC(model, X: pd.DataFrame, X_dtype):
    """
    Predicts on data X using the trained model provided on the function
    :param model: A model previously trained using train NBC
    :param X: A JxM matrix of variables. Rows correspond to samples and columns to
              variables.
    :param X_dtype: A string describing the data type of X, which could be either
                    ”categorical” or ”continuous”
    :return predictions: A Jx1 vector. It contains the predicted class for each of the
                         input samples.
    """
    predictions = []

    # For each sample
    for i in range(len(X)):
        sample = X.iloc[[i]]
        probability_per_class = {}
        # For each class label
        for label in model[CLASS_PROBABILITIES_KEY].keys():
            prob_prod = model[CLASS_PROBABILITIES_KEY][label]
            # For each Variable X_i calculate P(Y = yk) * prod(P(X_i | Y = yk))
            for X_i in sample.columns:
                x_value = sample[X_i].iloc[0]
                if X_dtype is CATEGORICAL:
                    prob_prod *= model[LIKELIHOODS_KEY][X_i][label][x_value]
                elif X_dtype is CONTINUOUS:
                    mu = model[LIKELIHOODS_KEY][X_i][label][MEAN_KEY]
                    sigma = model[LIKELIHOODS_KEY][X_i][label][STD_KEY]
                    prob_prod *= _gauss_pdf(x=x_value, mu=mu, sigma=sigma)
                else:
                    raise ValueError(
                        f"Unknown Data type provided. Please choose either \"{CATEGORICAL}\" or \"{CONTINUOUS}\"")
            probability_per_class[label] = prob_prod
        # argmax w.r.t. y_k from the calculated products
        predictions.append(max(probability_per_class, key=probability_per_class.get))
    return predictions


def _gauss_pdf(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp((-(x - mu) ** 2) / (2 * (sigma ** 2)))


def print_NBC_model(model):
    print(CLASS_PROBABILITIES_KEY, ':', '{')
    for label in model[CLASS_PROBABILITIES_KEY].keys():
        print('\t', label, ':', model[CLASS_PROBABILITIES_KEY][label])
    print('}')
    print('------------------------------------------------------')
    print(LIKELIHOODS_KEY, ':', '{')
    for X_i in model[LIKELIHOODS_KEY].keys():
        print(f"\t X_{X_i} : {'{'}")
        for label in model[LIKELIHOODS_KEY][X_i].keys():
            print(f"\t\t Y = {label} : {'{'}")
            for x_value in model[LIKELIHOODS_KEY][X_i][label]:
                print(f"\t\t\t P(X_{X_i} = {x_value} | Y = {label}) : {model[LIKELIHOODS_KEY][X_i][label][x_value]}")
            print('\t\t}')
        print('\t}')
    print('}')
