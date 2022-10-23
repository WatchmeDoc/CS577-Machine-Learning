import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
CLASS_PROBABILITIES_KEY = "class_probabilities"
LIKELIHOODS_KEY = "likelihoods"
MEAN_KEY = "mean"
STD_KEY = "std"
TRAIN_SET_PERCENTAGE = 0.75
PRINT_MODEL = True
REPEAT = 100
SMOOTH_STR = 0
PLOT_SMOOTH_STR_EXP = False

# Data filepaths
DISCRETE_X_FP = 'Assignment2_Data/DatasetA_X_categorical.csv'
DISCRETE_Y_FP = 'Assignment2_Data/DatasetA_Y.csv'
D_CATEGORICAL_FP = 'Assignment2_Data/DatasetA_D_categorical.csv'
CONTINUOUS_X_FP = 'Assignment2_Data/DatasetB_X_continuous.csv'
CONTINUOUS_Y_FP = 'Assignment2_Data/DatasetB_Y.csv'


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
                    prob_prod *= gauss_pdf(x=x_value, mu=mu, sigma=sigma)
                else:
                    raise ValueError(
                        f"Unknown Data type provided. Please choose either \"{CATEGORICAL}\" or \"{CONTINUOUS}\"")
            probability_per_class[label] = prob_prod
        # argmax w.r.t. y_k from the calculated products
        predictions.append(max(probability_per_class, key=probability_per_class.get))
    return predictions


def gauss_pdf(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp((-(x - mu) ** 2) / (2 * (sigma ** 2)))


def compute_accuracy(predictions, actual_y):
    total_predictions = len(predictions)
    if total_predictions != len(actual_y):
        raise ValueError("Dimensions between predictions and actual_y must match!")
    score = 0
    for i in range(total_predictions):
        score += 1 if predictions[i] == actual_y[i] else 0
    return score / total_predictions


def print_model(model):
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


def train_test_split(X, Y):
    if len(X) != len(Y):
        raise ValueError("Dimensions between X and Y must match!")
    msk = np.random.rand(len(X)) < TRAIN_SET_PERCENTAGE
    X_train = X[msk]
    X_test = X[~msk]

    Y_train = Y[msk]
    Y_test = Y[~msk]
    if TRAIN_SET_PERCENTAGE == 1.0:
        X_test = X_train
        Y_test = Y_train
    return X_train, Y_train, X_test, Y_test


def main(X_dtype):
    if X_dtype is CATEGORICAL:
        X = pd.read_csv(DISCRETE_X_FP, header=None)
        Y = pd.read_csv(DISCRETE_Y_FP, header=None)
        D_categorical = pd.read_csv(D_CATEGORICAL_FP, header=None)
    elif X_dtype is CONTINUOUS:
        X = pd.read_csv(CONTINUOUS_X_FP, header=None)
        Y = pd.read_csv(CONTINUOUS_Y_FP, header=None)
        D_categorical = None
    else:
        raise ValueError(f"Unknown Data type provided. Please choose either \"{CATEGORICAL}\" or \"{CONTINUOUS}\"")
    accuracies = []
    print(
        f"Experiment running for {X_dtype} type variables, repeating {REPEAT} times with smoothing strength = {SMOOTH_STR}"
    )
    for i in tqdm(range(REPEAT)):
        X_train, y_train, X_test, y_test = train_test_split(X=X, Y=Y)
        model = train_NBC(X=X_train, X_dtype=X_dtype, Y=y_train, L=SMOOTH_STR, D_categorical=D_categorical)
        if PRINT_MODEL and i == 0:
            print_model(model=model)
        predictions = predict_NBC(model=model, X=X_test, X_dtype=X_dtype)
        accuracy = compute_accuracy(predictions=predictions, actual_y=y_test.values.tolist())
        accuracies.append(accuracy)
    mean_acc = np.mean(accuracies)
    print('Average Naive Bayes Accuracy = ', mean_acc)
    return mean_acc


if __name__ == "__main__":
    main(X_dtype=CATEGORICAL)
    main(X_dtype=CONTINUOUS)
    if PLOT_SMOOTH_STR_EXP:
        experiments = [0, 1, 2, 5, 8, 10, 20, 50, 85, 95, 100, 300, 512, 666, 732, 856, 947, 1000]
        results = []
        tried_experiments = []
        for SMOOTH_STR in experiments:
            mean_acc = main(X_dtype=CATEGORICAL)
            results.append(mean_acc)
            tried_experiments.append(SMOOTH_STR)
            if SMOOTH_STR == 0:
                plt.figure()
            elif SMOOTH_STR == 10 or SMOOTH_STR == 100 or SMOOTH_STR == 1000:
                plt.scatter(tried_experiments, results)
                plt.plot(tried_experiments, results)
                plt.xlabel('L')
                plt.ylabel('Mean Classification Accuracy')
                plt.show()
                plt.figure()
                results = [mean_acc]
                tried_experiments = [SMOOTH_STR]
