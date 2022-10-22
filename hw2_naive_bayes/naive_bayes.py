import pandas as pd
import numpy as np
from tqdm import tqdm

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
CLASS_PROBABILITIES = "class_probabilities"
LIKELIHOODS = "likelihoods"
TRAIN_SET_PERCENTAGE = 0.75
PRINT_MODEL = False
REPEAT = 100
SMOOTH_STR = 0


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
    model = None

    if X_dtype is CATEGORICAL:
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
            for i, D_m in enumerate(D_categorical.iloc[0]):
                if likelihoods.get(i) is None:
                    likelihoods[i] = {}
                if likelihoods[i].get(label) is None:
                    likelihoods[i][label] = {}
                for value in range(D_m):
                    sample_freq = len(class_x[class_x[i] == value])
                    # print(f"Given label = {label}, for variable X_{i} we have {sample_freq} instances of value {value}.")
                    likelihoods[i][label][value] = (sample_freq + L) / (len(class_x[i]) + L * D_m)

        model = {CLASS_PROBABILITIES: class_probability,
                 LIKELIHOODS: likelihoods}
    elif X_dtype is CONTINUOUS:
        pass
    else:
        raise ValueError("Unknown Data type provided. Please choose either \"categorical\" or \"continuous\"")
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
    if X_dtype is CATEGORICAL:
        # For each sample
        for i in range(len(X)):
            sample = X.iloc[[i]]
            probability_per_class = {}
            # For each class label
            for label in model[CLASS_PROBABILITIES].keys():
                prob_prod = model[CLASS_PROBABILITIES][label]
                # For each Variable X_i calculate P(Y = yk) * prod(P(X_i | Y = yk))
                for X_i in sample.columns:
                    x_value = sample[X_i].iloc[0]
                    prob_prod *= model[LIKELIHOODS][X_i][label][x_value]
                probability_per_class[label] = prob_prod
            # argmax w.r.t. y_k from the calculated products
            predictions.append(max(probability_per_class, key=probability_per_class.get))
    elif X_dtype is CONTINUOUS:
        pass
    else:
        raise ValueError("Unknown Data type provided. Please choose either \"categorical\" or \"continuous\"")
    return predictions


def compute_accuracy(predictions, actual_y):
    total_predictions = len(predictions)
    if total_predictions != len(actual_y):
        raise ValueError("Dimensions between predictions and actual_y must match!")
    score = 0
    for i in range(total_predictions):
        score += 1 if predictions[i] == actual_y[i] else 0
    return score / total_predictions


def print_model(model):
    print(CLASS_PROBABILITIES, ':', '{')
    for label in model[CLASS_PROBABILITIES].keys():
        print('\t', label, ':', model[CLASS_PROBABILITIES][label])
    print('}')
    print('------------------------------------------------------')
    print(LIKELIHOODS, ':', '{')
    for X_i in model[LIKELIHOODS].keys():
        print(f"\t X_{X_i} : {'{'}")
        for label in model[LIKELIHOODS][X_i].keys():
            print(f"\t\t Y = {label} : {'{'}")
            for x_value in model[LIKELIHOODS][X_i][label]:
                print(f"\t\t\t P(X_{X_i} = {x_value} | Y = {label}) : {model[LIKELIHOODS][X_i][label][x_value]}")
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
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X = pd.read_csv('Assignment2_Data/DatasetA_X_categorical.csv', header=None)
    Y = pd.read_csv('Assignment2_Data/DatasetA_Y.csv', header=None)
    D_categorical = pd.read_csv('Assignment2_Data/DatasetA_D_categorical.csv', header=None)
    accuracies = []
    for i in tqdm(range(REPEAT)):
        X_train, y_train, X_test, y_test = train_test_split(X=X, Y=Y)
        model = train_NBC(X=X_train, X_dtype=CATEGORICAL, Y=y_train, L=SMOOTH_STR, D_categorical=D_categorical)
        if PRINT_MODEL:
            print_model(model=model)
        predictions = predict_NBC(model=model, X=X_test, X_dtype=CATEGORICAL)
        accuracy = compute_accuracy(predictions=predictions, actual_y=y_test.values.tolist())
        accuracies.append(accuracy)

    print('Average Naive Bayes Accuracy = ', np.mean(accuracies))
