import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CLASSIFIERS_TO_TRAIN = [{LogisticRegression: [{'C': 0}, {'C': 1}, {'C': 3}, {'C': 5}]},
                        {DecisionTreeClassifier: [{'min_samples_leaf': 1, 'max_features': None},
                                                  {'min_samples_leaf': 4, 'max_features': None},
                                                  {'min_samples_leaf': 10, 'max_features': None},
                                                  {'min_samples_leaf': 1, 'max_features': "sqrt"},
                                                  {'min_samples_leaf': 4, 'max_features': "sqrt"},
                                                  {'min_samples_leaf': 10, 'max_features': "sqrt"}
                                                  ]
                         }]

if __name__ == "__main__":
    df = pd.read_csv('data/Dataset6.A_XY.csv', header=None)
    y = df[df.columns[-1]]
    X = df[df.columns[:-1]]
