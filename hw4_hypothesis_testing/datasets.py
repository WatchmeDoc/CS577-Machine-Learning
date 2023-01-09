import pandas as pd

observations1 = pd.DataFrame([[0.05, 0.10], [0.25, 0.60]], columns=[0, 1], index=[0, 1])
observations2 = pd.DataFrame([[0.08, 0.23], [0.17, 0.52]], columns=[0, 1], index=[0, 1])
observations3 = pd.DataFrame([[0.10, 0.20], [0.20, 0.50]], columns=[0, 1], index=[0, 1])
observations4 = pd.DataFrame([[0.10, 0.10], [0.10, 0.70]], columns=[0, 1], index=[0, 1])
observations5 = pd.DataFrame([[0.15, 0.10], [0.10, 0.65]], columns=[0, 1], index=[0, 1])
observations6 = pd.DataFrame([[0.45, 0.05], [0.05, 0.45]], columns=[0, 1], index=[0, 1])

all_observations = [
    observations1,
    observations2,
    observations3,
    observations4,
    observations5,
    observations6,
]
