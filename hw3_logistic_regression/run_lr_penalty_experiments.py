import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_X_FP = "Assignment3_DataCode/Dataset3.3_X.csv"
DATA_Y_FP = "Assignment3_DataCode/Dataset3.3_Y.csv"
INCLUDE_BIAS_TERM = False

X = pd.read_csv(DATA_X_FP, header=None)
y = pd.read_csv(DATA_Y_FP, header=None)


penalty_values = [0.5, 10, 100]
plt.figure()

log_clf = LogisticRegression(penalty="none")
log_clf.fit(X, y)

weights = (
    np.hstack((log_clf.intercept_[:, None], log_clf.coef_))
    if INCLUDE_BIAS_TERM
    else log_clf.coef_
)
x_weights = [i for i in range(log_clf.coef_.shape[1])]
plt.scatter(x_weights, weights.ravel())
plt.plot(x_weights, weights.ravel(), label="0")
for C in penalty_values:
    log_clf = LogisticRegression(penalty="l1", solver="liblinear", C=1 / C)
    log_clf.fit(X, y)

    weights = (
        np.hstack((log_clf.intercept_[:, None], log_clf.coef_))
        if INCLUDE_BIAS_TERM
        else log_clf.coef_
    )
    plt.scatter(x_weights, weights.ravel())
    plt.plot(x_weights, weights.ravel(), label=C)

plt.ylabel("Value")
plt.xlabel("W_i")
plt.legend()
plt.show()
