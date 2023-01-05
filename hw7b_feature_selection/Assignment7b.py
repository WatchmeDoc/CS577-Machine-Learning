# Run: pip install causal-learn

from sklearn.datasets import load_breast_cancer
from causallearn.utils.cit import CIT
import numpy as np


# SEE THE PARAMETERS' INFO FROM THE PDF
def stat_test(D, V_idx, T_idx, S=None):
    kci_obj = CIT(D, "kci")  # construct a CIT instance with data and method name
    pValue = kci_obj(V_idx, T_idx, S)
    return pValue


def forward_selection(D, V_indices, T_idx, S, a):
    pass


def backward_selection(D, T_idx, S, a):
    pass


if __name__=='__main__':
    D = load_breast_cancer(as_frame=True).frame.values
    V_indices = np.arange(D.shape[1] - 1)
    T_idx = D.shape[1] - 1

    # START WRITING YOUR CODE (YOU CAN CREATE AS MANY FUNCTIONS AS YOU WANT)