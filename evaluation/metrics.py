import numpy as np

def classification_error(pred, true):
    return np.mean(pred != true)
