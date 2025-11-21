import numpy as np
from numpy.linalg import inv

def predict_mahalanobis(X, means, cov):
    cov_inv = inv(cov)
    predictions = []
    for x in X:
        scores = [(x - m).T @ cov_inv @ (x - m) for m in means]
        predictions.append(np.argmin(scores))
    return np.array(predictions)
