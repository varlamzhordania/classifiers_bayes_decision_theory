import numpy as np
from numpy.linalg import norm

def predict_euclidean(X, means):
    predictions = []
    for x in X:
        dists = [norm(x - m) for m in means]
        predictions.append(np.argmin(dists))
    return np.array(predictions)
