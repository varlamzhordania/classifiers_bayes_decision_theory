import numpy as np
from numpy.linalg import inv


def gaussian_log_likelihood(x, mean, cov):
    cov_inv = inv(cov)
    return -0.5 * (x - mean).T @ cov_inv @ (x - mean)


def predict_bayes(X, means, cov):
    predictions = []
    for x in X:
        scores = [gaussian_log_likelihood(x, m, cov) for m in means]
        predictions.append(np.argmax(scores))
    return np.array(predictions)


def bayes_with_priors(X, means, cov, priors):
    cov_inv = inv(cov)
    predictions = []

    for x in X:
        scores = []
        for i, m in enumerate(means):
            log_likelihood = -0.5 * (x - m).T @ cov_inv @ (x - m)
            log_prior = np.log(priors[i])
            scores.append(log_likelihood + log_prior)
        predictions.append(np.argmax(scores))

    return np.array(predictions)
