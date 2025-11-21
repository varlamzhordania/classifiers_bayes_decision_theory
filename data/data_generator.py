import numpy as np

def generate_dataset(N=1000):
    N_per_class = N // 3

    m1 = np.array([1, 1])
    m2 = np.array([12, 8])
    m3 = np.array([16, 1])

    S = 4 * np.eye(2)

    X1 = np.random.multivariate_normal(m1, S, N_per_class)
    X2 = np.random.multivariate_normal(m2, S, N_per_class)
    X3 = np.random.multivariate_normal(m3, S, N_per_class)

    X = np.vstack([X1, X2, X3])
    y = np.array([0]*N_per_class + [1]*N_per_class + [2]*N_per_class)

    means = [m1, m2, m3]

    return X, y, means, S

def generate_dataset_2(N=1000):
    N_per_class = N // 3

    m1 = np.array([1, 1])
    m2 = np.array([14, 7])
    m3 = np.array([16, 1])

    S = np.array([[5, 3],
                  [3, 4]])

    X1 = np.random.multivariate_normal(m1, S, N_per_class)
    X2 = np.random.multivariate_normal(m2, S, N_per_class)
    X3 = np.random.multivariate_normal(m3, S, N_per_class)

    X = np.vstack([X1, X2, X3])
    y = np.array([0]*N_per_class + [1]*N_per_class + [2]*N_per_class)

    means = [m1, m2, m3]

    return X, y, means, S

def generate_dataset_3(N=1000, priors=None):
    m1 = np.array([1, 1])
    m2 = np.array([4, 4])
    m3 = np.array([8, 1])
    means = [m1, m2, m3]

    S = 2 * np.eye(2)

    if priors is None:
        priors = np.array([1/3, 1/3, 1/3])

    assert np.isclose(priors.sum(), 1), "Priors must sum to 1."

    N1 = int(N * priors[0])
    N2 = int(N * priors[1])
    N3 = N - N1 - N2

    X1 = np.random.multivariate_normal(m1, S, N1)
    X2 = np.random.multivariate_normal(m2, S, N2)
    X3 = np.random.multivariate_normal(m3, S, N3)

    X = np.vstack([X1, X2, X3])
    y = np.array([0]*N1 + [1]*N2 + [2]*N3)

    return X, y, means, S