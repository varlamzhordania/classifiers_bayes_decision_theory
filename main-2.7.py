from data.data_generator import generate_dataset_3
from classifiers.bayes import bayes_with_priors
from classifiers.euclidean import predict_euclidean
from evaluation.metrics import classification_error
from visualization.plot import plot_data
import numpy as np

def run_experiment(name, priors):
    print(f"\n===== Running {name} =====")

    X, y_true, means, cov = generate_dataset_3(1000, priors)

    plot_data(X, y_true)

    bayes_pred = bayes_with_priors(X, means, cov, priors)
    bayes_err = classification_error(bayes_pred, y_true)

    euclid_pred = predict_euclidean(X, means)
    euclid_err = classification_error(euclid_pred, y_true)

    print(f"Bayesian error:  {bayes_err}")
    print(f"Euclidean error: {euclid_err}")

    return bayes_err, euclid_err

def main():
    priors_equ = np.array([1/3, 1/3, 1/3])
    run_experiment("(equiprobable)", priors_equ)
    priors_non = np.array([0.8, 0.1, 0.1])
    run_experiment("(non-equiprobable)", priors_non)

if __name__ == "__main__":
    main()
