from data.data_generator import generate_dataset_2
from classifiers.bayes import predict_bayes
from classifiers.euclidean import predict_euclidean
from classifiers.mahalanobis import predict_mahalanobis
from evaluation.metrics import classification_error
from visualization.plot import plot_data

def main():
    X, y_true, means, cov = generate_dataset_2()

    plot_data(X, y_true)

    bayes_pred = predict_bayes(X, means, cov)
    euclid_pred = predict_euclidean(X, means)
    mahal_pred = predict_mahalanobis(X, means, cov)

    print("Bayesian error:     ", classification_error(bayes_pred, y_true))
    print("Euclidean error:    ", classification_error(euclid_pred, y_true))
    print("Mahalanobis error:  ", classification_error(mahal_pred, y_true))

if __name__ == "__main__":
    main()
