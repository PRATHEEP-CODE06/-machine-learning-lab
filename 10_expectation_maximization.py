"""
Experiment 10: Expectation & Maximization (EM) Algorithm
Write a Python Program to Implement Expectation & Maximization Algorithm
using Gaussian Mixture Models (GMM).
"""

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score


if __name__ == "__main__":
    iris = load_iris()
    X, y_true = iris.data, iris.target

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(X)
    cluster_labels = gmm.predict(X)

    print("Converged:", gmm.converged_)
    print("Number of EM Iterations:", gmm.n_iter_)
    print("Log-Likelihood:", round(gmm.score(X), 4))
    print("Silhouette Score:", round(silhouette_score(X, cluster_labels), 4))
    print("Adjusted Rand Index (vs True Labels):", round(adjusted_rand_score(y_true, cluster_labels), 4))
    print("\nCluster Means:\n", gmm.means_)
