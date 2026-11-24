"""
Experiment 12: Iris Flower Classification using KNN
Implement Iris Flower Classification using K-Nearest Neighbors (KNN).
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Iris KNN Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    sample = [[5.1, 3.5, 1.4, 0.2]]
    pred_class = iris.target_names[knn.predict(sample)[0]]
    print(f"Sample {sample[0]} Predicted Species: {pred_class}")
