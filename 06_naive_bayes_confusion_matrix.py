"""
Experiment 6: Naive Bayes Algorithm with Confusion Matrix and Accuracy
Write a program to implement Naive Bayes algorithm in Python and display the results
using confusion matrix and accuracy.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


if __name__ == "__main__":
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=dataset.target_names))
