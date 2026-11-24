"""
Experiment 17: Mobile Price Range Prediction
Implement Mobile Price Prediction using an appropriate machine learning algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def generate_mobile_dataset(n_samples=600, seed=42):
    rng = np.random.default_rng(seed)
    ram = rng.integers(512, 8192, n_samples)
    battery_power = rng.integers(1000, 5000, n_samples)
    px_height = rng.integers(480, 1920, n_samples)
    px_width = rng.integers(720, 2560, n_samples)
    int_memory = rng.integers(8, 256, n_samples)
    five_g = rng.integers(0, 2, n_samples)

    spec_score = (ram / 2000) + (battery_power / 2500) + ((px_height * px_width) / 2_000_000) + (int_memory / 128) + five_g
    price_range = pd.qcut(spec_score, q=4, labels=[0, 1, 2, 3]).astype(int)

    return pd.DataFrame({
        "RAM": ram,
        "Battery_Power": battery_power,
        "Px_Height": px_height,
        "Px_Width": px_width,
        "Internal_Memory": int_memory,
        "Five_G": five_g,
        "Price_Range": price_range,
    })


if __name__ == "__main__":
    df = generate_mobile_dataset()
    X = df.drop("Price_Range", axis=1)
    y = df["Price_Range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_clf = SVC(kernel="rbf", C=2.0, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    print(f"Mobile Price Range Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
