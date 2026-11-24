"""
Experiment 11: Credit Score Classification
Write a program for the task of Credit Score Classification (Poor, Standard, Good).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def generate_credit_data(n_samples=600, seed=42):
    rng = np.random.default_rng(seed)
    annual_income = rng.uniform(20000, 150000, n_samples)
    outstanding_debt = rng.uniform(500, 50000, n_samples)
    credit_utilization = rng.uniform(5, 95, n_samples)
    num_delayed_payments = rng.integers(0, 15, n_samples)
    credit_history_months = rng.integers(6, 240, n_samples)

    score_index = (
        (annual_income / 15000)
        - (outstanding_debt / 5000)
        - (credit_utilization / 20)
        - (num_delayed_payments * 0.6)
        + (credit_history_months / 40)
    )

    credit_score = np.where(
        score_index > 4.0, "Good",
        np.where(score_index > -1.5, "Standard", "Poor")
    )

    return pd.DataFrame({
        "Annual_Income": annual_income,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization": credit_utilization,
        "Delayed_Payments": num_delayed_payments,
        "Credit_History_Months": credit_history_months,
        "Credit_Score": credit_score,
    })


if __name__ == "__main__":
    df = generate_credit_data()
    X = df.drop("Credit_Score", axis=1)
    y = df["Credit_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Credit Score Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
