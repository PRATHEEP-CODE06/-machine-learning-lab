"""
Experiment 19: Naive Bayes Classification for Bank Loan Prediction
Implementation of Naive Bayes classification for Bank Loan prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generate_loan_dataset(n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    applicant_income = rng.integers(2000, 15000, n_samples)
    coapplicant_income = rng.integers(0, 6000, n_samples)
    loan_amount = rng.integers(50, 500, n_samples)
    loan_term = rng.choice([180, 240, 360], n_samples, p=[0.15, 0.15, 0.70])
    credit_history = rng.choice([0, 1], n_samples, p=[0.2, 0.8])

    total_income = applicant_income + coapplicant_income
    approved = ((credit_history == 1) & ((total_income / loan_amount) > 22)).astype(int)

    return pd.DataFrame({
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Loan_Status": approved,
    })


if __name__ == "__main__":
    df = generate_loan_dataset()
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Bank Loan Prediction (Naive Bayes) Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
