"""
Experiment 13: Car Price Prediction Model
Implement a Car Price Prediction Model using Python.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def get_car_dataset(n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    years = rng.integers(2010, 2025, n_samples)
    present_price = rng.uniform(4.0, 35.0, n_samples)
    kms_driven = rng.integers(5000, 150000, n_samples)
    fuel_type = rng.choice(["Petrol", "Diesel", "CNG"], n_samples, p=[0.55, 0.40, 0.05])
    transmission = rng.choice(["Manual", "Automatic"], n_samples, p=[0.7, 0.3])

    age = 2026 - years
    fuel_bonus = np.where(fuel_type == "Diesel", 1.5, 0.0)
    trans_bonus = np.where(transmission == "Automatic", 2.0, 0.0)
    selling_price = np.maximum(
        1.0,
        present_price * (0.91 ** age) - (kms_driven / 100000) * 1.2 + fuel_bonus + trans_bonus + rng.normal(0, 0.4, n_samples)
    )

    return pd.DataFrame({
        "Year": years,
        "Present_Price": np.round(present_price, 2),
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Selling_Price": np.round(selling_price, 2),
    })


if __name__ == "__main__":
    df = get_car_dataset()
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), ["Fuel_Type", "Transmission"])],
        remainder="passthrough",
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Car Price Prediction MAE: {mean_absolute_error(y_test, y_pred):.4f} Lakhs")
    print(f"Car Price Prediction R2 Score: {r2_score(y_test, y_pred):.4f}")
