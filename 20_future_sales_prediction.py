"""
Experiment 20: Future Sales Prediction
Implement Future Sales Prediction using a suitable machine learning algorithm.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def generate_advertising_sales_data(n_samples=300, seed=42):
    rng = np.random.default_rng(seed)
    tv_spend = rng.uniform(10, 300, n_samples)
    radio_spend = rng.uniform(0, 50, n_samples)
    newspaper_spend = rng.uniform(0, 100, n_samples)

    sales = 3.0 + 0.046 * tv_spend + 0.188 * radio_spend + 0.005 * newspaper_spend + rng.normal(0, 1.2, n_samples)

    return pd.DataFrame({
        "TV": np.round(tv_spend, 2),
        "Radio": np.round(radio_spend, 2),
        "Newspaper": np.round(newspaper_spend, 2),
        "Sales": np.round(sales, 2),
    })


if __name__ == "__main__":
    df = generate_advertising_sales_data()
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    future_budget = pd.DataFrame({"TV": [230.1], "Radio": [37.8], "Newspaper": [69.2]})
    predicted_sales = model.predict(future_budget)[0]
    print(f"\nFuture Ad Budget: TV=$230.1k, Radio=$37.8k, Newspaper=$69.2k")
    print(f"Predicted Future Sales: {predicted_sales:.2f} units")
