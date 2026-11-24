"""
Experiment 14: House Price Prediction
Implement House Price Prediction using an appropriate machine learning algorithm.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


if __name__ == "__main__":
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("House Price Prediction (California Housing Dataset)")
    print(f"MAE:  {mae:.4f} ($100k units)")
    print(f"RMSE: {rmse:.4f} ($100k units)")
    print(f"R2 Score: {r2:.4f}")
