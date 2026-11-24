"""
Experiment 9: Compare Linear and Polynomial Regression
Compare Linear and Polynomial Regression using Python on non-linear synthetic data.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    np.random.seed(42)
    X = np.sort(6 * np.random.rand(80, 1) - 3, axis=0)
    y = 0.8 * (X ** 2).ravel() + 1.5 * X.ravel() + 2.0 + np.random.randn(80) * 1.2

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_pred_poly = poly_reg.predict(X_poly)

    print("=== Linear Regression Performance ===")
    print(f"MSE: {mean_squared_error(y, y_pred_lin):.4f}")
    print(f"R2 Score: {r2_score(y, y_pred_lin):.4f}")

    print("\n=== Polynomial Regression (Degree 2) Performance ===")
    print(f"MSE: {mean_squared_error(y, y_pred_poly):.4f}")
    print(f"R2 Score: {r2_score(y, y_pred_poly):.4f}")
