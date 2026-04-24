"""
Program Name: Comparator of SLR and MLR on a same dataset
Description : This program compares simple Linear Regression and Multiple Linear Regression on a same dataset
Course      : Machine Learning Laboratory (CS4201)
Date        : 05-03-2026
Language    : Python
"""
# -------------------------------
# Import all modules here
# -------------------------------
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# -------------------------------
# Linear Regression (Least Squares)
# -------------------------------
class LinearRegressionLS:

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        self.weights = np.linalg.pinv(X_b) @ y

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        return X_b @ self.weights


# -------------------------------
# Evaluation Metrics
# -------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)


# -------------------------------
# Load Dataset (Boston)
# -------------------------------
data = fetch_openml(name="boston", version=1, as_frame=False)
X = np.array(data.data, dtype=np.float64)
y = np.array(data.target, dtype=np.float64)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Simple Linear Regression (1 feature)
# -------------------------------
X_train_slr = X_train[:, [0]]   # only one feature
X_test_slr = X_test[:, [0]]

slr = LinearRegressionLS()
slr.fit(X_train_slr, y_train)
y_pred_slr = slr.predict(X_test_slr)

# -------------------------------
# Multiple Linear Regression (all features)
# -------------------------------
mlr = LinearRegressionLS()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

# -------------------------------
# Results
# -------------------------------
print("----- Simple Linear Regression -----")
print("MSE:", mse(y_test, y_pred_slr))
print("R2:", r2_score(y_test, y_pred_slr))

print("\n----- Multiple Linear Regression -----")
print("MSE:", mse(y_test, y_pred_mlr))
print("R2:", r2_score(y_test, y_pred_mlr))