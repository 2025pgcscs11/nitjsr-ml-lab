"""
Program Name: Evaluator of self written Linear Regression Model on Boston Housing Dataset
Description : This program implements Linear Regression from Scratch (Least Squares Method)for Boston Housing Dataset and evalate its performance
Course      : Machine Learning Laboratory (CS4205)
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
        self.weights = np.linalg.pinv(X_b) @ y   # pseudo-inverse

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        return X_b @ self.weights


# -------------------------------
# MSE Function
# -------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# Load Dataset
# -------------------------------
boston = fetch_openml(name="boston", version=1, as_frame=False)

X = boston.data
y = boston.target

# Convert data to numeric 
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LinearRegressionLS()
model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
print("Mean Squared Error (MSE):", mse(y_test, y_pred))