"""
Program Name: Evaluator of self written Linear Regression Model
Description : This program implements Linear Regression from Scratch (Least Squares Method) and evalate its performance
Course      : Machine Learning Laboratory (CS4205)
Date        : 05-03-2026
Language    : Python
"""

# Import all modules here
import numpy as np

# Linear Regression Model using Least Square Method
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
    
# Calculate mean squared error value
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Calculate R2 score
def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)


# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.5, 11.7, 13.9, 16.2])
# Train model
model = LinearRegressionLS()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
print("MSE:", mse(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))