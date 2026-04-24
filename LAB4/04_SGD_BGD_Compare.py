"""
Program Name: Comparator of Stochastic Gradient Descent Linear Regression and Batch Gradient Descent Linear Regression Model
Description : This program implements Gradient Descent Linear Regression Model and Compares both SGD and BGD Linear Regression Model using the numerical columns of the IRIS Dataset
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4201)
Date        : 19-02-2026
Language    : Python
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Features and target
X = df.drop(columns=['petal length (cm)']).values
y = df['petal length (cm)'].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add bias term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Initialize parameters
np.random.seed(42)
theta = np.random.randn(X_train_b.shape[1], 1)

learning_rate = 0.01
epochs = 1000
m = len(X_train_b)

# Batch Gradient Descent
for epoch in range(epochs):
    gradients = (2/m) * X_train_b.T.dot(X_train_b.dot(theta) - y_train)
    theta -= learning_rate * gradients

# Predictions
y_pred_bgd = X_test_b.dot(theta)

# Evaluation
mse_bgd = mean_squared_error(y_test, y_pred_bgd)
r2_bgd = r2_score(y_test, y_pred_bgd)

print("Batch GD MSE:", mse_bgd)
print("Batch GD R2:", r2_bgd)


# Reinitialize parameters
theta_sgd = np.random.randn(X_train_b.shape[1], 1)

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_train_b[random_index:random_index+1]
        yi = y_train[random_index:random_index+1]
        
        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        theta_sgd -= learning_rate * gradients

# Predictions
y_pred_sgd = X_test_b.dot(theta_sgd)

# Evaluation
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print("SGD MSE:", mse_sgd)
print("SGD R2:", r2_sgd)


# When to use each optimization technique? 

# Use Batch Gradient Descent When:
    # Dataset is small/medium
    # Stable convergence is required
    # High numerical precision needed
    # Memory is not a constraint

# Use Stochastic Gradient Descent When:
    # Dataset is very large
    # Online learning required
    # Memory limited
    # Need faster updates
    # Streaming data scenario