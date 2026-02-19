"""
Program Name: Linear Regression Model on Boston Housing Dataset
Description : This program compares performance of  inbuild Linear Regression model and self-build Gradient Descent based Linear Regression Model
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 19-02-2026
Language    : Python
"""

# Import all modules here
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def linear_regression_gradient_descent(X, Y, learning_rate=0.01, epochs=1000):
    
    X = np.array(X)
    Y = np.array(Y)
    
    n, m = X.shape   # n samples, m features
    
    w = np.zeros(m)  # vector of weights
    b = 0
    
    for _ in range(epochs):
        
        # Prediction
        Y_pred = np.dot(X, w) + b
        
        # Gradients
        dw = (-2/n) * np.dot(X.T, (Y - Y_pred))
        db = (-2/n) * np.sum(Y - Y_pred)
        
        # Update
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b


# Load dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data.astype(float)
Y = boston.target.astype(float)  

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sk_model = LinearRegression()
sk_model.fit(X_train, Y_train)

sk_pred = sk_model.predict(X_test)
sk_w = sk_model.coef_[0]
sk_b = sk_model.intercept_

sk_mse = mean_squared_error(Y_test, sk_pred)
sk_r2 = r2_score(Y_test, sk_pred)

print("Scikit-Learn Results")
print("MSE:", sk_mse)
print("R2:", sk_r2)

gd_w, gd_b = linear_regression_gradient_descent(X_train, Y_train)

gd_pred = np.dot(X_test, gd_w) + gd_b

gd_mse = mean_squared_error(Y_test, gd_pred)
gd_r2 = r2_score(Y_test, gd_pred)

print("\nGradient Descent Results")
print("MSE:", gd_mse)
print("R2:", gd_r2)


plt.figure(figsize=(8,6))

# Actual vs Predicted (Gradient Descent)
plt.scatter(Y_test, gd_pred, color='blue', label='Gradient Descent')

# Actual vs Predicted (Scikit-Learn)
plt.scatter(Y_test, sk_pred, color='green', label='Scikit-Learn')

# Ideal 45-degree line
plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()],
    color='red',
    linestyle='--',
    label='Ideal Fit'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Model Comparison: Actual vs Predicted")
plt.legend()
plt.grid(True)

plt.show()