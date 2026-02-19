"""
Program Name: Linear Regression Model
Description : This program implements Linear Regression from Scratch (Least Squares Method & Gradient Descent)
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def simple_linear_regression(X, Y):
    """
    Implements Simple Linear Regression using Least Squares Method.
    
    Parameters:
    X : list or array-like (independent variable)
    Y : list or array-like (dependent variable)
    
    Returns:
    m : slope
    c : intercept
    """
    
    n = len(X)
    
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_xy = sum(x * y for x, y in zip(X, Y))
    sum_x2 = sum(x * x for x in X)
    
    # Calculate slope (m)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x ** 2
    
    if denominator == 0:
        raise ValueError("Denominator is zero. Cannot compute slope.")
    
    m = numerator / denominator
    
    # Calculate intercept (c)
    c = (sum_y - m * sum_x) / n
    
    return m, c


def linear_regression_gradient_descent(X, Y, learning_rate=0.01, epochs=1000):
    """
    Linear Regression using Batch Gradient Descent
    
    Parameters:
    X : numpy array (independent variable)
    Y : numpy array (target variable)
    learning_rate : step size
    epochs : number of iterations
    
    Returns:
    w : learned weight
    b : learned bias
    cost_history : list of cost values per epoch
    """
    
    n = len(X)
    
    # Initialize parameters
    w = 0
    b = 0
    
    for _ in range(epochs):
        
        # Predictions
        Y_pred = w * X + b
        
        # Compute Gradients
        dw = (-2/n) * np.sum(X * (Y - Y_pred))
        db = (-2/n) * np.sum(Y - Y_pred)
        
        # Update Parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
    
    return w, b


# Sample Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

# Scikit-learn Implementation
model = LinearRegression()
model.fit(X, Y)


# Least-Squares 
ls_w, ls_b = simple_linear_regression(X.flatten(), Y)
ls_pred = ls_w * X.flatten() + ls_b

# Gradient Descent
gd_w, gd_b = linear_regression_gradient_descent(X.flatten(), Y)
gd_pred = gd_w * X.flatten() + gd_b

# Scikit-learn
sk_pred = model.predict(X)
sk_w = model.coef_[0]
sk_b = model.intercept_


# Evaluation
print("Least Squares:")
print("     w =", ls_w, "b =", ls_b)
print("     MSE =", mean_squared_error(Y, ls_pred))
print("     R2 =", r2_score(Y, ls_pred))

print("\nGradient Descent:")
print("     w =", gd_w, "b =", gd_b)
print("     MSE =", mean_squared_error(Y, gd_pred))
print("     R2 =", r2_score(Y, gd_pred))

print("\nScikit-learn:")
print("     w =", sk_w, "b =", sk_b)
print("     MSE =", mean_squared_error(Y, sk_pred))
print("     R2 =", r2_score(Y, sk_pred))


plt.figure(figsize=(8,6))

# Scatter plot of actual data
plt.scatter(X, Y, color='black', label='Actual Data')

# Regression lines
plt.plot(X, [ls_w*x + ls_b for x in X], color='red',   label='Least Squares')
plt.plot(X, [gd_w*x + gd_b for x in X], color='blue',  label='Gradient Descent')
plt.plot(X, [sk_w*x + sk_b for x in X], color='green', label='Scikit-learn')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Comparison of Linear Regression Methods")
plt.legend()
plt.grid(True)

plt.show()