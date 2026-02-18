# Import all modules here
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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


# Load dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data.astype(float)
Y = boston.target.astype(float)  

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

sk_model = LinearRegression()
sk_model.fit(X_train, Y_train)

sk_pred = sk_model.predict(X_test)

sk_mse = mean_squared_error(Y_test, sk_pred)
sk_r2 = r2_score(Y_test, sk_pred)

print("Scikit-Learn Results")
print("MSE:", sk_mse)
print("R2:", sk_r2)

w, b = linear_regression_gradient_descent(X_train, Y_train)

gd_pred = np.dot(X_test, w) + b

gd_mse = mean_squared_error(Y_test, gd_pred)
gd_r2 = r2_score(Y_test, gd_pred)

print("\nGradient Descent Results")
print("MSE:", gd_mse)
print("R2:", gd_r2)
