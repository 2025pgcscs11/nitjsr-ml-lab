"""
Program Name: Comparator of Simple Linear Regression and Multiple Linear Regression Model
Description : This program compares and calculates various evaluation matrices of Simple Linear Regression Model and Multiple Linear Regression Model using the IRIS dataset
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4201)
Date        : 19-02-2026
Language    : Python
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score , mean_squared_error, median_absolute_error,mean_absolute_percentage_error



# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Target Feature
y = df['petal length (cm)']

# Simple Linear Regression (One feature)
X_simple = df[['sepal length (cm)']]


# Multiple Linear Regression (More than one faetures)
X_multiple = df.drop(columns=['petal length (cm)'])


# Dataset spliting into train and test
X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

X_train_m, X_test_m, _, _ = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42)


# Simple Linear Regression
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train)

# Multiple Linear Regression
model_multiple = LinearRegression()
model_multiple.fit(X_train_m, y_train)

# Model prediction
y_pred_simple = model_simple.predict(X_test_s)
y_pred_multiple = model_multiple.predict(X_test_m)

# Comparison of performance

# Simple LR metrics
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)

# Multiple LR metrics
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print("Simple Linear Regression")
print("MSE:", mse_simple)
print("R2 Score:", r2_simple)

print("\nMultiple Linear Regression")
print("MSE:", mse_multiple)
print("R2 Score:", r2_multiple)

# Other evaluation metrices
# Metrics for Simple LR
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
medae_simple = median_absolute_error(y_test, y_pred_simple)
mape_simple = mean_absolute_percentage_error(y_test, y_pred_simple)

# Metrics for Multiple LR
rmse_multiple = np.sqrt(mean_squared_error(y_test, y_pred_multiple))
medae_multiple = median_absolute_error(y_test, y_pred_multiple)
mape_multiple = mean_absolute_percentage_error(y_test, y_pred_multiple)

# ---------------------------
# Print Results
# ---------------------------
print("===== Simple Linear Regression =====")
print("RMSE:", rmse_simple)
print("Median Absolute Error:", medae_simple)
print("MAPE:", mape_simple)

print("\n===== Multiple Linear Regression =====")
print("RMSE:", rmse_multiple)
print("Median Absolute Error:", medae_multiple)
print("MAPE:", mape_multiple)


# Impact of Multiple Predictors on Model Accuracy
# Case 1: Predictors are Informative
    # MSE ↓
    # R² ↑ significantly
    # Model generalizes better

# Case 2: Predictors are Redundant / Noisy
    # R² ↑ slightly
    # Adjusted R² ↓
    # Test MSE may increase
    # Overfitting occurs