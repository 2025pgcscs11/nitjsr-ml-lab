"""
Program Name: Linear Regression model trainer and its performance evaluation
Description : This program split a dataset into train and test sets
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4201)
Date        : 02-02-2026
Language    : Python
"""

# Import all modules here
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np 

# Load dataset
data = fetch_california_housing()

# Convert to DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# plt.figure(figsize=(8,6)) 
# plt.scatter(X, y, color='blue', label='Data Points') 
# plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line') 
# plt.title('Linear Regression on Random Dataset')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.show()





