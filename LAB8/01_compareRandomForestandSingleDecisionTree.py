"""
Program Name: Implement Random Forest and Single Decision Tree 
Description : This program compares Random Forest and Single Decision Tree 
Course      : Machine Learning Laboratory (CS4205)
Date        : 16-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Models

# Single Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Evaluation
def evaluate_model(name, model):
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))

# Evaluate both
evaluate_model("Decision Tree", dt_model)
evaluate_model("Random Forest", rf_model)

# Feature Importance Comparison

# Decision Tree importance
plt.figure()
plt.bar(range(len(dt_model.feature_importances_)), dt_model.feature_importances_)
plt.title("Decision Tree Feature Importance")
plt.show()

# Random Forest importance
plt.figure()
plt.bar(range(len(rf_model.feature_importances_)), rf_model.feature_importances_)
plt.title("Random Forest Feature Importance")
plt.show()