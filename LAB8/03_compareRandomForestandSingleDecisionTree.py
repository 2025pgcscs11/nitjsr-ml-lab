"""
Program Name: Implement Random Forest and Single decision Tree
Description : This program compares Random Forest and Single decision Tree on IRIS dataset
Course      : Machine Learning Laboratory (CS4201)
Date        : 16-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Iris
data = load_iris()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Models

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Evaluate Performance
def evaluate(name, model):
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall   :", recall_score(y_test, y_pred, average='macro'))
    print("F1-score :", f1_score(y_test, y_pred, average='macro'))

evaluate("Decision Tree", dt)
evaluate("Random Forest", rf)


# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.show()