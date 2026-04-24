"""
Program Name: Evaluator of self written Logistic Regression Model
Description : This program implements Logistic Regression from Scratch (Least Squares Method) and evalate its performance
Course      : Machine Learning Laboratory (CS4201)
Date        : 05-03-2026
Language    : Python
"""

# Import all modules here
import numpy as np

# Logistic Regression Model
class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([1 if i >= 0.5 else 0 for i in probs])
    
# Evaluate Confusion matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[TN, FP],
                     [FN, TP]])

# Accuracy Score
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])       # Feature: study hours
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])                       # Target: 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
cm = confusion_matrix(y, y_pred)
acc = accuracy(y, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)