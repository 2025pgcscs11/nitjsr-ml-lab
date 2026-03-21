"""
Program Name: Generator of ROC curve in a Binary Classification task
Description : This program implements Logistic Regression from Scratch for Titanic Dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 05-03-2026
Language    : Python
"""

# -------------------------------
# Import all modules here
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# -------------------------------
# Logistic Regression (From Scratch)
# -------------------------------
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


# -------------------------------
# Load Dataset (Titanic - direct)
# -------------------------------
df = sns.load_dataset('titanic')
df = df[['pclass', 'sex', 'age', 'fare', 'survived']]

# Preprocessing
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['age'] = df['age'].fillna(df['age'].mean())
df = df.dropna()

X = df.drop('survived', axis=1).values.astype(float)
y = df['survived'].values.astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# ROC Curve
# -------------------------------
y_prob = model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()