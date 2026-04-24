"""
Program Name: Implement SVM with different kernel functions
Description : This program implements SVM with different kernel functions and evaluate their performance on a binary classification task
Course      : Machine Learning Laboratory (CS4201)
Date        : 09-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Train SVM with Different Kernels
models = {
    "Linear": SVC(kernel='linear', C=1),
    "Polynomial": SVC(kernel='poly', degree=3, C=1),
    "RBF": SVC(kernel='rbf', gamma=1, C=1)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model


# Evaluate Performance
results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Kernel:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# Decision Boundary Visualization
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.show()


# Plot all kernels
for name, model in trained_models.items():
    plot_decision_boundary(model, X, y, f"{name} Kernel Decision Boundary")