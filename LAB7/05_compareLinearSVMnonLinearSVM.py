"""
Program Name: Implement Linear SVM and Non-Linear SVM 
Description : This program Compares the performance of Linear SVM and Non-Linear SVM on the same dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 09-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ============================
# RBF Kernel
# ============================
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# ============================
# Kernel SVM Class
# ============================
class KernelSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=500, gamma=1.0):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.gamma = gamma

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Convert labels to {-1, 1}
        y = np.where(y <= 0, -1, 1)

        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Precompute Kernel Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = rbf_kernel(X[i], X[j], self.gamma)

        # Training
        for _ in range(self.n_iters):
            for i in range(n_samples):
                decision = np.sum(self.alpha * y * K[:, i]) + self.b
                condition = y[i] * decision >= 1

                if condition:
                    self.alpha[i] -= self.lr * (2 * self.lambda_param * self.alpha[i])
                else:
                    self.alpha[i] += self.lr * (1 - y[i] * decision)
                    self.b += self.lr * y[i]

    def predict(self, X):
        y_pred = []
        for x in X:
            result = 0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 1e-5:  # support vectors
                    result += self.alpha[i] * self.y[i] * rbf_kernel(self.X[i], x, self.gamma)
            result += self.b
            y_pred.append(np.sign(result))
        return np.array(y_pred)
    

# Generate Non-Linear Dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
y = np.where(y == 0, -1, 1)

# Train Model
model = KernelSVM(gamma=2)
model.fit(X, y)


# Decision Boundary Visualization
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.title("Non-Linear SVM with RBF Kernel")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X, y, model)