"""
Program Name: Implement Non-linear SVM with RBF kernel
Description : This program implements Non-Linear SVM with RBF kernel for a non-linear dataset
Course      : Machine Learning Laboratory ((CS4201)
Date        : 09-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


# ============================
# RBF Kernel Function
# ============================
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# ============================
# Kernel SVM Class
# ============================
class KernelSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=500, gamma=0.5):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert labels to {-1, 1}
        y = np.where(y <= 0, -1, 1)

        self.X = X
        self.y = y

        # Initialize alphas
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
                condition = y[i] * (np.sum(self.alpha * y * K[:, i]) + self.b) >= 1

                if condition:
                    self.alpha[i] -= self.lr * (2 * self.lambda_param * self.alpha[i])
                else:
                    self.alpha[i] += self.lr * (1 - y[i] * (np.sum(self.alpha * y * K[:, i]) + self.b))
                    self.b += self.lr * y[i]

    def predict(self, X):
        y_pred = []
        for x in X:
            result = 0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 0:
                    result += self.alpha[i] * self.y[i] * rbf_kernel(self.X[i], x, self.gamma)
            result += self.b
            y_pred.append(np.sign(result))
        return np.array(y_pred)
    

# Non-Linear Dataset (Circles)
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5)
# Convert labels to {-1, 1}
y = np.where(y == 0, -1, 1)

# Train Model
model = KernelSVM(gamma=2)
model.fit(X, y)

# Decision Boundary Visualization
def plot_nonlinear_decision_boundary(X, y, model):
    def predict_grid(x, y):
        return model.predict(np.array([[x, y]]))[0]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = np.array([predict_grid(x, y) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Non-Linear SVM with RBF Kernel")
    plt.show()

plot_nonlinear_decision_boundary(X, y, model)