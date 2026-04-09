"""
Program Name: Implement Linear SVM
Description : This program implements Linear SVM and plot the decision boundary on the Iris dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 09-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# ============================
# Load Iris Dataset
# ============================
iris = datasets.load_iris()
X = iris.data[:, :2]   # take first 2 features
y = iris.target

# Take only 2 classes (0 and 1)
mask = y < 2
X = X[mask]
y = y[mask]

# Convert labels to {-1, 1}
y = np.where(y == 0, -1, 1)

# ============================
# Linear SVM Class
# ============================
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# ============================
# Train Model
# ============================
model = LinearSVM()
model.fit(X, y)

# Decision Boundary Visualization
def plot_decision_boundary(X, y, model):
    def hyperplane(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]

    fig, ax = plt.subplots()

    # Scatter points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)

    x0_1 = np.min(X[:, 0])
    x0_2 = np.max(X[:, 0])

    # Decision boundary
    x1_1 = hyperplane(x0_1, model.w, model.b, 0)
    x1_2 = hyperplane(x0_2, model.w, model.b, 0)

    # Margins
    x1_1_m = hyperplane(x0_1, model.w, model.b, 1)
    x1_2_m = hyperplane(x0_2, model.w, model.b, 1)

    x1_1_p = hyperplane(x0_1, model.w, model.b, -1)
    x1_2_p = hyperplane(x0_2, model.w, model.b, -1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'r--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'r--')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Linear SVM on Iris Dataset")

    plt.show()

plot_decision_boundary(X, y, model)


