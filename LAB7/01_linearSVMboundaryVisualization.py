"""
Program Name: Implement SVM
Description : This program implements SVM and shows the decision boundaries
Course      : Machine Learning Laboratory (CS4205)
Date        : 09-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

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

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)
    
# Generate Non-Linear Dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
# Convert labels {0,1} → {-1,1}
y = np.where(y == 0, -1, 1)

# Train Model
model = LinearSVM()
model.fit(X, y)


# Decision Boundary Visualization
def plot_decision_boundary(X, y, model):
    def hyperplane(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plt.scatter(X[:,0], X[:,1], c=y)

    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])

    # Decision boundary (w·x + b = 0)
    x1_1 = hyperplane(x0_1, model.w, model.b, 0)
    x1_2 = hyperplane(x0_2, model.w, model.b, 0)

    # Margin lines (w·x + b = ±1)
    x1_1_m = hyperplane(x0_1, model.w, model.b, 1)
    x1_2_m = hyperplane(x0_2, model.w, model.b, 1)

    x1_1_p = hyperplane(x0_1, model.w, model.b, -1)
    x1_2_p = hyperplane(x0_2, model.w, model.b, -1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'r--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'r--')

    plt.title("Linear SVM Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, model)