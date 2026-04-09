"""
Program Name: Implement AdaBoost for boosting a weak classifier
Description : This program implements AdaBoost for boosting a weak classifier and analyze the improvement.
Course      : Machine Learning Laboratory (CS4205)
Date        : 16-04-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    flip_y=0.2, class_sep=0.5, random_state=42
)

# Convert labels to {-1, 1}
y = np.where(y == 0, -1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Weak Learner
def get_weak_learner():
    return DecisionTreeClassifier(max_depth=1)


# AdaBoost Implementation
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Initialize weights
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = get_weak_learner()
            model.fit(X, y, sample_weight=w)

            predictions = model.predict(X)

            # Compute weighted error
            error = np.sum(w * (predictions != y)) / np.sum(w)

            # Avoid division issues
            error = max(error, 1e-10)

            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # Update weights
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            # Store
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)

        return np.sign(final_pred)
    

# Train Models
# Weak classifier (single stump)
weak_model = get_weak_learner()
weak_model.fit(X_train, y_train)

# AdaBoost
ada = AdaBoost(n_estimators=50)
ada.fit(X_train, y_train)

# Evaluation

# Predictions
y_pred_weak = weak_model.predict(X_test)
y_pred_ada = ada.predict(X_test)

# Accuracy
print("Weak Learner Accuracy:", accuracy_score(y_test, y_pred_weak))
print("AdaBoost Accuracy   :", accuracy_score(y_test, y_pred_ada))


# Decision Boundary Visualization
def plot_boundary(model, X, y, title):
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
    plt.title(title)
    plt.show()

plot_boundary(weak_model, X_test, y_test, "Weak Learner")
plot_boundary(ada, X_test, y_test, "AdaBoost")