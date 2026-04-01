"""
Program Name: Visulization of Decision Tree using CART Algorithm
Description : This program implements Decision Tree using CART algorithm and Visulaize it with a sample dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 12-03-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# import graphviz

# Decision Tree Model
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf node value


class DecisionTreeCART:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            impurity -= p ** 2
        return impurity

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gini = float("inf")

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                left_gini = self._gini(y[left_idx])
                right_gini = self._gini(y[right_idx])

                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):

            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._most_common_label(y))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature, threshold, left, right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
    
# Visualize a decision tree
def print_tree(node, depth=0):
    if node.value is not None:
        print("  " * depth + f"Leaf: Class {node.value}")
        return

    print("  " * depth + f"[X{node.feature} <= {node.threshold}]")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)



# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
tree = DecisionTreeCART(max_depth=3)
tree.fit(X_train, y_train)

# Predict
predictions = tree.predict(X_test)

# Accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Visualize
print_tree(tree.root)

# clf = DecisionTreeClassifier(max_depth=3)
# clf.fit(X, y)

# dot_data = export_graphviz(clf, out_file=None,
#                           feature_names=data.feature_names,
#                           class_names=data.target_names,
#                           filled=True)

# graph = graphviz.Source(dot_data)
# graph.render("tree", format="png", view=True)