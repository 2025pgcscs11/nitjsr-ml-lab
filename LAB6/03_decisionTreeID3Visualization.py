"""
Program Name: Visulization of Decision Tree using ID3 Algorithm
Description : This program implements Decision Tree using ID3 algorithm and Visulaize it with a sample dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 12-03-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from collections import Counter
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# import graphviz


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class ID3Tree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    # 🔸 Entropy
    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    # 🔸 Information Gain
    def _information_gain(self, X_column, y, threshold):
        parent_entropy = self._entropy(y)

        left_idx = X_column <= threshold
        right_idx = X_column > threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_idx), np.sum(right_idx)

        e_l = self._entropy(y[left_idx])
        e_r = self._entropy(y[right_idx])

        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        return parent_entropy - child_entropy

    # 🔸 Best split
    def _best_split(self, X, y):
        best_feature, best_thresh = None, None
        best_gain = -1

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                gain = self._information_gain(X[:, feature], y, t)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = t

        return best_feature, best_thresh

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return Node(value=Counter(y).most_common(1)[0][0])

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._build_tree(X[left_idx], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

def print_tree(node, depth=0):
    if node.value is not None:
        print("  " * depth + f"Leaf: Class {node.value}")
        return

    print("  " * depth + f"[X{node.feature} <= {node.threshold}]")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train ID3
tree = ID3Tree(max_depth=3)
tree.fit(X_train, y_train)

# Predict
preds = tree.predict(X_test)

# Accuracy
accuracy = np.sum(preds == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Print tree
print_tree(tree.root)


# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# clf.fit(X, y)

# dot_data = export_graphviz(
#     clf,
#     out_file=None,
#     feature_names=data.feature_names,
#     class_names=data.target_names,
#     filled=True
# )

# graph = graphviz.Source(dot_data)
# graph.render("id3_tree", format="png", view=True)