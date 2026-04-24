"""
Program Name: Visulization of Random Forest
Description : This program implements Random Forest and Visulaize it with a sample dataset
Course      : Machine Learning Laboratory (CS4201)
Date        : 12-03-2026
Language    : Python
"""

# Import all modules here
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import export_graphviz
# import graphviz

#  Decision Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.root = self._grow_tree(X, y)

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_feature, best_thresh = None, None
        best_gini = float("inf")

        feature_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)

        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = (len(left)/len(y)) * self._gini(left) + \
                       (len(right)/len(y)) * self._gini(right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_thresh = t

        return best_feature, best_thresh

    def _grow_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split):

            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feat, thresh = self._best_split(X, y)

        if feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_idx = X[:, feat] <= thresh
        right_idx = X[:, feat] > thresh

        left = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feat, thresh, left, right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ------------------ Random Forest ------------------

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])
    

def print_tree(node, depth=0):
    if node.value is not None:
        print("  " * depth + f"Leaf: {node.value}")
        return
    print("  " * depth + f"[X{node.feature} <= {node.threshold}]")
    print_tree(node.left, depth+1)
    print_tree(node.right, depth+1)


# Load data
data = load_iris()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest
rf = RandomForest(n_trees=3, max_depth=3, n_features=2)
rf.fit(X_train, y_train)

# Predict
preds = rf.predict(X_test)

# Accuracy
accuracy = np.sum(preds == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Visualize individual trees
for i, tree in enumerate(rf.trees):
    print(f"\nTree {i+1}:")
    print_tree(tree.root)


# rf = RandomForestClassifier(n_estimators=3, max_depth=3)
# rf.fit(X, y)

# Visualize first tree
# tree = rf.estimators_[0]

# dot_data = export_graphviz(
#     tree,
#     out_file=None,
#     feature_names=data.feature_names,
#     class_names=data.target_names,
#     filled=True
# )

# graph = graphviz.Source(dot_data)
# graph.render("rf_tree", format="png", view=True)