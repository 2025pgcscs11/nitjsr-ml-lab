"""
Program Name: Load and Split Iris dataset into a train and test dataset
Description : This program split a dataset into train and test sets
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4201)
Date        : 02-02-2026
Language    : Python
"""

# Import all modules here
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = df.drop(columns=['species'])
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)








