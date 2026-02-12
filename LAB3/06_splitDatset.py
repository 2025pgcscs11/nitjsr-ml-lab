"""
Program Name: Load and Split Iris dataset into a train and test dataset
Description : This program split a dataset into train and test sets
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("IRIS.csv")

X = df.drop(columns=['species'])
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)








