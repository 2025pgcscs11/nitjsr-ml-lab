"""
Program Name: Stacking Classifier on Titanic Dataset
Description : Implement stacking using multiple classifiers
Course      : Machine Learning Laboratory (CS4201)
Date        : 16-04-2026
Language    : Python
"""

# ==============================
# IMPORTS
# ==============================
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier


# ==============================
# LOAD DATASET
# ==============================
df = sns.load_dataset('titanic')

# ==============================
# PREPROCESSING
# ==============================
df = df[['survived', 'pclass', 'sex', 'age', 'fare']].copy()

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())

# Encode categorical
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Final cleanup
df = df.dropna()

# Split X and y
X = df.drop('survived', axis=1)
y = df['survived']


# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# BASE MODELS (LEVEL-0)
# ==============================
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('lr', LogisticRegression(max_iter=1000))
]


# ==============================
# META MODEL (LEVEL-1)
# ==============================
meta_model = LogisticRegression()


# ==============================
# STACKING CLASSIFIER
# ==============================
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Train
stack_model.fit(X_train, y_train)

# Predict
y_pred = stack_model.predict(X_test)


# ==============================
# EVALUATION
# ==============================
print("Stacking Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))