"""
Program Name: AdaBoost with Decision Tree on Titanic Dataset
Description : Apply AdaBoost to improve weak classifier performance
Course      : Machine Learning Laboratory (CS4205)
Date        : 16-04-2026
Language    : Python
"""

# ==============================
# IMPORTS
# ==============================
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# LOAD DATASET (AUTO FETCH)
# ==============================
df = sns.load_dataset('titanic')

print("First 5 rows:\n", df.head())
print("\nColumns:\n", df.columns)


# ==============================
# DATA PREPROCESSING
# ==============================

# Select features
df = df[['survived', 'pclass', 'sex', 'age', 'fare']].copy()

# Handle missing values (correct way)
df['age'] = df['age'].fillna(df['age'].median())

# Encode categorical variable
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Final safety check (VERY IMPORTANT)
print("\nMissing values:\n", df.isnull().sum())

# Drop any remaining NaN rows (just in case)
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
# WEAK CLASSIFIER (DECISION STUMP)
# ==============================
weak_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
weak_clf.fit(X_train, y_train)

y_pred_weak = weak_clf.predict(X_test)


# ==============================
# ADABOOST MODEL
# ==============================
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)


# ==============================
# EVALUATION
# ==============================
print("\n===== RESULTS =====")

print("Weak Learner Accuracy :", accuracy_score(y_test, y_pred_weak))
print("AdaBoost Accuracy     :", accuracy_score(y_test, y_pred_ada))

print("\nClassification Report (AdaBoost):\n")
print(classification_report(y_test, y_pred_ada))