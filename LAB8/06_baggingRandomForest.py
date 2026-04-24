"""
Program Name: Bagging using Random Forest on Titanic Dataset
Description : Apply Random Forest and evaluate using accuracy and cross-validation
Course      : Machine Learning Laboratory (CS4201)
Date        : 16-04-2026
Language    : Python
"""

# ==============================
# IMPORTS
# ==============================
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


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

# Drop any remaining NaN
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
# RANDOM FOREST MODEL (BAGGING)
# ==============================
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# Train model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)


# ==============================
# EVALUATION (TEST ACCURACY)
# ==============================
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==============================
# CROSS-VALIDATION
# ==============================
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())