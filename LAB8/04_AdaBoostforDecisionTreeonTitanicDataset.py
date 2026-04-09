"""
Program Name: Implement AdaBoost for boosting a Decision Tree
Description : This program implements AdaBoost for boosting a Decision Tree on Titanic dataset
Course      : Machine Learning Laboratory (CS4205)
Date        : 16-04-2026
Language    : Python
"""

# Import all modules here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load Titanic dataset (train.csv from Kaggle)
df = sns.load_dataset('titanic')

print(df.head())

# Data Preprocessing

# Select features
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical feature
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Split X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Weak Classifier
weak_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
weak_clf.fit(X_train, y_train)


# AdaBoost Model
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)


# Evaluation
# Weak model predictions
y_pred_weak = weak_clf.predict(X_test)

# AdaBoost predictions
y_pred_ada = ada.predict(X_test)

print("Weak Learner Accuracy:", accuracy_score(y_test, y_pred_weak))
print("AdaBoost Accuracy   :", accuracy_score(y_test, y_pred_ada))

print("\nAdaBoost Classification Report:\n")
print(classification_report(y_test, y_pred_ada))