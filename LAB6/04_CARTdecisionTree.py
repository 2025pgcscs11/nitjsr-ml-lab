"""
Program Name: Visulization of Decision Tree using CART Algorithm on Titanic Datset
Description : This program implements Decision Tree using CART algorithm and Visulaize it with Titanic Dataset
Course      : Machine Learning Laboratory (CS4201)
Date        : 12-03-2026
Language    : Python
"""

# Import all modules here
import seaborn as sns
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = sns.load_dataset('titanic')

# Select useful features
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Encode categorical variables
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Features and target
X = df.drop('survived', axis=1)
y = df['survived']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CART model (Gini is default)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)


# Predictions & Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Visualize Confusion Matrix
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.show()

# Visualize Decision Tree
# plt.figure(figsize=(12, 8))
# plot_tree(
#     model,
#     feature_names=X.columns,
#     class_names=["Not Survived", "Survived"],
#     filled=True
# )
# plt.show()