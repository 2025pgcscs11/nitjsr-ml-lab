"""
Program Name: Box Plotter
Description : This program plots Histograms to show the distribution of a data
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create a box plot for each feature grouped by species
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, orient="h")  # Horizontal box plot for all features

plt.title("Box Plot of Iris Dataset Features", fontsize=16)
plt.xlabel("Value")
plt.ylabel("Features")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Separate box plots for each feature by species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x="species", y=feature, data=df)
    plt.title(f"Box Plot of {feature} by Species")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



