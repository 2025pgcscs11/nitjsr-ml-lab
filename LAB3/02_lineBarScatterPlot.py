"""
Program Name: Plotter of Line, Bar, Scatter graph
Description : This program plots Line graph, Bar graph, Scatter graph using Matplotlib
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

# Import all modules here
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset into a pandas DataFrame
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)


# Line Plot
plt.figure(figsize=(6, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length', color='orange')
plt.title('Line Plot of Sepal & Petal Length')
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Bar Plot (Average Petal Length per Species)
avg_petal_length = df.groupby('species')['petal length (cm)'].mean()

plt.figure(figsize=(6, 4))
avg_petal_length.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Scatter Plot
plt.figure(figsize=(6, 4))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], label=species)

plt.title('Scatter Plot of Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()