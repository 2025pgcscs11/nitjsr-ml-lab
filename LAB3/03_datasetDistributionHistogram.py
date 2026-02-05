"""
Program Name: Histogram Plotter
Description : This program plots Histograms to show the distribution of a data
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

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Plot histogram for 'sepal length (cm)'
plt.figure(figsize=(8, 6))
plt.hist(df['sepal length (cm)'], bins=20, color='green', edgecolor='black')
plt.title('Frequency Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for 'sepal width (cm)'
plt.figure(figsize=(8, 6))
plt.hist(data['sepal width (cm)'], bins=20, color='red', edgecolor='black')
plt.title('Frequency Histogram of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for 'petal length (cm)'
plt.figure(figsize=(8, 6))
plt.hist(df['petal length (cm)'], bins=20, color='orange', edgecolor='black')
plt.title('Frequency Histogram of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for 'petal width (cm)'
plt.figure(figsize=(8, 6))
plt.hist(data['petal width (cm)'], bins=20, color='blue', edgecolor='black')
plt.title('Frequency Histogram of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')
plt.show()