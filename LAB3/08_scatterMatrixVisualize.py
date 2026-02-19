"""
Program Name: Scatter Matrix Generator
Description : This program creates a scatter matrix for the Iris dataset to visualize relationships between features
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 02-02-2026
Language    : Python
"""

# Import all modules here
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd


# Load the Iris Dataset
iris = load_iris()

# Convert the iris dataset into a Pandas Dataframe
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

# Visulaization of Pairplot of each numerical variables
pd.plotting.scatter_matrix(
    df, 
    figsize=(10,10),
    diagonal='hist',  
    alpha=0.7
)
plt.show()


