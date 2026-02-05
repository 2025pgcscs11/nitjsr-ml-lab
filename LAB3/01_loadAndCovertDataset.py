"""
Program Name: Iris dataset loader and Converter to Panadas Dataframe 
Description : This program loads Iris dataset and convert the datset into Pnadas DataFrame
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

# Import all modules here
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris Dataset
iris = load_iris()

# Convert the iris dataset into a Pandas Dataframe
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

# Display Dataframe
print(df)

# Structure of the Dataset
print(df.info())

#Summary Statistics of all columns
print(df.describe(include='all'))




