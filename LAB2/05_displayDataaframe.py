"""
Program Name: CSV file loader and DataFrame Displayer
Description : This program displays first 10 rows of a pandas dataframe
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 29-01-2026
Language    : Python
"""

# Importing python modules here
import pandas as pd


def file_loader_displayer(path):
    """
    Load CSV file located in path

    Parameters:
        path (string): Location to the CSV file
    
    Returns:
        Pandas DataFrame: A DataFrame
    """

    # Load CSV file
    df = pd.read_csv(path)

    return df


# Program execution starts here
if __name__ == "__main__":
    df = file_loader_displayer("data.csv")

    # Display first 10 rows    
    print(df.head(10),end ="\n\n")

    # Add a Column
    df["Pass"] = df["Score"] >= 80
    print(df.head(),end ="\n\n")

    # Drop a Column
    df_dropped = df.drop(columns=["Age"])
    print(df_dropped.head(),end ="\n\n")

    # filter rows based on score > 85
    high_scorers = df[df["Score"] > 85]
    print(high_scorers,end ="\n\n")

    # Mean of Score column
    print("Mean Score:", df["Score"].mean(),end ="\n\n")

    # Median of Score column
    print("Median Score:", df["Score"].median(),end ="\n\n")

    # Mode of Score column
    print("Mode Score:\n",df["Score"].mode(),end ="\n\n")

    # Standard Deviation of Score column
    print("Standard Deviation:", df["Score"].std(),end ="\n\n")

    # Group data by a SCORE column and calculate aggregate statistics
    grouped_stats = df.groupby("Department")["Score"].agg(
        mean_score="mean",
        max_score="max",
        min_score="min",
        count_students="count"
    )   

    print(grouped_stats,end ="\n\n")





