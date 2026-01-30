"""
Program Name: Matrix Multiplication Calcultor
Description : This program calculates multiplication of matrices
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 29-01-2026
Language    : Python
"""

# Importing python modules here
import numpy as np


def matrix_multiplication(mat1,mat2):
    """
    Calculate matrix multiplication of two matrix of any shape

    Parameters:
        mat1 (2D Numpy Array): First matrix
        mat2 (2D Numpy Array): Second matrix

    Returns:
        result (2D Numpy Array): Multiplication of two matrix 
    """

    row1 ,col1 = mat1.shape
    row2, col2 = mat2.shape

    if col1 != row2:
        return None

    # Numeric type check
    if not (np.issubdtype(mat1.dtype, np.number) and np.issubdtype(mat2.dtype, np.number)):
        return None

    result = np.matmul(mat1,mat2)

    return result


def run_all_testcases():
    """
    Run various test cases on basic_operations function 
    """

    test_cases = [
         # 2D Array
        ( 
        np.array([[1, 2, 3],
              [4, 5, 6]]),
        np.array([[6, 5, 4],
              [3, 2, 1]])
        ),

        # Non-numerice data
        ( 
        np.array([['e', 'c', 'p'],
              ['a', 'd', 's']]),
        np.array([['q', '5', '3'],
              ['d', 'r', '5']])
        ),

        # 2D Array with 0
        (
        np.array([[1, 2, 3],
              [4, 5, 6]]),
        np.array([[6, 0, 4],
              [3, 0, 1]])
        ),

        # 2D Array with Unequal length
        (
        np.array([[1, 2],
              [4, 5]]),
        np.array([[6, 5, 4],
              [3, 2, 1]])
        )
    ]

    for idx, case in enumerate(test_cases, start=1):
        mat1, mat2 = case
        print("\nTest Case: ",idx)
        print("Matrix 1 is: ",mat1)
        print("Matrix 2 is: ",mat2)
        result = matrix_multiplication(mat1,mat2)
        if result is None:
            print("Incompatible Shapes")
        else:
            print("Matrix multiplication of matrix 1 & 2: ",result)



# Program execution starts here
if __name__ == "__main__":
    run_all_testcases()