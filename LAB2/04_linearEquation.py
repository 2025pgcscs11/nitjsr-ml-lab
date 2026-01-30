"""
Program Name: System of Linear Equations solver
Description : This program solve system of linear equations
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 29-01-2026
Language    : Python
"""

# Importing python modules here
import numpy as np

def linear_equations_solver(A,B):
    """
    Solve system of linear equations written in form of matrix A and B
    
    Parameters:
        A (matrix): Coefficient matrix of linear equations
        B (matrix): Matrix of constant vlue in linear equations

    Returns:
        tuple: A tuple of solution 
    """

    row, col = A.shape
    if row != col:
        return None

    detA = np.linalg.det(A)

    if detA == 0:
        return None
    
    return np.matmul(np.linalg.inv(A),B)


def run_all_testcases():
    """
    Run various test cases on basic_operations functio
    """

    test_cases = [

    # ---------- UNIQUE SOLUTION ----------
    (
        np.array([[2, 1],
                  [1, 3]]),
        np.array([5, 6])
    ),

    (
        np.array([[1, 2],
                  [3, 4]]),
        np.array([5, 11])
    ),

    (
        np.array([[3, 2, -1],
                  [2, -2, 4],
                  [-1, 0.5, -1]]),
        np.array([1, -2, 0])
    ),


    # ---------- NO SOLUTION  ----------
    (
        np.array([[1, 1],
                  [2, 2]]),
        np.array([2, 5])
    ),

    (
        np.array([[2, 4],
                  [1, 2]]),
        np.array([6, 3.5])
    ),

    # ---------- FLOATING POINT ----------
    (
        np.array([[0.1, 0.2],
                  [0.3, 0.4]]),
        np.array([0.3, 0.7])
    ),


    # ---------- ZERO ROW ----------
    (
        np.array([[1, 2],
                  [0, 0]]),
        np.array([3, 1])
    ),

    ]

    for idx, case in enumerate(test_cases, start=1):
        A, b = case
        print("\nTest Case: ",idx)
        print("Equation Matrices are: ",A,"  ",b)
        solution = linear_equations_solver(A,b)
        if solution is None:
            print("No Solutions or Infinite solutions")
        else:
            print("Unique solutions are: ",solution)



# Program execution starts here
if __name__ == "__main__":
    run_all_testcases()
