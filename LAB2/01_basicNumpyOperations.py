"""
Program Name: Basic Operations on Numpy Array
Description : This program implements four basic operations respectively addition, Subtraction, multiplication, division
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 29-01-2026
Language    : Python
"""


# Importing python modules here
import numpy as np

def basic_operations(arr1,arr2):
    """
    Perform basic operations (addition, subtraction,multiplication, division) on two Numpy array

    Parameters:
        arr1 (Numpy array): The first array of numbers
        arr2 (Numpy array): The second array of numbers
    
    Returns:
        dict: A dictionary contains addition, subtraction, multiplication, division
    """
    # Shape check
    if arr1.shape != arr2.shape:
        return None
    

    # Numeric type check
    if not (np.issubdtype(arr1.dtype, np.number) and np.issubdtype(arr2.dtype, np.number)):
        return None


    addition = arr1 + arr2
    subtraction = arr1 -  arr2
    multiplication = arr1 * arr2

    # Check if any element is zero
    if(np.any(arr2 == 0)):
        division = None
    else:
        division = arr1 / arr2

    return {
        "addition": addition,
        "subtraction": subtraction,
        "multiplication": multiplication,
        "division": division
    }


def run_test_cases():
    """
    Run various test cases on basic_operations function 
    """

    test_cases = [
        # Basic positive integers
        (np.array([1, 2, 3]),
        np.array([4, 5, 6])),

        # Different values
        (np.array([10, 20, 30]),
        np.array([1, 2, 3])),

        # Zeros included
        (np.array([0, 5, 10]),
         np.array([2, 0, 4])),

        # Negative numbers
        (np.array([-1, -2, -3]),
        np.array([3, -2, 1])),

        # Mixed positive and negative
        (np.array([5, -10, 15]),
         np.array([-5, 10, -15])),

        # Floating point values
        (np.array([1.5, 2.5, 3.5]),
         np.array([0.5, 1.25, 1.75])),

        # Integer + float mix
        (np.array([1, 2, 3]),
        np.array([1.5, 2.5, 3.5])),

        # Division-safe
        (np.array([10, 20, 30]),
        np.array([2, 4, 5])),

        # Division by zero edge case
        (np.array([5, 10, 15]),
        np.array([1, 0, 3])),

        # Large numbers
        (np.array([100000, 200000, 300000]),
         np.array([10, 20, 30])),

        # Single-element arrays
        (np.array([5]),
         np.array([2])),

        # Unequal length (error case)
        (np.array([1, 2, 3]),
        np.array([4, 5])),

        # 2D Array
        ( 
        np.array([[1, 2, 3],
              [4, 5, 6]]),
        np.array([[6, 5, 4],
              [3, 2, 1]])
        ),

        # Non-numerice data
        (np.array(['a', 'c', 'v']),
        np.array(['N', 'l','k'])),

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
        print(f"\nTest Case: {idx}")
        arr1,arr2 = case
        print("First Array is:",arr1)
        print("Second Array is:",arr2)
        results = basic_operations(arr1,arr2)
        
        if results is None:
            print("Operation not possible (shape mismatch or non-numeric data)")
        else:
            print("Addition: ", results["addition"])
            print("Subtraction: ", results["subtraction"])
            print("Multiplication: ", results["multiplication"])

            if results["division"] is not None:
                print("Division: ", results["division"])
            else:
                print("Division by zero element is not allowed")


# Program execution starts here
if __name__ == "__main__":
    run_test_cases()



