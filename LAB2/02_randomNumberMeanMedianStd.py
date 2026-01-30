"""
Program Name: Mean,Median,Standard Deviation Calculator
Description : This program calculate the mean, median, standard deviation of a randomly generated Numpy array
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 29-01-2026
Language    : Python
"""

# Importing python modules here
import numpy as np

def random_number(arr):
    """
    Calculate the mean, median, standard deviation of array arr

    Parameters:
        arr (Numpy array): The input array

    Returns:
        mean (int/float): Mean of the array
        median (int,float): Median of the array
        standard_deviation (int/float): Standard Deviation of the array
    """

    # Reject empty arrays
    if arr.size == 0:
        return None, None, None

    # Reject non-numeric arrays (strings, objects)
    if not np.issubdtype(arr.dtype, np.number):
        return None, None, None
        
    return np.mean(arr), np.median(arr), np.std(arr)


def run_all_testcases():
    """
    Run various test cases on random_number function 
    """

    test_cases = [
        # ---------- 1D Arrays ----------
        np.array([1, 2, 3, 4]),                   
        np.array([1.5, 2.5, 3.5]),                 
        np.array(['a', 'b', 'c']),                
        np.array([True, False, True]),           

        # ---------- 2D Arrays ----------
        np.array([[1, 2],
              [3, 4]]),                       
        np.array([[1.1, 2.2],
              [3.3, 4.4]]),                   
        np.array([['x', 'y'],
              ['z', 'w']]),                    
        np.array([[True, False],
              [False, True]]),                

        # ---------- 3D Arrays ----------
        np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]]),             

        np.array([[['a', 'b'], ['c', 'd']],
              [['e', 'f'], ['g', 'h']]]),     

        # ---------- Edge / Special Cases ----------
        np.array([42]),                            
        np.array([]),                             
        np.array([[1], [2], [3]]),                
        np.array([[1, 2, 3]]),                    
    ]

    for idx, case in enumerate(test_cases, start=1):
        mean, median, std = random_number(case)

        print(f"\nTest Case: {idx}")
        print("Array: ", case)
        
        if mean is None:
            print("Operation not possible (empty array or non-numeric data)") 
        else:  
            print("Mean:", mean)
            print("Median:", median)
            print("Std Dev:", std)



# Program execution starts here
if __name__ == "__main__":
    run_all_testcases()

