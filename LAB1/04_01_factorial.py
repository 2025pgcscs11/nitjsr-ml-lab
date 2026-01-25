"""
Program Name: Factorial Calculator
Description : This program calculates factorial of a number
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def factorial(num):
    """
    Calculate factorial of num integer value

    Parameters:
        num (int): The input number

    Returns:
        int: The calculated factorial of num

    """

    if not isinstance(num,int) or num < 0:
        return None

    factorial = 1
    while num > 0:
        factorial *= num
        num -= 1

    return factorial


def run_test_cases():
    """
    Calculate factorial of a number
    """

    test_cases = [5,-2,0,6,2.5]

    for case in test_cases:
        res = factorial(case)
        if res is not None:
            print("The factorial of number",case,"is:",res,"\n")
        else:
            print("The input number is invlid\n")

            
# Program execution starts here
if __name__ == "__main__":
    run_test_cases()
