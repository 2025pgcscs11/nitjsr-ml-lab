"""
Program Name: Sign Checker
Description : This program checks whether a number is postive ,negtive or zero 
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def signOfNumber(input_number):
    """
    Check whether input_number is positive, negative, or zero

    Parmeters:
        input_number (int/float): The number to check its sign

    Returns:
        int: 1 for positive, -1 for negative and 0 for zero
    """

    if input_number > 0:
        return 1
    elif input_number < 0:
        return -1
    else:
        return 0

def run_test_cases():
    """
    Test Sign of different numbers
    """

    test_cases = [2,-2,0,12,+23,-30.89,000,23.56,-1233,0.0]

    for case in test_cases:
        if signOfNumber(case) == 1:
            print("\nInput Number: ",case,"(Positive)")
        elif signOfNumber(case) == -1:
            print("\nInput Number:",case,"(Negative)") 
        else:
            print("\nInput Number: ",case,"(Zero)")


# Program execution starts here    
if __name__ == "__main__":
    run_test_cases()