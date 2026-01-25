"""
Program Name: Basic Arithmatic Operations
Description : This program implements four basic operations respectively sum, difference, product, quotient
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def basic_operations(input1,input2):
    """
    Perform basic arithmetic operations on two numbers

    Operations performed:
    - Sum
    - Difference
    - Product
    - Quotient (with division-by-zero handling)

    Parameters:
        input1 (int/float): The first input number
        input2 (int/float): The second input number

    Returns:
        dict: A dictionary containing results of all operations
    """
    addition = input1 + input2 
    subtraction = input1 - input2
    multiplication = input1 * input2

    # Calculate quotient
    if input2 != 0:
        division = input1 / input2
    else:
        division = None  # Division by zero is not allowed


    return {
        "sum" : addition,
        "difference" : subtraction,
        "product" : multiplication,
        "quotient" :division 
    }


def run_test_cases():
    """
    Test the basic_operations function using
    trivial, edge, and typical cases.
    """

    test_cases = [
        (10, 5),     # Typical case
        (0, 5),      # Trivial case
        (5, 0),      # Edge case (division by zero)
        (-4, 2),     # Negative number case
        (3.5, 1.5)   # Floating-point numbers
    ]

    for first_number, second_number in test_cases:
        print("\nInput Numbers:", first_number, "and", second_number)

        results = basic_operations(first_number, second_number)

        print("Sum:", results["sum"])
        print("Difference:", results["difference"])
        print("Product:", results["product"])

        if results["quotient"] is not None:
            print("Quotient:", results["quotient"])
        else:
            print("Quotient: Division by zero is not allowed")


# Program execution starts here
if __name__ == "__main__":
    run_test_cases()

