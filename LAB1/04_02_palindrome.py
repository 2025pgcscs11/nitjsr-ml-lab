"""
Program Name: Palindrome Checker
Description : This program checks whether a string is palindrome or not
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def is_palindrome(s):
    """
    Checks whether input string is Palindrome or not

    Parmeters:
        input (string): The string input

    Returns:
        boolean: True or False
    """

    if not isinstance(s,str):
        return None
    
    return s == s[::-1]


def run_test_cases():
    """
    Checks whether a string is Plaindrome or not
    """

    test_cases = ["Madam","madam","sir","12321","ma'am",123]

    for case in test_cases:
        res = is_palindrome(case)
        if res is not None:
            if res is True:
                print("The string",case,"is Palindrome\n")
            else:
                print("The string",case,"is not Palindrome\n")
        else:
            print("The input is a invalid string\n")


# Program execution starts here
if __name__ == "__main__":
    run_test_cases()


