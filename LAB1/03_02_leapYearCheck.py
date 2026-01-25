"""
Program Name: Leap Year Checker
Description : This program checks whether a year is leap year or not
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def is_leap_year(year):
    """
    Checks whether year is a leap year or not

    Parameters:
        year (int): The Year value
    
    Returns:
        boolean: True or False
    """
    if not isinstance(year,int) or year < 0:
        return None

    if year % 400 == 0 or (year % 4 == 0 and year % 100 !=0):
        return True
    else:
        return False

def run_test_cases():
    """
    Test a year is Leap year or not
    """

    test_cases = [2026,0,25,2016,300,1600,2999,25000,-2300,34.45]

    for case in test_cases:
        res = is_leap_year(case)
        if res is not None:
            if res is True:
                print("The year",case,"is a Leap Year\n")
            else:
                print("The year",case,"is not a Leap Year\n")
        else:
            print("The year",case,"is a Invalid Year\n")

# Program execution starts here
if __name__ == "__main__":
    run_test_cases()
