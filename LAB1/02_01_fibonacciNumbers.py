"""
Program Name: Fibonacci Number Generator
Description : This program print first 10 Fibonacci numbers using loop
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def fibonacci_numbers(n):
    """
    Print the first n Fibonacci numbers using a loop
    
    Parameters:
        n (int): The number of Fibonacci numbers to print

    """
    
    # first two numbers in Fibonacci series
    a = 0
    b = 1

    print("\nFirst 10 Fibonacci Numbers are: ",a," ",b," ",end="")
    
    for i in range(n - 2):
        # current number = sum of previous two numbers
        c = a + b
        a = b
        b = c
        print(c," ",end="")

    print("\n")

# Program execution starts here    
if __name__ == "__main__":
    fibonacci_numbers(10)