"""
Program Name: Prime Number Generator
Description : This program print all prime numbers between 1 and 100
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def prime_numbers(start,end):
    """
    Print all prime numbers between start and end
    
    Parameters:
        start (int): The starting index
        end (int)  : The ending index

    """

    if start <= 0 or end <= 0:
        print("Prime number is not possible in this range")
        return None

    print("Prime numbers between",start,"and",end,"are: ")

    # Loop through numbers from start to end
    for i in range(start, end+1):

        # Assume the number is prime initially
        is_prime = True

        # Check divisibility from 2 to i - 1
        for divisor in range(2, i):
            if i % divisor == 0:
                is_prime = False
                break

        # If number is prime, print it
        if is_prime:
            print(i, end=" ")
        
    print("\n")


if __name__ == "__main__":
    prime_numbers(1,100)