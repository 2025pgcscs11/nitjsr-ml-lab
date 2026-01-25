"""
Program Name: Operations on a List
Description : This program finds the maximum, minimum, and average of a list of numbers
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def list_operations(input_list):
    """
    Finds the maximum, minimum, average of a list of numbers

    Parameters:
        input_list (list): A list of numbers
    
    Returns:
        max_list (int/float): Maximum number in the List
        min_list (int/float): Minimum number in the List
        avg_list (float)    : Average of ll numbers in the List
    """

    for x in input_list:
        if not isinstance(x,(int,float)):
            return None,None,None

    if not input_list:
        return None,None,None

    max_list = max(input_list)
    min_list = min(input_list)
    avg_list = sum(input_list)/len(input_list)

    return max_list, min_list, avg_list


def run_test_cases():
    """
    Finds the maximum, minimum and average of a list of numbers
    """

    test_cases = [
        [1, 2, 3, 4, 5],                  
        [0, -1, -5, 10, 20],             
        [3.14, 2.71, 0.0, -1.1],         
        [1000000, 500000, 999999],        
        [1],                              
        [],                                
        [0.1, 0.01, 0.001, 0.0001],      
        [-100, -200, -300],               
        [42, 3.14, -7, 0],                
        [1e3, 2e4, -3.5e2],               
        [0, 0.0, -0],
        ['dss',123,-23,23.0,-23.5,[3,5],{2,4}]                      
    ]

    for case in test_cases:
        print("\n\nThe List is:",case)
        maximum,minimum,average = list_operations(case)

        if maximum is None:
            if not case:
                print("The List is empty\n")
            else:
                print("The List is invalid\n")
        else:
            print("Maximum number in the List is :",maximum)
            print("Minimum number in the List is :",minimum)
            print("Average number in the List is :",average)



# Program execution starts here
if __name__ == "__main__":
    run_test_cases()

