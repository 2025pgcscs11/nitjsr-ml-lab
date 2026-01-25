"""
Program Name: Duplicate Remover
Description : This program remove duplicates from a list
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def duplicates_remove(input_list):
    """
    Remove duplicates from the list

    Parameters:
        input_list (list): A list of elements
    
    Returns:
        list: A list without duplicates
    """

    if not input_list:
        return None

    unique_list = []
    for x in input_list:
        flag = False
        for y in unique_list:
            if x == y and type(x) == type(y):
                flag = True
                break
        
        if not flag:
            unique_list.append(x)
            

    return unique_list



def run_test_cases():
    """
    Remove duplicates from every test case list
    """

    test_cases = [
        [1, 2, 2, 3, "a", "a", True],                 
        ["x", "y", "x", 10, 10, 20],                
        [3.14, 3.14, 2.71, "pi", "pi"],               
        [True, False, True, 1, 0, 1],                 
        ["apple", "banana", "apple", "banana"],      
        [1, "1", 1, "1", 2],                          
        [None, None, 0, "None"],                    
        [1, 2, [3, 4], [3, 4], 2],                 
        ["a", 1, "b", 1, "a", 2.0, 2],               
        [True, "True", False, "False", True],         
    ]


    for case in test_cases:
        print("\n\nThe List is:",case)
        if not case:
            print("The list is empty\n")
        else:    
            res = duplicates_remove(case)
            print("After removing Duplicates:",res)



# Program execution starts here
if __name__ == "__main__":
    run_test_cases()

