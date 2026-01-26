"""
Program Name: Basic Set Operations 
Description : This program find the union, intersection, and difference of two sets
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def basic_set_operations(set_a,set_b):
    """
    Find union,intersection and difference of two sets


    Parameters:
        set_a (set): The first input set
        set_b (set): The second input set

    Returns:
        union (set)        : Union of two sets
        intersection (set) : Intersection of two sets
        difference         : Difference of two sets
    """

    union = set_a | set_b
    intersection = set_a & set_b
    difference = set_a - set_b

    return union, intersection, difference


def run_test_cases():
    """
    Run basic_set_operations function on different test case 
    """

    test_cases = [
            ({1, 2, 3}, {3, 4, 5}),                      
            ({1, 2, 3}, {1, 2}),                          
            ({1.5, 2.5}, {2.5, 3.5}),                    
            ({"a", "b", "c"}, {"c", "d"}),                
            ({1, "a", 2.5}, {"a", 3, 4.5}),               
            ({True, False}, {1, 0}),                    
            (set(), {1, 2, 3}),                           
            ({(1, 2), (3, 4)}, {(3, 4), (5, 6)}),         
            ({-1, -2, -3}, {-3, -4, -5}),                  
            ({1, 2, 3}, {4, 5, 6})   
        ]

    for case in test_cases:
        set_a, set_b = case
        print("\n\nTwo set are",set_a,"and",set_b)
        union, intersection, difference = basic_set_operations(set_a,set_b)
        print("Union of two sets:",union)
        print("Intersection of two sets:",intersection)
        print("Difference of two set (set_a - set_b):",difference)



# Program execution starts here
if __name__ == "__main__":
    run_test_cases()
