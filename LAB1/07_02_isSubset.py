"""
Program Name: Subset Checker
Description : This program check if one set is subset of another
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def is_subset(set_a,set_b):
    """
    Check if set_b is subset of set_a

    Parameters:
        set_a (set): The large set
        set_b (set): The small set

    Returns:
        boolean: True or False
    """

    # is set_b is subset of set_a
    return set_b.issubset(set_a)


def run_test_cases():
    """
    Run is_subset function with multiple test cases
    """

    test_cases = [
            ({1, 2, 3}, {3, 2, 5}),                      
            ({1, 2, 3}, {1, 2}),                          
            ({1.5, 2.5}, {2.5, 3.5}),                    
            ({"a", "b", "c"}, {"c"}),                
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
        res = is_subset(set_a,set_b)
        if res:
            print("second set is subset of First set")
        else:
            print("second set is not subset of First set")



# Program execution starts here
if __name__ == "__main__":
    run_test_cases()
   