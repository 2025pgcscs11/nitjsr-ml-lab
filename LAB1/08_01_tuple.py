"""
Program Name: Tuple Dsiplayer and Calculator of frequency of its element  
Description : This program unpacks and displays a tuple elements and calaculate frequency of each elements in it 
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def unpack_tuple(data):
    """"
    Unpack and display a tuple's elements

    Parameters:
        data (tuple):  The input tuple

    """

    print("After unpacking, the elements of the tuple are: ",end ="")
    for element in data:
        print(element,end =" ")
    print("")


def tuple_element_frequency(data):
    """"
    Find the frequency of each elements in the tuple

    Parameters:
        data (tuple): The input tuple

    Returns:
        dict : A dictionary contains each element with its frequency in the tuple
        """

    # Dictionary to store frequency
    frequency = {}

    # Count frequency
    for element in data:
        frequency[element] = frequency.get(element, 0) + 1
    
    return frequency


def run_test_cases():
    """
    Run unpack_tuple and tuple_element_frequency function on different tuples  
    """

    test_cases = [
        (1, 2, 3),                         
        (1, 1, 2, 2, 3),                  
        ("a", "b", "a", "c"),             
        (1, "a", 2.5, "b"),               
        (True, False, True),               
        (2.5, 3.5, 2.5, 4.5),              
        ((1, 2), (1, 2), (3, 4)),       
        (),                               
        (10,),                           
        ("apple", "banana", "apple")       
    ]

    for case in test_cases:
        print("\n\nThe tuple is:",case)
        unpack_tuple(case)
        print("Frequency of each elements in the tuple is:")
        res = tuple_element_frequency(case)
        for key,value in res.items():
            print("Element:",key, "     Frequency:", value)




# Program execution starts here
if __name__ == "__main__":
    run_test_cases()