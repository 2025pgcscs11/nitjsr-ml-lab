"""
Program Name: Average score and Highest Score Finder
Description : This program find highest score and calculate the average score of students
Author      : Sujit Halder
Roll No     : 2025PGCSCS11
Course      : Machine Learning Laboratory (CS4205)
Date        : 22-01-2026
Language    : Python
"""

def average_score(data):
    """
    Calculate the averge score of all students

    Parameters:
        data (dict): A database that stores student name and score

    Returns:
        avg_score (int)  : Average score of all students
    """
    if not data:
        return None
    
    total_score = 0
    num_student = 0

    for score in data.values():
        if isinstance(score,(int,float)) and  0 <= score <=100:
            total_score += score
            num_student += 1
        else:
            return None
    
    avg_score = total_score / num_student
    return avg_score


def highest_score(data):
    """
    Find highest score from a student's database stored in a dictionary

    Parameters:
        data (dict): A database that stores student name and score

    Returns:
        name (list): Names of the students with highest score
        score (int)  : Score of the Hhighest scoring student
    """
    if not data:
        return None,None
    
    top_score = -1
    top_student = []

    for name, score in data.items():
        if isinstance(name,str) and name!="" and isinstance(score,(int,float)) and 0 <= score <=100:
            if score > top_score:
                top_score = score
        else:
            return None,None
    
    for name, score in data.items():
        if score == top_score:
            top_student.append(name)

    return top_student,top_score


def run_test_cases():
    """"
    Run test cases with different student database and print average score and top student(s) name with highest score
    """

    test_cases = [
        {
            "Alice": 85,
            "Bob": 92,
            "Charlie": -5,    
            "David": 105,   
            "Eve": 78
        },
        {
            "Frank": 88,
            "Grace": 95,
            "Heidi": 100,
            "Ivan": 0,         
            "Judy": 100
        },
        {
            "Ken": 67,
            "Leo": "90",      
            "Mia": None,      
            "Nina": 120,      
            "Oscar": 55
        },
        {
            "": 60,          
            "Paul": 83,
            "Quinn": 97,
            "Rita": 102,      
            "Steve": -12     
        }
    ]
    for case in test_cases:
        print("\n\nThe Student Data is:",case)
        if case:
            avg_score = average_score(case)
            if avg_score is not None:
                print("Average Score of the Student is:",avg_score)
                top_student,top_score = highest_score(case)
                print("Highest score is:",top_score)
                print("Top Student(s) are:",",".join(top_student))
            else:
                print("Student Data is invalid")
        else:
            print("Student Data is empty")



# Program execution starts here
if __name__ == "__main__":
    run_test_cases()


