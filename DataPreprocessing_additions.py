import pandas as pd

# Load dataset
df = pd.read_csv("fisk_courses_preprocessing  (version 1).xlsb.csv")

# Dropping columns with too little information
df = df.drop(columns=["Offered", "Description"], errors="ignore")

# Define mapping (based on the balance sheets)
major_course_map = {
    "Computer Science": {
        "Core": ["CORE 100", "CORE 150", "CORE 160", "CORE 120", "CORE 201", "CORE 260", "CORE 360"],
        "Cognates": ["MATH 120", "MATH 125", "MATH 130", "MATH 240", "NSCI 360"],
        "Major Requirements": ["CSCI 110", "CSCI 120", "CSCI 120L", "CSCI 210", 
                               "CSCI 230", "CSCI 241", "CSCI 261", "CSCI 282", 
                               "CSCI 291", "CSCI 310", "CSCI 312", "CSCI 411",
                                "CSCI 412"],
    
    },
    "Data Science": {
        "Core": ["CORE 100", "CORE 150", "CORE 160", "CORE 120", "CORE 201", "CORE 260", "CORE 360"],
        "Math Requirements": ["MATH 120", "MATH 125", "MATH 130", "MATH 240"],
        "Statistics": ["NSCI 360", "BAD 260", "HSS 290", "MATH 390B"],
        "CS Requirements": ["CSCI 110", "CSCI 120", "CSCI 241", "CSCI 312", 
                            "CSCI 380", "CSCI 210", "CSCI 310", "CSCI 411", 
                            "CSCI 412"],
    },
    "Mathematics":{
        "Core": ["CORE 100", "CORE 150", "CORE 160", "CORE 120", "CORE 201", "CORE 260", "CORE 360"],
        "Cognates": ["CSCI 110", "CSCI 120", "NSCI 360", "PHYS 130"],
        "Major Requirements": ["MATH 120", "MATH 130", "MATH 210", "MATH 220",
                                "MATH 240", "MATH 270", "MATH 320", "MATH 353",
                                "MATH 395"]
    }
}
df["Course code"] = df["Course code"].astype(str)

# Helper Function to tag applicable major(s) and requirement type of a course
def find_majors_and_types(Course_code):
    applicable = []
    requirement_type = []

    for major, categories in major_course_map.items():
        for req_type, course_list in categories.items():

            for c in course_list:
                subj, num = c.split()
                if Course_code.startswith(subj) and num in Course_code:
                    applicable.append(major)
                    requirement_type.append(req_type)

    if not applicable:
        return ["OTHER"], ["General Elective"]

    return applicable, requirement_type

# Apply to DataFrame
df["Major_Applicable"], df["Requirement_Type"] = zip(*df["Course code"].apply(find_majors_and_types))

# Save processed dataset
df.to_csv("fisk_courses_tagged.csv", index=False)

#File name: Fisk_courses_tagged.csv
