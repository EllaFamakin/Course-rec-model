# import pandas as pd
# import random

# # Load course data
# courses = pd.read_csv("fisk_courses_tagged.csv")

# # Define majors
# majors = ["Computer Science", "Data Science", "Mathematics"]
# School_program = ["Honors", "Regular"]
# # Generate 300 mock students
# students = []
# for i in range(300):
#     major = random.choice(majors)
#    # honors = random.choice(School_program)
    
#     # pick courses applicable to this major
#     eligible_courses = courses[courses["Major_Applicable"].str.contains(major, na=False)]
#     taken_courses = eligible_courses.sample(n=random.randint(3, 8))  # courses already taken
    
#     students.append({
#         "Student_ID": f"F{i+1:05d}",
#         "Major": major,
#         "Year": random.choice(["Freshman", "Sophomore", "Junior", "Senior"]),
#         "Courses_Taken": ", ".join(taken_courses["Course code"].tolist()),
#         "is Honors or Regular": random.choice(["Honors", "Regular"])
#     })

# mock_students = pd.DataFrame(students)
# mock_students.to_csv("mock_student_profiles.csv", index=False)
# #Filename: mock_student_profiles.csv



# 

import pandas as pd
import random
import ast

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("fisk_courses_tagged.csv")

# Load prereq & coreq dictionaries
prereq_df = pd.read_csv("prerequisite_dictionary.csv")
prereq_dict = {
    row["Course"]: ast.literal_eval(row["Parsed_Prereqs"])
    for _, row in prereq_df.iterrows()
}

coreq_df = pd.read_csv("corequisites_dictionary.csv")
coreq_dict = {
    row["Course"]: ast.literal_eval(row["Parsed_Coreqs"])
    for _, row in coreq_df.iterrows()
}

# ---------------------------
# HELPERS
# ---------------------------

def get_major_courses(major):
    return df[df["Major_Applicable"].str.contains(major, case=False, na=False)]["Course code"].tolist()

general_electives = df[df["Requirement_Type"].str.contains("General Elective", case=False, na=False)]["Course code"].tolist()

def add_prereqs(course, taken_list):
    """Ensure that if C requires A and B, those get added."""
    if course not in prereq_dict:
        return taken_list

    for group in prereq_dict[course]:  # AND-of-OR groups
        # OR case
        if len(group) > 1:
            selected = random.choice(group)
            if selected not in taken_list:
                taken_list.append(selected)
        else:  # single course required
            if group[0] not in taken_list:
                taken_list.append(group[0])
    return taken_list

def add_coreqs(course, taken_list):
    """If a course has a co-req bundle, pick the valid option(s)."""
    if course not in coreq_dict:
        return taken_list

    for group in coreq_dict[course]:
        if len(group) == 1:
            if group[0] not in taken_list:
                taken_list.append(group[0])
        else:
            # pick one option
            chosen = random.choice(group)
            if chosen not in taken_list:
                taken_list.append(chosen)
    return taken_list


def realistic_course_count(year):
    """Approximate number of completed courses per grade level."""
    mapping = {
        "Freshman": (0, 8),      # 0–8 courses
        "Sophomore": (8, 16),    # 8–16 courses
        "Junior": (16, 24),      # 16–24 courses
        "Senior": (24, 32)       # 24–32 courses
    }
    low, high = mapping[year]
    return random.randint(low, high)

# ---------------------------
# GENERATE PROFILES
# ---------------------------
profiles = []

majors = ["Computer Science", "Data Science", "Mathematics"]
years = ["Freshman", "Sophomore", "Junior", "Senior"]

for id in range(1, 151):

    major = random.choice(majors)
    year = random.choice(years)
    honors = random.choice([True, False])

    major_courses = get_major_courses(major)

    # Number of courses the student has "completed"
    num_courses = realistic_course_count(year)

    taken = []

    # Pick courses
    while len(taken) < num_courses:
        # 70% chance pick major requirement, 30% pick elective
        if random.random() < 0.7:
            course = random.choice(major_courses)
        else:
            course = random.choice(general_electives)

        # Add course
        taken.append(course)

        # Ensure prereqs and coreqs are also added
        taken = add_prereqs(course, taken)
        taken = add_coreqs(course, taken)

        # Remove duplicates
        taken = list(set(taken))

    profiles.append({
        "Student_ID": f"A{id+1:05d}",
        "Major": major,
        "Year": year,
        "is Honors or Regular": honors,
        "Courses_Taken": ", ".join(taken)
    })

profiles_df = pd.DataFrame(profiles)
profiles_df.to_csv("mock_student_profiles.csv", index=False)

print("Mock student profiles generated successfully!")
