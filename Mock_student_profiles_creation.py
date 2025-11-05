import pandas as pd
import random

# Load course data
courses = pd.read_csv("fisk_courses_tagged.csv")

# Define majors
majors = ["Computer Science", "Data Science", "Math"]
School_program = ["Honors", "Regular"]
# Generate 300 mock students
students = []
for i in range(300):
    major = random.choice(majors)
   # honors = random.choice(School_program)
    
    # pick courses applicable to this major
    eligible_courses = courses[courses["Major_Applicable"].str.contains(major, na=False)]
    taken_courses = eligible_courses.sample(n=random.randint(3, 8))  # courses already taken
    
    students.append({
        "Student_ID": f"F{i+1:05d}",
        "Major": major,
        "Year": random.choice(["Freshman", "Sophomore", "Junior", "Senior"]),
        "Courses_Taken": ", ".join(taken_courses["Course code"].tolist()),
        "is Honors or Regular": random.choice(["Honors", "Regular"])
    })

mock_students = pd.DataFrame(students)
mock_students.to_csv("mock_student_profiles.csv", index=False)
#Filename: mock_student_profiles.csv
