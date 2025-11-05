
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dataset
courses = pd.read_csv("fisk_courses_tagged.csv")
students = pd.read_csv("mock_student_profiles.csv")

# Cleaning / formatting
courses["Course_level"] = (
    courses["Course_level"].astype(str).str.extract(r"(\d{3})").fillna("100").astype(int)
)
courses["Credit Hours"] = pd.to_numeric(courses["Credit Hours"], errors="coerce").fillna(3)

# the filter function
def get_level_range(classification):
    """Map student classification to course level range"""
    ranges = {
        "Freshman": (0, 199),
        "Sophomore": (200, 299),
        "Junior": (300, 399),
        "Senior": (400, 499)
    }
    return ranges.get(classification, (100, 499))

def check_prereqs(prereq, taken_list):
    """Return True if prereqs are satisfied or none."""
    if pd.isna(prereq) or prereq.strip().lower() in ["none", ""]:
        return True
    return any(t in prereq for t in taken_list)

# dataset merge and TD-IDF initializing
courses["text"] = (
    courses["Course code"].fillna("") + " " +
    courses["Course Name"].fillna("") + " " +
    courses["Requirement_Type"].fillna("") + " " +
    courses["Major_Applicable"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(courses["text"])

# RECOMMENDER Function
def recommend_for_student(student_row, top_n=5):
    major = student_row["Major"]
    classification = student_row["Year"].title()
    honors = student_row["is Honors or Regular"]
    taken_courses = [c.strip().upper() for c in student_row["Courses_Taken"].split(",")]

    low, high = get_level_range(classification)
   # Credit hour limits
    if honors:
        credit_limit = 21 
    else:
        credit_limit = 18

    total_credits = 0

    #Filter the courses
    filtered = courses[
        (courses["Course_level"].between(low, high)) &
        (courses["Major_Applicable"].str.contains(major, case=False, na=False) |
         courses["Requirement_Type"].str.contains("General Elective", case=False, na=False))
    ].copy()

    # Remove taken courses
    filtered = filtered[~filtered["Course code"].isin(taken_courses)]

    # Check prerequisites
    filtered["Eligible"] = filtered["Prerequisite"].apply(lambda p: check_prereqs(p, taken_courses))
    eligible = filtered[filtered["Eligible"]]

    #cosine similarity
    indices = courses[courses["Course code"].isin(taken_courses)].index.tolist()
    if not indices:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])
    
    student_vector = tfidf_matrix[indices].mean(axis=0)
    student_vector = np.asarray(student_vector).reshape(1, -1)
    tfidf_arr = tfidf_matrix.toarray()
    sim_scores = cosine_similarity(student_vector, tfidf_arr).flatten()

    eligible = eligible.copy()
    eligible["similarity"] = sim_scores[eligible.index]

    # filter by credit limit
    eligible_sorted = eligible.sort_values(by="similarity", ascending=False)
    recommendations = []
    for _, row in eligible_sorted.iterrows():
        if total_credits + row["Credit Hours"] <= credit_limit:
            recommendations.append(row)
            total_credits += row["Credit Hours"]
        if total_credits >= credit_limit:
            break

    if not recommendations:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    recs_df = pd.DataFrame(recommendations)[["Course code", "Course Name", "similarity"]]
    recs_df["Student_ID"] = student_row["Student_ID"]
    return recs_df.head(top_n)

# generating the recommendations
all_recs = pd.concat([recommend_for_student(row) for _, row in students.iterrows()], ignore_index=True)

# results/check
all_recs.to_csv("baseline_recommendations.csv", index=False)
#file name: baseline_recommendations.csv
