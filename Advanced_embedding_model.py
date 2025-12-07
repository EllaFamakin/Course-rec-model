
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re 

# dataset
courses = pd.read_csv("fisk_courses_tagged.csv")
students = pd.read_csv("mock_student_profiles.csv")

# Cleaning / formatting
courses["Course_level"] = (
    courses["Course_level"].astype(str).str.extract(r"(\d{3})").fillna("100").astype(int)
)
courses["Credit Hours"] = pd.to_numeric(courses["Credit Hours"], errors="coerce").fillna(3)

# If Major_Applicable is saved like a list then convert it to a clean string
courses["Major_Applicable"] = courses["Major_Applicable"].astype(str).str.replace(r"[\[\]']", "", regex=True)

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
    # Look for actual course codes inside prerequisite string
    for code in taken_list:
        pattern = r"\b" + re.escape(code) + r"\b"
        if re.search(pattern, prereq):
            return True

    return False


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

   # Credit hour limits
    if honors:
        credit_limit = 21 
    else:
        credit_limit = 18

    total_credits = 0

    low, high = get_level_range(classification)

    #Filter the courses
    filtered = courses[
        (courses["Course_level"].between(low, high)) &
        (   courses["Major_Applicable"].str.contains(major, case=False, na=False) |
            courses["Requirement_Type"].str.contains("General Elective", case=False, na=False))
    ].copy()

    # Remove taken courses
    filtered = filtered[~filtered["Course code"].isin(taken_courses)]

    # Check prerequisites
    filtered["Eligible"] = filtered["Prerequisite"].apply(lambda p: check_prereqs(p, taken_courses))
    eligible = filtered[filtered["Eligible"]].copy()

    # Separate major courses from electives
    elective_pool = eligible[eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()
    major_pool = eligible[~eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()

# If no major courses are found, fall back to all eligible
    if major_pool.empty:
        major_pool = eligible.copy()


    #cosine similarity
    indices = courses[courses["Course code"].isin(taken_courses)].index.tolist()
    if not indices:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])
    
    student_vector = tfidf_matrix[indices].mean(axis=0)
    student_vector = np.asarray(student_vector).reshape(1, -1)

    # # Only compute similarity with eligible courses
    # eligible_vectors = tfidf_matrix[eligible.index].toarray()
    # similarities = cosine_similarity(student_vector, eligible_vectors).flatten()
    # eligible["similarity"] = similarities

    # Compute similarity ONLY for major courses
    major_vectors = tfidf_matrix[major_pool.index].toarray()
    major_sim = cosine_similarity(student_vector, major_vectors).flatten()
    major_pool["similarity"] = major_sim

    # Sort major courses by similarity
    major_sorted = major_pool.sort_values(by="similarity", ascending=False)

    # Compute similarity & weighted score for electives
    if not elective_pool.empty:
        elective_vectors = tfidf_matrix[elective_pool.index].toarray()
        elective_sim = cosine_similarity(student_vector, elective_vectors).flatten()
        elective_pool["similarity"] = elective_sim

        elective_pool["weighted"] = (
            0.7 * elective_pool["similarity"] +
            0.3 * np.random.rand(len(elective_pool))
        )
    elective_pool = elective_pool.sort_values("weighted", ascending=False)
    
    recommendations = []
    total_credits = 0

    # Add major courses first (sorted by similarity)
    for _, row in major_sorted.iterrows():
        ch = int(row["Credit Hours"])
        if total_credits + ch <= credit_limit:
            recommendations.append(row)
            total_credits += ch
        if total_credits >= credit_limit:
            break
    
    # Remaining credits after major courses - Add electives
    remaining_credits = credit_limit - total_credits

    # Randomly choose 1–2 electives
    elective_pool = elective_pool.sample(frac=1, random_state=42)  # shuffle
    num_electives = min(len(elective_pool), np.random.randint(1, 3))

    elective_recs = []
    credits_used = 0

    for _, row in elective_pool.iterrows():
        ch = int(row["Credit Hours"])
        if credits_used + ch <= remaining_credits:
            elective_recs.append(row)
            credits_used += ch
            if len(elective_recs) >= num_electives:
                break

    # Add electives to final recommendations
    recommendations.extend(elective_recs)

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
