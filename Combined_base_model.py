import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

# -----------------------------
# LOAD DATA
# -----------------------------
courses = pd.read_csv("fisk_courses_tagged.csv")
students = pd.read_csv("mock_student_profiles.csv")

# Load dictionaries
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

# -----------------------------
# CLEANING
# -----------------------------
courses["Course_level"] = (
    courses["Course_level"].astype(str).str.extract(r"(\d{3})").fillna("100").astype(int)
)

courses["Credit Hours"] = pd.to_numeric(
    courses["Credit Hours"], errors="coerce"
).fillna(3)

courses["Major_Applicable"] = (
    courses["Major_Applicable"].astype(str)
    .str.replace(r"[\[\]']", "", regex=True)
)

# Ensure no NaN text
courses["text"] = (
    courses["Course code"].fillna("") + " " +
    courses["Course Name"].fillna("") + " " +
    courses["Requirement_Type"].fillna("") + " " +
    courses["Major_Applicable"].fillna("")
).astype(str)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(courses["text"])
import joblib

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save TF-IDF matrix (as dense array)
np.save("tfidf_matrix.npy", tfidf_matrix.toarray())


# -----------------------------
# HELPERS
# -----------------------------
def get_level_range(classification):
    ranges = {
        "Freshman": (0, 199),
        "Sophomore": (200, 299),
        "Junior": (300, 399),
        "Senior": (400, 499),
    }
    return ranges.get(classification, (100, 499))

def prereqs_satisfied(course, taken_list):
    """Check AND-of-OR prerequisite groups."""
    if course not in prereq_dict:
        return True

    groups = prereq_dict[course]

    for group in groups:
        # group = ["CSCI 120"] or ["MATH 130", "MATH 140"]
        if not any(g in taken_list for g in group):
            return False

    return True

def choose_coreqs(course):
    """Returns list of required co-reqs. Picks one from OR groups."""
    if course not in coreq_dict:
        return []

    chosen = []
    for group in coreq_dict[course]:
        if len(group) == 1:
            chosen.append(group[0])
        else:
            chosen.append(np.random.choice(group))

    return chosen

# -----------------------------
# MAIN RECOMMENDER
def recommend_for_student(row, top_n=5):
    major = row["Major"]
    classification = row["Year"].title()
    honors = row["is Honors or Regular"]

    # Safe split
    if isinstance(row["Courses_Taken"], str):
        taken_courses = [x.strip().upper() for x in row["Courses_Taken"].split(",")]
    else:
        taken_courses = []

    # Credit limits
    credit_limit = 21 if honors else 18
    low, high = get_level_range(classification)

    # -------- 1. FILTER ELIGIBLE COURSES --------
    eligible = courses[
        (courses["Course_level"].between(low, high)) &
        (
            courses["Major_Applicable"].str.contains(major, case=False, na=False) |
            courses["Requirement_Type"].str.contains("General Elective", case=False, na=False)
        )
    ].copy()

    # Remove taken
    eligible = eligible[~eligible["Course code"].isin(taken_courses)]

    # Apply prereqs (AND-of-OR logic)
    eligible["Eligible"] = eligible["Course code"].apply(
        lambda c: prereqs_satisfied(c, taken_courses)
    )
    eligible = eligible[eligible["Eligible"]].copy()

    # Split into major & elective pools
    elective_pool = eligible[eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()
    major_pool = eligible[~eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()

    # If nothing in either pool → return empty
    if major_pool.empty and elective_pool.empty:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    # If no major courses, fall back to electives as main pool
    if major_pool.empty:
        print(f"WARNING: No major courses available for {row['Student_ID']}. Using electives instead.")
        major_pool = elective_pool.copy()

    # -------- 2. BUILD STUDENT VECTOR --------
    taken_idx = courses[courses["Course code"].isin(taken_courses)].index.tolist()
    if not taken_idx:
        # No taken courses found in the catalog → cannot compute similarity
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    student_vec = tfidf_matrix[taken_idx].mean(axis=0)
    student_vec = np.asarray(student_vec).reshape(1, -1)

    # -------- 3. SIMILARITY FOR MAJOR COURSES --------
    major_sims = cosine_similarity(student_vec, tfidf_matrix[major_pool.index].toarray()).flatten()
    major_pool = major_pool.copy()
    major_pool["similarity"] = major_sims

    major_sorted = major_pool.sort_values("similarity", ascending=False)

    # -------- 4. SIMILARITY + WEIGHTING FOR ELECTIVES --------
    if not elective_pool.empty:
        elective_pool = elective_pool.copy()
        elective_sims = cosine_similarity(student_vec, tfidf_matrix[elective_pool.index].toarray()).flatten()
        elective_pool["similarity"] = elective_sims

        # Weighted: 70% similarity + 30% randomness for diversity
        elective_pool["weighted"] = (
            0.7 * elective_pool["similarity"] +
            0.3 * np.random.random(len(elective_pool))
        )
        elective_pool = elective_pool.sort_values("weighted", ascending=False)

    # -------- 5. SELECTION WITH CREDIT LIMIT + CO-REQ BUNDLES --------
    recommendations = []
    total_credits = 0

    def try_add(primary_row):
        """
        Add a primary course (which has similarity) plus its co-reqs (given same similarity),
        if the whole bundle fits inside the remaining credit limit.
        """
        nonlocal total_credits, recommendations

        primary_code = primary_row["Course code"]
        primary_credits = float(primary_row["Credit Hours"])
        primary_sim = float(primary_row.get("similarity", 0.0))

        # Get co-reqs
        coreq_codes = choose_coreqs(primary_code)
        coreq_df = courses[courses["Course code"].isin(coreq_codes)].copy()

        bundle_credit = primary_credits + coreq_df["Credit Hours"].sum()

        if total_credits + bundle_credit > credit_limit:
            return False

        # 1️. Add primary course with its similarity
        rec_primary = primary_row.to_dict()
        rec_primary["similarity"] = primary_sim
        recommendations.append(rec_primary)

        # 2. Add each co-req course, inheriting the primary similarity
        for _, c_row in coreq_df.iterrows():
            rec_coreq = c_row.to_dict()
            rec_coreq["similarity"] = primary_sim
            recommendations.append(rec_coreq)

        total_credits += bundle_credit
        return True

    # 5a. Add major courses first
    for _, r_major in major_sorted.iterrows():
        if total_credits >= credit_limit:
            break
        try_add(r_major)

    # 5b. Then add electives (if any) with remaining credit space
    if not elective_pool.empty and total_credits < credit_limit:
        for _, r_el in elective_pool.iterrows():
            if total_credits >= credit_limit:
                break
            try_add(r_el)

    # -------- 6. BUILD FINAL DF --------
    if not recommendations:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    final_df = pd.DataFrame(recommendations)

    # Drop duplicate courses (e.g., if co-req and main reused)
    final_df = final_df.drop_duplicates(subset=["Course code"])

    # Ensure similarity column exists & has no NaN
    if "similarity" not in final_df.columns:
        final_df["similarity"] = 0.0
    else:
        final_df["similarity"] = final_df["similarity"].fillna(0.0)

    final_df = final_df[["Course code", "Course Name", "similarity"]]
    final_df["Student_ID"] = row["Student_ID"]

    return final_df.head(top_n)

# ----------------------------- # RUN MODEL # -----------------------------
all_recs = pd.concat( [recommend_for_student(row) for _, row in students.iterrows()], ignore_index=True ) 
all_recs.to_csv("baseline_recommendations.csv", index=False) 
print("Baseline model complete!")

