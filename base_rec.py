import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# 1. LOAD & CLEAN DATA
# ============================
courses = pd.read_csv("fisk_courses_tagged.csv")
students = pd.read_csv("mock_student_profiles.csv")

# Clean NaN issues early
courses = courses.fillna("")
students = students.fillna("")

# Fix types
courses["Course_level"] = (
    courses["Course_level"].astype(str)
    .str.extract(r"(\d{3})")[0]
    .fillna("100").astype(int)
)

courses["Credit Hours"] = pd.to_numeric(courses["Credit Hours"], errors="coerce").fillna(3)

# Clean Major_Applicable string
courses["Major_Applicable"] = (
    courses["Major_Applicable"]
    .astype(str)
    .str.replace(r"[\[\]']", "", regex=True)
)


# ============================
# 2. HELPERS
# ============================

def get_level_range(classification):
    mapping = {
        "Freshman": (0, 199),
        "Sophomore": (200, 299),
        "Junior": (300, 399),
        "Senior": (400, 499)
    }
    return mapping.get(classification, (100, 499))


def check_prereqs(prereq_string, taken_list):
    """Return True if prereqs are satisfied OR none exist."""
    if not prereq_string or prereq_string.strip().lower() in ["none", ""]:
        return True

    # Check each taken course for exact match inside prereq text
    for c in taken_list:
        pattern = r"\b" + re.escape(c) + r"\b"
        if re.search(pattern, prereq_string):
            return True

    return False


# ============================
# 3. TF-IDF TEXT FEATURES
# ============================

courses["text"] = (
    courses["Course code"] + " " +
    courses["Course Name"] + " " +
    courses["Requirement_Type"] + " " +
    courses["Major_Applicable"]
).fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(courses["text"])


# ============================
# 4. RECOMMENDER FUNCTION
# ============================

def recommend_for_student(student_row, top_n=5):

    # ---- normalize student row safely ----
    major = str(student_row.get("Major", "")).strip()
    classification = str(student_row.get("Year", "")).title()
    honors = str(student_row.get("is Honors or Regular", "")).lower() in ["true", "honors", "yes"]

    raw_taken = student_row.get("Courses_Taken", "")
    raw_taken = "" if pd.isna(raw_taken) else str(raw_taken)

    taken_courses = [
        c.strip().upper()
        for c in raw_taken.split(",")
        if c.strip() != "" and c.strip().lower() != "nan"
    ]

    # ---- credit limits ----
    credit_limit = 21 if honors else 18

    # ---- level filtering ----
    low, high = get_level_range(classification)

    filtered = courses[
        (courses["Course_level"].between(low, high)) &
        (
            courses["Major_Applicable"].str.contains(major, case=False, na=False) |
            courses["Requirement_Type"].str.contains("General Elective", case=False, na=False)
        )
    ].copy()

    # ---- remove already taken ----
    filtered = filtered[~filtered["Course code"].isin(taken_courses)].copy()

    # ---- prerequisite check ----
    filtered["Eligible"] = filtered["Prerequisite"].apply(
        lambda p: check_prereqs(p, taken_courses)
    )
    eligible = filtered[filtered["Eligible"]].copy()

    # ---- split eligible into pools ----
    elective_pool = eligible[eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()
    major_pool = eligible[~eligible["Requirement_Type"].str.contains("General Elective", case=False, na=False)].copy()

    if major_pool.empty:
        major_pool = eligible.copy()

    # ---- TF-IDF similarity only for courses student has taken ----
    indices = courses[courses["Course code"].isin(taken_courses)].index.tolist()
    if not indices:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    student_vector = np.asarray(tfidf_matrix[indices].mean(axis=0)).reshape(1, -1)

    # ---- compute similarity for major courses ----
    major_vectors = tfidf_matrix[major_pool.index].toarray()
    major_sims = cosine_similarity(student_vector, major_vectors).flatten()
    major_pool = major_pool.assign(similarity=major_sims)

    # sort major courses
    major_sorted = major_pool.sort_values(by="similarity", ascending=False)

    # ---- weighted electives ----
    if not elective_pool.empty:
        elective_vectors = tfidf_matrix[elective_pool.index].toarray()
        elective_sims = cosine_similarity(student_vector, elective_vectors).flatten()
        elective_pool = elective_pool.assign(similarity=elective_sims)

        # 70% relevance, 30% diversity/randomness
        elective_pool["weighted"] = (
            0.7 * elective_pool["similarity"] +
            0.3 * np.random.rand(len(elective_pool))
        )

        elective_pool = elective_pool.sort_values(by="weighted", ascending=False)

    # ============================
    # BUILD RECOMMENDATIONS
    # ============================

    recommendations = []
    total_credits = 0

    # ---- Add major courses until nearly full ----
    for _, row in major_sorted.iterrows():
        ch = int(row["Credit Hours"])
        if total_credits + ch <= credit_limit:
            recommendations.append(row)
            total_credits += ch
        if total_credits >= credit_limit:
            break

    # ---- Add electives to fill remaining credit space ----
    remaining_credits = credit_limit - total_credits
    elective_recs = []

    if not elective_pool.empty and remaining_credits > 0:
        for _, row in elective_pool.iterrows():
            ch = int(row["Credit Hours"])
            if ch <= remaining_credits:
                elective_recs.append(row)
                remaining_credits -= ch
                if remaining_credits <= 0:
                    break

    recommendations.extend(elective_recs)

    # ---- return formatted results ----
    if not recommendations:
        return pd.DataFrame(columns=["Student_ID", "Course code", "Course Name", "similarity"])

    out = pd.DataFrame(recommendations)[["Course code", "Course Name", "similarity"]]
    out["Student_ID"] = student_row["Student_ID"]
    return out.head(top_n)


# ============================
# 5. RUN FOR ALL STUDENTS
# ============================

all_recs = pd.concat(
    [recommend_for_student(row) for _, row in students.iterrows()],
    ignore_index=True
)

all_recs.to_csv("baseline_rec.csv", index=False)
print("Baseline recommendations saved!")
