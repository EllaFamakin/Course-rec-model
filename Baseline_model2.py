import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#dataset
df = pd.read_csv("fisk_courses_tagged.csv")

# Ensure consistent formats
df["Course code"] = df["Course code"].astype(str)
df["Course_level"] = df["Course_level"].astype(str).str.extract(r"(\d{3})").fillna("100")  # extract 100, 200, etc.
df["Course_level"] = df["Course_level"].astype(int)
df["Credit Hours"] = df.get("Credit Hours", pd.Series([3]*len(df)))  # assume 3 if missing

#Ask for student inputs
print("Welcome to the Fisk Course Recommender!")

major = input("Enter your major (e.g., Computer Science, Data Science, Math, Bioinformatics): ").strip()
classification = input("Enter your classification (Freshman, Sophomore, Junior, Senior): ").strip().title()
honors = input("Are you an Honors student? (yes/no): ").lower() == "yes"

# Credit hour limits
if honors:
    credit_limit = 21 
else:
    credit_limit = 18

#Filter by level and major ---
def filter_by_level(classification):
    level_ranges = {
        "Freshman": (0, 199),
        "Sophomore": (200, 299),
        "Junior": (300, 399),
        "Senior": (400, 499)
    }
    return level_ranges.get(classification, (100, 499))

low, high = filter_by_level(classification)
filtered = df[
    (df["Course_level"].between(low, high)) &
    (df["Major_Applicable"].str.contains(major, case=False, na=False) |
    df["Requirement_Type"].str.contains("General Elective", case=False, na=False) |
    df["Requirement_Type"].str.contains("Core", case=False, na=False))
]

print(f"\nCourses available for {classification} in {major}:")
print(filtered[["Course code", "Course Name", "Credit Hours", "Prerequisite"]])

#Ask what courses have been taken
taken = input("\nEnter the course codes you have completed (comma-separated, e.g., 'CSCI 110, MATH 120'): ")
taken_courses = [c.strip().upper() for c in taken.split(",") if c.strip()]

# Remove taken courses
remaining = filtered[~filtered["Course code"].isin(taken_courses)].copy()

# Apply prerequisite logic
def check_prereqs(prereq, taken_list):
    """Return True if the course can be taken (prereqs met or none)."""
    if pd.isna(prereq) or prereq.strip().lower() in ["none", ""]:
        return True
    for t in taken_list:
        if t in prereq:
            return True
    return False

remaining["Eligible"] = remaining["Prerequisite"].apply(lambda p: check_prereqs(p, taken_courses))
eligible = remaining[remaining["Eligible"] == True]

# Enforce credit hour limit
# Sort eligible courses by level, then add until limit is reached
eligible_sorted = eligible.sort_values(by=["Course_level"])
recommendations = []
total_credits = 0

eligible_sorted["Credit Hours"] = pd.to_numeric(eligible_sorted["Credit Hours"], errors="coerce").fillna(0)
for _, row in eligible_sorted.iterrows():
    if total_credits + row["Credit Hours"] <= credit_limit:
        recommendations.append(row)
        total_credits += row["Credit Hours"]

# Rank by dependency (prereqs met first)
recommendations = pd.DataFrame(recommendations)
recommendations["Has_Prereq"] = recommendations["Prerequisite"].notna()
recommendations = recommendations.sort_values(by=["Has_Prereq", "Course_level"], ascending=[True, True])

# --- Display final recommendations ---
print("\nRecommended courses for you this semester:")
print(recommendations[["Course code", "Course Name", "Credit Hours", "Prerequisite"]])

print(f"\nTotal Credit Hours: {total_credits} (Limit: {credit_limit})")


#Cosine Similarity check

# Create a “text” field combining what we know about each course
df["text"] = (
    df["Course code"].fillna("") + " " +
    df["Course Name"].fillna("") + " " +
    df["Requirement_Type"].fillna("") + " " +
    df["Major_Applicable"].fillna("")
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

# Function to get recommendations
def recommend_courses(taken_courses, df, tfidf_matrix, top_n=5):
    # Get indices of courses taken
    indices = df[df["Course code"].isin(taken_courses)].index.tolist()
    if not indices:
        print("None of the taken courses found in dataset.")
        return None

    # Average the vectors for courses taken (creating a “student profile”)
    student_vector = tfidf_matrix[indices].mean(axis=0)
    
    student_vector = np.asarray(student_vector).reshape(1, -1)
    tfidf_matrix = tfidf_matrix.toarray()
    # Compute cosine similarity between this profile and all courses
    sim_scores = cosine_similarity(student_vector, tfidf_matrix).flatten()

    # Create a DataFrame for ranking
    df["similarity"] = sim_scores
    
    # Exclude already taken courses
    recs = df[~df["Course code"].isin(taken_courses)]
    
    # Sort by similarity
    recs = recs.sort_values(by="similarity", ascending=False)
    
    # Return top N recommendations
    return recs[["Course code", "Course Name", "similarity", "Major_Applicable", "Requirement_Type"]].head(top_n)

# check
# taken_courses = ["CSCI 110", "MATH 130"]  # 🧑‍🎓 student input
recommendations = recommend_courses(taken_courses, df, tfidf_matrix, top_n=5)
print(recommendations)
