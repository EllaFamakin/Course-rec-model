import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np

vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = np.load("tfidf_matrix.npy")


# Load data
courses = pd.read_csv("fisk_courses_tagged.csv")
recs = pd.read_csv("baseline_recommendations.csv")
students = pd.read_csv("mock_student_profiles.csv")

# --------------- 1. AVERAGE SIMILARITY SCORE -----------------
def average_similarity(recs_df):
    if "similarity" not in recs_df.columns:
        print("No similarity column found in recommendations.")
        return None
    return recs_df["similarity"].mean()

avg_sim = average_similarity(recs)
print("\n Average Similarity Score:", avg_sim)


# --------------- 2. DIVERSITY SCORE --------------------------
from sklearn.metrics.pairwise import cosine_similarity

def diversity_score(recommendations, tfidf_matrix, courses_df):
    """
    Diversity = 1 - average pairwise cosine similarity.
    Values closer to 1 = more diverse.
    """

    if len(recommendations) < 2:
        return None
    
    # Get indices of recommended courses
    rec_indices = [
        courses_df.index[courses_df["Course code"] == code][0]
        for code in recommendations["Course code"]
        if code in courses_df["Course code"].values
    ]
    
    if len(rec_indices) < 2:
        return None

    # Extract TF-IDF vectors for recommended courses
    rec_vectors = tfidf_matrix[rec_indices]

    # Compute pairwise cosine similarity
    sims = cosine_similarity(rec_vectors)

    # Only upper triangle (excluding diagonal)
    upper_tri = sims[np.triu_indices(len(sims), k=1)]

    return 1 - upper_tri.mean()

div_scores = []

for student_id in recs["Student_ID"].unique():
    subset = recs[recs["Student_ID"] == student_id]
    score = diversity_score(subset, tfidf_matrix, courses)
    if score is not None:
        div_scores.append(score)

print("Average Diversity Score:", np.mean(div_scores))


# --------------- 3. COVERAGE RATE ----------------------------
def coverage_rate(recs, total_courses):
    unique_recommended = recs["Course code"].nunique()
    return unique_recommended / total_courses

coverage = coverage_rate(recs, len(courses))
print("\n Recommendation Coverage Rate:", coverage)


# --------------- 4. PREREQUISITE VIOLATION RATE --------------
def prereq_violations(recs, prereq_dict):
    violations = 0

    for _, row in recs.iterrows():
        course = row["Course code"]

        if course not in prereq_dict:
            continue

        # This assumes the recommender did filtering correctly  
        # so any unsatisfied prereqs here mean the recommender failed
        # For now, mock — since taken courses aren’t recorded in recs file
        # You can enhance this with student-by-student checking.
        pass

    print("\n Prereq Violation Checking integrated into full evaluation later.")


# --------------- 5. CLASSIFICATION LEVEL CHECK ---------------
def validate_level_bounds(recs, courses, students):
    wrong_level = 0
    merged = recs.merge(students, on="Student_ID")
    merged = merged.merge(courses, on="Course code")

    for _, row in merged.iterrows():
        level = row["Course_level"]
        year = row["Year"]

        valid_range = {
            "Freshman": (0,199),
            "Sophomore": (200,299),
            "Junior": (300,399),
            "Senior": (400,499)
        }[year]

        if not (valid_range[0] <= level <= valid_range[1]):
            wrong_level += 1

    return wrong_level

bad_levels = validate_level_bounds(recs, courses, students)
print("\n Level Violations:", bad_levels)


# # Load recommendations
# recs = pd.read_csv("baseline_recommendations.csv")

# # Evaluation Functions
# print("Columns in recs DataFrame:", recs.columns.tolist())
# print(recs.head())  # Optional, to see sample data

# def evaluate_logical_consistency(recs, classification):
#      #wrong_major = recs[recs["Major_Applicable"].str.contains(major, case=False, na=False)]
#      wrong_level = recs[(recs["Course code"].str.extract(r"(\d{3})").astype(float) > 200) & (classification.lower() == "freshman")]

#      print(" Logical Consistency Check:")
#      #print(f" - Out-of-major courses: {len(wrong_major)}")
#      print(f" - Too advanced for level: {len(wrong_level)}")
#      print()

# def evaluate_diversity(recs):
#     subject_prefixes = recs["Course code"].str.extract(r"([A-Za-z]+)")[0]
#     diversity_score = subject_prefixes.nunique() / len(subject_prefixes)
#     print(f"Diversity Score: {diversity_score:.2f}")
#     print("Good variety!" if diversity_score > 0.5 else " Too concentrated.")
#     print()

# def evaluate_similarity_strength(recs):
#     avg_similarity = recs["similarity"].mean()
#     print(f" Average similarity: {avg_similarity:.3f}")
#     print("Strong semantic similarity!" if avg_similarity > 0.3 else " Could be improved.")
#     print()

# # Run Evaluations
# major = "Computer Science"
# classification = "Sophomore"

# print("\n Evaluating Baseline Recommendations...\n")
# evaluate_logical_consistency(recs, classification)
# evaluate_diversity(recs)
# evaluate_similarity_strength(recs)

# print("Evaluation complete.")

