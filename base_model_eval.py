# evaluation/evaluate_baseline.py
import pandas as pd

# Load recommendations
recs = pd.read_csv("baseline_recommendations.csv")

# Evaluation Functions
print("Columns in recs DataFrame:", recs.columns.tolist())
print(recs.head())  # Optional, to see sample data

def evaluate_logical_consistency(recs, classification):
    # wrong_major = recs[~recs["Major_Applicable"].str.contains(student_major, case=False, na=False)]
    wrong_level = recs[(recs["Course code"].str.extract(r"(\d{3})").astype(float) > 200) & (classification.lower() == "freshman")]

    print(" Logical Consistency Check:")
    # print(f" - Out-of-major courses: {len(wrong_major)}")
    print(f" - Too advanced for level: {len(wrong_level)}")
    print()

def evaluate_diversity(recs):
    subject_prefixes = recs["Course code"].str.extract(r"([A-Za-z]+)")[0]
    diversity_score = subject_prefixes.nunique() / len(subject_prefixes)
    print(f"Diversity Score: {diversity_score:.2f}")
    print("Good variety!" if diversity_score > 0.5 else " Too concentrated.")
    print()

def evaluate_similarity_strength(recs):
    avg_similarity = recs["similarity"].mean()
    print(f" Average similarity: {avg_similarity:.3f}")
    print("Strong semantic similarity!" if avg_similarity > 0.3 else " Could be improved.")
    print()

# Run Evaluations
major = "Computer Science"
classification = "Sophomore"

print("\n Evaluating Baseline Recommendations...\n")
evaluate_logical_consistency(recs, classification)
evaluate_diversity(recs)
evaluate_similarity_strength(recs)

print("Evaluation complete.")
