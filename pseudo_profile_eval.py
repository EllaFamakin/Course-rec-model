import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from Combined_base_model import recommend_for_student, courses, vectorizer, tfidf_matrix  # IMPORT YOUR MODEL

# ----------------------------
# LOAD STUDENT PROFILE DATA
# ----------------------------
students = pd.read_csv("mock_student_profiles.csv")


# ----------------------------
# METRIC STORAGE
# ----------------------------
results = {
    "HR@1": [],
    "HR@3": [],
    "HR@5": [],
    "P@3": [],
    "P@5": [],
    "R@5": [],
    "MRR": []
}


# ----------------------------
# EVALUATION LOOP
# ----------------------------
for _, student in students.iterrows():

    taken_list = [
        c.strip().upper() for c in str(student["Courses_Taken"]).split(",") if c.strip()
    ]

    if len(taken_list) < 2:
        continue  # not enough data for LOO

    # --- Step 1: hold out one course ---
    held_out = np.random.choice(taken_list)
    remaining = [c for c in taken_list if c != held_out]

    # temporarily overwrite for evaluation
    student_eval = student.copy()
    student_eval["Courses_Taken"] = ", ".join(remaining)

    # --- Step 2: run recommender ---
    recs = recommend_for_student(student_eval, top_n=20)  # larger pool for metrics
    if recs is None or recs.empty:
        continue

    rec_list = recs["Course code"].tolist()

    # --- Step 3: compute metrics ---

    # Hit Rate
    results["HR@1"].append(int(held_out in rec_list[:1]))
    results["HR@3"].append(int(held_out in rec_list[:3]))
    results["HR@5"].append(int(held_out in rec_list[:5]))

    # Precision
    results["P@3"].append(int(held_out in rec_list[:3]) / 3)
    results["P@5"].append(int(held_out in rec_list[:5]) / 5)

    # Recall@5
    # since ground truth = 1 course
    results["R@5"].append(int(held_out in rec_list[:5]))

    # Mean Reciprocal Rank (MRR)
    if held_out in rec_list:
        rank = rec_list.index(held_out) + 1
        results["MRR"].append(1 / rank)
    else:
        results["MRR"].append(0)


# ----------------------------
# PRINT FINAL METRICS
# ----------------------------
print("\n===== LEAVE-ONE-OUT EVALUATION =====\n")
print(f"⭐ Hit Rate @1: {np.mean(results['HR@1']):.3f}")
print(f"⭐ Hit Rate @3: {np.mean(results['HR@3']):.3f}")
print(f"⭐ Hit Rate @5: {np.mean(results['HR@5']):.3f}")

print(f"\n⭐ Precision @3: {np.mean(results['P@3']):.3f}")
print(f"⭐ Precision @5: {np.mean(results['P@5']):.3f}")

print(f"\n⭐ Recall @5: {np.mean(results['R@5']):.3f}")

print(f"\n⭐ Mean Reciprocal Rank (MRR): {np.mean(results['MRR']):.3f}")

print("\n=====================================\n")
