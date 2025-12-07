
import pandas as pd
import re

courses = pd.read_csv("fisk_courses_tagged.csv")

prereq_dict = {}

def parse_prereq(pr_string):
    """
    Convert a prerequisite string into AND-of-ORs format.
    Example:
        "A AND B OR C AND (D OR E)"
    Output:
        [["A"], ["B", "C"], ["D", "E"]]
    """
    if pd.isna(pr_string) or pr_string.strip().lower() in ["", "none"]:
        return []

    # normalize text
    s = pr_string.upper().replace("/", " OR ").replace(",", " AND ")
    s = re.sub(r"\s+", " ", s)  # collapse spaces

    # Force parentheses around OR groups for easier parsing
    s = s.replace(" OR ", "|")
    s = s.replace(" AND ", "&")

    # Split by AND (&)
    and_parts = s.split("&")

    parsed = []
    for part in and_parts:
        # OR logic inside part
        ors = part.split("|")
        ors = [o.strip() for o in ors if o.strip() != ""]
        parsed.append(ors)

    return parsed

# Build dictionary from dataset
for _, row in courses.iterrows():
    course = row["Course code"]
    prereq_str = str(row.get("Prerequisite", ""))

    prereq_dict[course] = parse_prereq(prereq_str)

# Save dictionary
output = []
for course, prereqs in prereq_dict.items():
    output.append({
        "Course": course,
        "Parsed_Prereqs": prereqs
    })

df_out = pd.DataFrame(output)
df_out.to_csv("prerequisite_dictionary.csv", index=False)

print("Saved prerequisite_dictionary.csv!")



# import pandas as pd
# import re

# # Load dataset
# df = pd.read_csv("fisk_courses_tagged.csv")

# pre_req_dict = {}

# for _, row in df.iterrows():
#     course = row["Course code"]
    
#     # Convert to string first (prevents the float.lower() error)
#     prereq_raw = str(row.get("Prerequisite", "")).strip()
    
#     # Normalize to lowercase check
#     prereq_lower = prereq_raw.lower()

#     # Skip empty or none-like values
#     if prereq_lower in ["", "none", "nan"]:
#         continue

#     # Extract all course codes inside the co-requisite text
#     # Example: "CSCI 110 + CSCI 110L" → ["CSCI 110", "CSCI 110L"]
#     found = re.findall(r"[A-Z]{2,4}\s?\d{3}[A-Z]?", prereq_raw)

#     if found:
#         pre_req_dict[course] = list(set(found))  # remove duplicates, keep list

# # Save dictionary as CSV for inspection
# pd.DataFrame([
#     {"Course": k, "Pre-Requisites": ", ".join(v)}
#     for k, v in pre_req_dict.items()
# ]).to_csv("Pre_requisites_dictionary.csv", index=False)

# print("Pre-Requisite dictionary created successfully!")
