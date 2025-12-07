# import pandas as pd

# df = pd.read_csv("fisk_courses_tagged.csv")

# # Clean strings
# df["Course code"] = df["Course code"].astype(str).str.strip()
# df["Co-Requisite"] = df["Corequisite"].astype(str).str.strip()

# coreq_map = {}

# for _, row in df.iterrows():
#     course = row["Course code"]
#     # coreq = row["Corequisite"]
#     coreq = str(row.get("Corequisite", "")).strip()

#     if coreq and coreq.lower() in ["", "none", "nan"]:
#         continue
#     if coreq and coreq.lower() != "none" and coreq != "":
#         coreq_map.setdefault(course, []).append(coreq)

#         # Make it automatically bidirectional
#         coreq_map.setdefault(coreq, []).append(course)

# print(coreq_map)

import pandas as pd
import re

df = pd.read_csv("fisk_courses_tagged.csv")

def parse_coreqs(text):
    """Parse AND/OR expressions into structured list-of-lists."""
    if pd.isna(text):
        return []

    text = str(text).strip()
    if text.lower() in ["", "none", "nan"]:
        return []

    # Normalize separators
    cleaned = text.replace("/", " or ").replace("+", " and ").replace("&", " and ")

    # Split by AND → required groups
    and_parts = re.split(r"\band\b", cleaned, flags=re.IGNORECASE)

    structured = []
    for part in and_parts:
        # Within each AND, split OR options
        options = re.split(r"\bor\b", part, flags=re.IGNORECASE)

        course_codes = []
        for opt in options:
            found = re.findall(r"[A-Z]{2,4}\s?\d{3}[A-Z]?", opt)
            course_codes.extend(found)

        if course_codes:
            structured.append(sorted(set(course_codes)))

    return structured


# Build the dictionary
coreq_dict = {}

for _, row in df.iterrows():
    course = row["Course code"]
    coreq_text = row.get("Corequisite", "")
    parsed = parse_coreqs(coreq_text)

    if parsed:
        coreq_dict[course] = parsed


# Save results
out = pd.DataFrame([
    {"Course": c, "Parsed_Coreqs": str(groups)}
    for c, groups in coreq_dict.items()
])

out.to_csv("corequisites_dictionary.csv", index=False)
print("Co-requisite dictionary created!")
