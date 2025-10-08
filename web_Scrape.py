import requests
import pandas as pd

url = "https://fisk.smartcatalogiq.com/Institutions/Fisk-University/json/bulletin/undergraduate-local.json" 
response = requests.get(url)
data = response.json()

courses = []

def extract_courses(node, subject=None, level=None):
    """Recursively dig into Children until we hit actual courses."""
    name = node.get("Name")

    # If this looks like a subject for example "MATH" could be  "Mathematics"
    if " - " in (name or "") and not subject:
        subject = name

    # If this looks like a level e.g CSCI 310 would be under 300
    if name and name.isdigit():
        level = name

    # If children exist, recurse
    if "Children" in node and node["Children"]:
        for child in node["Children"]:
            extract_courses(child, subject, level)
    else:
        # Leaf node = actual course 
        courses.append({
            "Subject": subject,
            "Level": level,
            "Code": node.get("Name"),
            "Title": node.get("Title"),
            "Description": node.get("Description"),
            "Prerequisites": node.get("Prerequisite")
        })

# Start recursion
for child in data["Children"]:
    extract_courses(child)

# Convert to DataFrame
df = pd.DataFrame(courses)

# Save as CSV
df.to_csv("fisk_courses_clean.csv", index=False)
print("Saved:", len(df), "courses")
