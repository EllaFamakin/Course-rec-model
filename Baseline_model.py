import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Loading dataset
df = pd.read_csv("fisk_courses_tagged.csv")

#Create a column that merges course information
df["content"] = (
    df["Course code"].astype(str) + " " +
    df["Course Name"].astype(str) + " " +
    df["Major_Applicable"].astype(str) + " " +
    df["Requirement_Type"].astype(str)
)

#Initializing the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

#Transform the course content into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df["content"])

#Compute cosine similarity between all courses ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define a function to get course recommendations
def recommend_courses(course_code, n=5):
    # Check if the course exists in the dataset
    if course_code not in df["Course code"].values:
        return f" Course '{course_code}' not found in the dataset."
    
    # Get the index of the target course
    idx = df.index[df["Course code"] == course_code][0]
    
    # Get similarity scores for all other courses
    sim_scores = list(enumerate(cosine_sim[idx]))
    print (sim_scores)
    
    # Sort by highest similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Pick top N similar courses (minus the target course)
    top_courses = sim_scores[1:n+1]
    
    # Return the course codes and titles
    results = df.iloc[[i[0] for i in top_courses]][["Course code", "Course Name", "Major_Applicable"]]
    return results

#Test
print(recommend_courses("CSCI 110"))
