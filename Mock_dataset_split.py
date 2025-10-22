import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Mock student profile dataset
df = pd.read_csv("mock_student_profiles.csv")

# Split into training, testing and validiation datasets
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

#Saving each split into csv files
train.to_csv("mock_student_profiles_train.csv", index=False)
val.to_csv("mock_student_profiles_val.csv", index=False)
test.to_csv("mock_student_profiles_test.csv", index=False)