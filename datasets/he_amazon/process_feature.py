import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read the original feature file
print("Reading UserFeature_ori.txt...")

df = pd.read_csv("./UserFeature.txt", delimiter='\t', header=None)

# Check for missing values in the dataframe
print("Checking for missing values in the dataframe...")
missing_values = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)

# Check for empty strings or whitespace
print("\nChecking for empty strings or whitespace...")
empty_strings = (df == '').sum()
whitespace = df.applymap(lambda x: isinstance(x, str) and x.isspace()).sum()
print("Number of empty strings in each column:")
print(empty_strings)
print("Number of whitespace-only strings in each column:")
print(whitespace)

# Replace empty strings or whitespace with 0.0
print("\nReplacing empty strings and whitespace with 0.0...")
df = df.applymap(lambda x: 0.0 if (pd.isna(x) or (isinstance(x, str) and (x == '' or x.isspace()))) else x)

# Verify the replacement
print("Verifying replacement...")
empty_strings_after = (df == '').sum()
whitespace_after = df.applymap(lambda x: isinstance(x, str) and x.isspace()).sum()
print("Number of empty strings after replacement:")
print(empty_strings_after)
print("Number of whitespace-only strings after replacement:")
print(whitespace_after)

# Save the processed features to UserFeature.txt
print("\nSaving processed features to UserFeature.txt...")
df.to_csv("./UserFeature.txt", sep='\t', header=None, index=None)

print("Processing complete!")

