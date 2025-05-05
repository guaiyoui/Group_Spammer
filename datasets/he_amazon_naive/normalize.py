import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read the original feature file
print("Reading UserFeature_ori.txt...")
# try:
#     # Try reading with header
#     df = pd.read_csv("./UserFeature_ori.txt", delimiter='\t')
# except:
    # If that fails, try reading without header
df = pd.read_csv("./UserFeature_ori.txt", delimiter='\t', header=None)
# Drop the first 8 columns

# print(df.head())

print("Dropping the first 8 columns...")
df = df.iloc[:, 0:20]
# Save the normalized features to a new file
print("Saving normalized features to UserFeature.txt...")
df.to_csv("./UserFeature_noID.txt", sep='\t', index=False, header=None)

print("Normalization complete!")
