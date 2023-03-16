import sys
import numpy as np
import pandas as pd

# Read the filenames from the command line arguments
file1_name = sys.argv[1]
nobounce_name = sys.argv[2]

# Load the two CSV files into pandas dataframes
file1_df = pd.read_csv(file1_name, header=None)
nobounce_df = pd.read_csv(nobounce_name, header=None)

# Extract the "Hue" column as numpy arrays
file1_hue = file1_df.iloc[:, 1].values
nobounce_hue = nobounce_df.iloc[:, 1].values

# Define the length of the smaller and larger vectors
smaller_len = len(file1_hue)
larger_len = len(nobounce_hue)

# Calculate cosine similarity for each possible consecutive interval
max_similarity = -1
max_frame = -1
for i in range(larger_len - smaller_len + 1):
    subvector = nobounce_hue[i:i+smaller_len]
    similarity = np.dot(file1_hue, subvector) / (np.linalg.norm(file1_hue) * np.linalg.norm(subvector))
    max_similarity = max(max_similarity, similarity)
    max_frame = i

# Print the maximum cosine similarity
print("Maximum cosine similarity:", max_similarity)
