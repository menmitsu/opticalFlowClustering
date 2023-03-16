import sys
import numpy as np
import pandas as pd

def calculate_cosine_similarity(file1_hue, nobounce_hue):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    file1_hue (numpy.ndarray): The first vector.
    nobounce_hue (numpy.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    file1_hue_norm = np.linalg.norm(file1_hue)
    nobounce_hue_norm = np.linalg.norm(nobounce_hue)

    # Handle case where one of the vectors has zero norm
    if file1_hue_norm == 0 or nobounce_hue_norm == 0:
        return 0

    # Compute cosine similarity
    similarity = np.dot(file1_hue, nobounce_hue) / (file1_hue_norm * nobounce_hue_norm)

    return similarity


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

print("Vector sizes are: ",smaller_len,larger_len)

# Calculate cosine similarity and sum of squared differences for each possible consecutive interval
max_similarity = -1
max_frame = -1
min_euclidean = 0

for i in range(larger_len - smaller_len + 1):
    subvector = nobounce_hue[i:i+smaller_len]

    # Calculate cosine similarity between the two vectors
    similarity = calculate_cosine_similarity(file1_hue, subvector)
    max_similarity = max(max_similarity, similarity)

    # Keep track of the index where the maximum cosine similarity occurs
    if similarity == max_similarity:
        max_frame = i

# Print the results
print("Maximum cosine similarity:", max_similarity)
print("Minimum sum of squared differences:", min_euclidean)
print("Max frame:",max_frame)
