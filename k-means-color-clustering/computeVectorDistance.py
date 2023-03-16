import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load the CSV files
file1 = open('file1.csv', 'r')
file2 = open('file2.csv', 'r')

# extract the HSV values into separate vectors
hsv1 = []
hsv2 = []

reader1 = csv.reader(file1)
for row in reader1:
    hsv1.append(row[1:])

reader2 = csv.reader(file2)
for row in reader2:
    hsv2.append(row[1:])

# convert the vectors to NumPy arrays
hsv1 = np.array(hsv1, dtype=float)
hsv2 = np.array(hsv2, dtype=float)

cos_similarity = np.dot(hsv1, hsv2.T) / (np.linalg.norm(hsv1, axis=1) * np.linalg.norm(hsv2, axis=1))
similarity = cosine_similarity(hsv1.reshape(1,-1), hsv2.reshape(1,-1))
print(similarity)

cos_similarity = cos_similarity[0]  # convert the result to a scalar

# compute the Euclidean distance between the two vectors
euclidean_distance = 0

min_len = min(len(hsv1), len(hsv2))

for i in range(min_len):
    subvector_distance = np.linalg.norm(hsv1[i] - hsv2[i])
    euclidean_distance += subvector_distance

if len(hsv1) != len(hsv2):
    print("Warning: The vectors have different lengths, only the Euclidean distance of the common subvectors has been computed.")

print("Cosine similarity:", cos_similarity)
print("Euclidean distance:", euclidean_distance)
