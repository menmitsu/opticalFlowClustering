import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

# load the data
bounce_df = pd.read_csv('bounceVectors_reduceDim.csv')

# fill empty cells with 0
bounce_df.fillna(0, inplace=True)

# compute the covariance matrix of the bounce data
cov_matrix = np.cov(bounce_df.iloc[:, 2:].T)

# compute the inverse of the covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# compute the Mahalanobis distance between all pairs of vectors in the bounce dataset
n = bounce_df.shape[0]
distances = []
for i in range(n):
    for j in range(i+1, n):
        x = bounce_df.iloc[i, 2:].values
        y = bounce_df.iloc[j, 2:].values
        distance = mahalanobis(x, y, inv_cov_matrix)
        distances.append(distance)

# compute the average, maximum, median, and minimum distance between the vectors
avg_distance = np.mean(distances)
max_distance = np.max(distances)
med_distance = np.median(distances)
min_distance = np.min(distances)

print("Average Mahalanobis distance:", avg_distance)
print("Maximum Mahalanobis distance:", max_distance)
print("Median Mahalanobis distance:", med_distance)
print("Minimum Mahalanobis distance:", min_distance)
