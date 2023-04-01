import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

# load the data
bounce_df = pd.read_csv('bounceVectors_reduceDim.csv')
nobounce_df = pd.read_csv('noBounceVectors_reduceDim.csv')

# fill empty cells with 0
bounce_df.fillna(0, inplace=True)
nobounce_df.fillna(0, inplace=True)

# compute the covariance matrix of the nobounce data
cov_matrix = np.cov(nobounce_df.iloc[:, 2:].T)

# compute the inverse of the covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# count the number of times Euclidean distance of closest vector in nobounce.csv is less than Euclidean distance of closest vector in bounce.csv
count = 0

# find the closest vector in bounce.csv and nobounce.csv for each vector in bounce.csv
for i in range(bounce_df.shape[0]):
    x = bounce_df.iloc[i, 2:].values
    bounce_vectors_to_search = np.delete(bounce_df.iloc[:, 2:].values, i, axis=0)
    bounce_closest_index = np.argmin([mahalanobis(x, bounce_vectors_to_search[j], inv_cov_matrix) for j in range(bounce_vectors_to_search.shape[0])])
    bounce_closest_vector = bounce_vectors_to_search[bounce_closest_index]
    nobounce_closest_index = np.argmin([mahalanobis(x, nobounce_df.iloc[j, 2:].values, inv_cov_matrix) for j in range(nobounce_df.shape[0])])
    bounce_distance_euclidean = np.linalg.norm(x - bounce_closest_vector)
    nobounce_distance_euclidean = np.linalg.norm(x - nobounce_df.iloc[nobounce_closest_index, 2:].values)
    bounce_distance_manhattan = np.sum(np.abs(x - bounce_closest_vector))
    nobounce_distance_manhattan = np.sum(np.abs(x - nobounce_df.iloc[nobounce_closest_index, 2:].values))
    if nobounce_distance_euclidean < bounce_distance_euclidean:
        count += 1
    print("Vector:", x)
    print("Mahalanobis distance to closest vector in bounce.csv:", mahalanobis(x, bounce_closest_vector, inv_cov_matrix))
    print("Euclidean distance to closest vector in bounce.csv:", bounce_distance_euclidean)
    print("Manhattan distance to closest vector in bounce.csv:", bounce_distance_manhattan)
    print("Mahalanobis distance to closest vector in nobounce.csv:", mahalanobis(x, nobounce_df.iloc[nobounce_closest_index, 2:].values, inv_cov_matrix))
    print("Euclidean distance to closest vector in nobounce.csv:", nobounce_distance_euclidean)
    print("Manhattan distance to closest vector in nobounce.csv:", nobounce_distance_manhattan)
    print("")

print("Number of times Euclidean distance of closest vector in nobounce.csv is less than Euclidean distance of closest vector in bounce.csv:", count)
