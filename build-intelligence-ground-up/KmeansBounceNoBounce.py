import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the data from the csv files
bounce_data = pd.read_csv('bounce_combined.csv', header=None)
nobounce_data = pd.read_csv('nobounce_combined.csv', header=None)

# Extract the labels and feature vectors
bounce_labels = bounce_data.iloc[:, 1]
bounce_vectors = bounce_data.iloc[:, 2:10]

nobounce_labels = nobounce_data.iloc[:, 1]
nobounce_vectors = nobounce_data.iloc[:, 2:10]

# Cluster the data using KMeans with k=2 (one cluster for each label)
kmeans = KMeans(n_clusters=2)
kmeans.fit(pd.concat([bounce_vectors, nobounce_vectors]))

# Assign labels to each cluster based on which label is most common in the cluster
bounce_cluster_label = bounce_labels.mode()[0]
nobounce_cluster_label = nobounce_labels.mode()[0]

cluster_labels = []
for label in kmeans.labels_:
    if label == 0:
        cluster_labels.append(bounce_cluster_label)
    else:
        cluster_labels.append(nobounce_cluster_label)

# Convert the string labels to numerical labels
label_encoder = LabelEncoder()
label_encoder.fit(cluster_labels)
numerical_labels = label_encoder.transform(cluster_labels)

# Fit a linear regression model to the clustered data
regression_model = LinearRegression()
regression_model.fit(pd.concat([bounce_vectors, nobounce_vectors]), numerical_labels)

# Print the coefficients of the regression model to see if there is a clear separation
print(regression_model.coef_)
