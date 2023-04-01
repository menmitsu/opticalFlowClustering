# Importing dependancies
import os
import re
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import time
import multiprocessing
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# Load the CSV files
bounce_df = pd.read_csv('bounceVectors_reduceDim.csv',header=None)
nobounce_df = pd.read_csv('noBounceVectors_reduceDim.csv',header=None)

# Combine the two datasets and shuffle them
combined_df = pd.concat([bounce_df, nobounce_df], ignore_index=True)
print((combined_df.shape))

# Check for NaN values in each column and print the result
nan_columns = combined_df.columns[combined_df.isna().any()].tolist()
print("Columns containing NaN values:", nan_columns)

# combined_df = combined_df.dropna()
print((combined_df.shape))

combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Split the data into features (X) and labels (y)
X = combined_df.iloc[:, 3:75]
y = combined_df.iloc[:, 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the DataFrame to a NumPy array
X_train_np = X_train.to_numpy()

X_test_np = X_test.to_numpy()

# Reshape the NumPy array
X_train_np = np.reshape(X_train_np, (X_train_np.shape[0], X_train_np.shape[1], 1))
# Reshape the NumPy array
X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1], 1))


# Assuming X_train, X_test, y_train, and y_test have been prepared
# Reshape the input data into a format compatible with LSTMs
X_train_np = np.reshape(X_train_np, (X_train_np.shape[0], X_train_np.shape[1], 1))
X_test_np = np.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1], 1))



# Create a simple LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))
