#!/bin/bash

# Check if image directory argument was passed
if [ -z "$1" ]
  then
    echo "Error: Please provide the path to the image directory as the first argument."
    exit 1
fi

IMAGES_DIR="$1"
CSV_FILE="$2"

# Replace "python_program.py" with the name of your Python program
PYTHON_PROGRAM="color_kmeans.py"

# Loop through all files in the directory and run the Python program on each one
for file in "$IMAGES_DIR"/*; do
    "python3" "$PYTHON_PROGRAM" "-i" "$file" "-c" "1" "-f" "$CSV_FILE"
done