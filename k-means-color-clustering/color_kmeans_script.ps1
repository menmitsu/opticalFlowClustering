# Check if image directory argument was passed
if (-not $args[0])
{
  echo "Error: Please provide the path to the image directory as the first argument."
  exit 1
}

$IMAGES_DIR=$args[0]
$CSV_FILE=$args[1]

# Replace "python_program.py" with the name of your Python program
$PYTHON_PROGRAM="color_kmeans.py"

# Loop through all files in the directory and run the Python program on each one
foreach ($file in Get-ChildItem $IMAGES_DIR)
{
  & python $PYTHON_PROGRAM "-i" $file.FullName "-c" "1" "-f" $CSV_FILE
}