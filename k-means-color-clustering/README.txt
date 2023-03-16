Example commands:


This will do findCosineSimilarity in a sliding window like manner. A vector is created out of file1 and compared with all possible combinations of vectors 
python3 findCosineDifferentVectors.py bounce.csv 601_3_3_cropped.csv

This Script takes a folder of images, computes the dominant color in the  expected csv file
sh color_kmeans_script.sh images/cropped_trimmed_2/cropped cropped_trimmed2.csv

Use only 2 clusters for now
python3 color_kmeans.py -i images/601_3_cropped_1_OF/cropped/crop_of0021.png -c 1 