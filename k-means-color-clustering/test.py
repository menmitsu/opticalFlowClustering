# import cv2

# # Replace "path_to_video" with the actual path to your input video file
# path_to_video = "video_lq_optical.mp4"

# # Create a VideoCapture object and open the input video file
# cap = cv2.VideoCapture(path_to_video)

# # Check if the VideoCapture object was successfully opened
# if not cap.isOpened():
#     print("Error opening video file")

# # Read the first frame from the video
# ret, frame = cap.read()

# # Check if the frame was successfully read
# if not ret:
#     print("Error reading first frame")

# # Display the first frame
# cv2.imwrite("F.jpg", frame)

# import pandas as pd

# # file1_name = 'cropped_trimmed2.csv'
# file1_name = 'addnew.csv'
# nobounce_name = 'bounce.csv'

# file1_df = pd.read_csv(file1_name, header=None)
# nobounce_df = pd.read_csv(nobounce_name, header=None)

# # Extract the "Hue" column as numpy arrays
# file1_hue = file1_df.iloc[:, 3].values
# print(file1_hue)

import os
import re
import pandas as pd

def get_number(filename):
    pattern = re.compile(r'(\d+)')
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    else:
        return None

dirs = 'OutImgs/video_lqmastertrim'
for contentFolder in sorted(os.listdir(dirs), key=get_number):
    filepathcsv = f'{contentFolder}.csv'
    hsv_colors_flat = []
    for img_path in sorted(os.listdir(dirs+'/'+contentFolder), key=get_number):

        df = pd.DataFrame(data=[hsv_colors_flat], columns=[f"cell_{i}" for i in range(350)])

        # if(framNum<=2):
        #     df.to_csv(csv_file, index=False)
        # else:
        #     df.to_csv(csv_file,mode='a', index=False, header=False)