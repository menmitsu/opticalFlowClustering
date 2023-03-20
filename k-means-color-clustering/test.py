import cv2

# Replace "path_to_video" with the actual path to your input video file
path_to_video = "video_lq_optical.mp4"

# Create a VideoCapture object and open the input video file
cap = cv2.VideoCapture(path_to_video)

# Check if the VideoCapture object was successfully opened
if not cap.isOpened():
    print("Error opening video file")

# Read the first frame from the video
ret, frame = cap.read()

# Check if the frame was successfully read
if not ret:
    print("Error reading first frame")

# Display the first frame
cv2.imwrite("F.jpg", frame)
