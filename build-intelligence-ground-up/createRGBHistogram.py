import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print('Usage: python rgb_histogram.py <video_path>')
    sys.exit()

# Get input video file path from command line argument
video_path = sys.argv[1]

# Load video file
cap = cv2.VideoCapture(video_path)

# Initialize empty lists to store histogram data
r_hist = []
g_hist = []
b_hist = []

while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()
    if ret == False:
        break

    # Convert frame to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate histogram for each channel
    r_hist_frame = cv2.calcHist([frame],[0],None,[256],[0,256])
    g_hist_frame = cv2.calcHist([frame],[1],None,[256],[0,256])
    b_hist_frame = cv2.calcHist([frame],[2],None,[256],[0,256])

    # Normalize histogram data and add to lists
    r_hist.append(r_hist_frame / np.sum(r_hist_frame))
    g_hist.append(g_hist_frame / np.sum(g_hist_frame))
    b_hist.append(b_hist_frame / np.sum(b_hist_frame))

# Combine histograms for all frames
r_hist = np.sum(r_hist, axis=0)
g_hist = np.sum(g_hist, axis=0)
b_hist = np.sum(b_hist, axis=0)

# Plot RGB histogram
plt.figure()
plt.title('RGB Histogram')
plt.xlabel('Color value')
plt.ylabel('Pixel count')
plt.plot(r_hist, color='r', label='Red')
plt.plot(g_hist, color='g', label='Green')
plt.plot(b_hist, color='b', label='Blue')
plt.legend()

# Save histogram as JPG image with same name as input video file
histogram_file = video_path.split('.')[0] + '_rgb_histogram.jpg'
plt.savefig(histogram_file)

# Release video file
cap.release()
