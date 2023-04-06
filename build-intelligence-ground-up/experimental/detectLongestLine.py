import cv2
import numpy as np

# Load the image
img = cv2.imread('image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Canny edge detector
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply the Hough Transform to the edge map
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Find the longest line
max_len = 0
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length > max_len:
        max_len = length
        longest_line = line

# Draw the longest line on the original image
cv2.line(img, (longest_line[0][0], longest_line[0][1]), (longest_line[0][2], longest_line[0][3]), (0, 0, 255), 2)

# Display the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
