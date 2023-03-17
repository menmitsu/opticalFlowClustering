import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd  
import os,argparse
  

parser = argparse.ArgumentParser(
                    prog = 'OpticalFlow',
                    description = 'find optical flow of video')

parser.add_argument('-i', '--input')
args = parser.parse_args() 

# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture(args.input)
number_of_videoFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


size = (2*int(cap.get(3)), int(cap.get(4)))

# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
outputVideo = cv.VideoWriter(args.input+"_opticalflow.mp4", 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         cap.get(cv.CAP_PROP_FPS), size)

output_onlyOpticalFlow = cv.VideoWriter(args.input+"onlyOpticalflow.mp4", 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         cap.get(cv.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))


# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence
ret, first_frame = cap.read()


width=first_frame.shape[1]
height=first_frame.shape[0]
# first_frame.shape[0]=first_frame.shape[0]*2


outputImg=np.zeros([height,2*width,3], dtype=first_frame.dtype)



print("Type1:",first_frame.dtype)
print("outputImg.shape",outputImg.shape)

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255


x_values=[]
y_values=[]

frameNum=0

while(cap.isOpened()):
	
	# ret = a boolean return value from getting
	# the frame, frame = the current frame being
	# projected in the video
	ret, frame = cap.read()
	
	# Opens a new window and displays the input
	# frame

	if(not ret):
		break

	# cv.imshow("input", frame)

	print("Type2:",frame.shape)


	outputImg[0:height,0:width]=frame;
	
	# Converts each frame to grayscale - we previously
	# only converted the first frame to grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	
	# Calculates dense optical flow by Farneback method
	flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
									None,
									0.5, 3, 15, 3, 5, 1.2, 0)
	
	# Computes the magnitude and angle of the 2D vectors
	magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
	
	# Sets image hue according to the optical flow
	# direction
	mask[..., 0] = angle * 180 / np.pi / 2
	
	# Sets image value according to the optical flow
	# magnitude (normalized)
	mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

	print("Average Magnitude of optical flow ",np.mean(magnitude))
	
	x_values.append(frameNum)
	y_values.append(np.mean(magnitude))
	
	# Converts HSV to RGB (BGR) color representation
	rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
	
	# # Opens a new window and displays the output frame
	# # cv.imshow("dense optical flow", rgb)
	# # rgb.copyTo(outputImg(Rect(0, rgb.rows, frame.cols, frame.rows)));
	
	# outputImg[0:height,width:width*2]=rgb;

	# outputVideo.write(outputImg)
	output_onlyOpticalFlow.write(rgb)
	
	# Updates previous frame
	prev_gray = gray
	
	# Frames are read by intervals of 1 millisecond. The
	# programs breaks out of the while loop when the
	# user presses the 'q' key
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

	frameNum=frameNum+1
	# cv.imshow("combined", outputImg)

	print("Number of VideoFrames processed",frameNum,"/",number_of_videoFrames)


dict = {'Frame': x_values, 'Average Magnitude':y_values}  
df = pd.DataFrame(dict)

df.to_csv(args.input+'_opticalFlow.csv')  

# If you have something like this
plt.plot(x_values, y_values, color='black')

# plt.show()
plt.savefig(args.input+"_squares.png")

# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()
