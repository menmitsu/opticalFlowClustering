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

from computeOpticalFlowModule import ComputeOpticalFLow
from sklearn.cluster import KMeans

def preprocess_image(image):
    """
    Preprocess an image by converting it to grayscale, applying thresholding, and
    merging the result with the original image to create a four-channel RGBA image.
    """


     # Threshold black pixels
    image[image < 30] = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    alpha[alpha > 0] = 255
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]


    return cv2.merge(rgba, 4)

def cluster_colors(image, n_clusters):
    """
    Cluster the colors in an image using K-means clustering and return the
    resulting color bar image as a NumPy array.
    """

    # print(image.shape)

    # # # Filter out pixels with alpha value of 0
    # image = image[image[:, :, 3] > 0]



    flattened_image = image.reshape(image.shape[0] * image.shape[1], 4)
    clt = KMeans(n_clusters=n_clusters,max_iter=1)
    clt.fit(flattened_image)

    # get labels for all points
    labels = clt.predict(flattened_image)

    # get counts for each unique label
    label_counts = np.bincount(labels)

     # Get percentages of each cluster and sort by percentage
    label_percentages = label_counts.astype(float) / len(flattened_image)
    label_info = []
      
    for i, centroid in enumerate(clt.cluster_centers_):
        label_info.append((label_percentages[i], f"Cluster {i+1}", centroid))

    # Sort the list of tuples by label percentages in descending order
    label_info = sorted(label_info, key=lambda x: x[0], reverse=True)
    
    r0, g0, b0, a0 = np.rint(label_info[0][2])   

    rgb0 = np.array([[[r0, g0, b0]]], dtype=np.uint8)
    

    # Convert RGB to HSV using OpenCV
    hsv0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2HSV)
        

    return hsv0[0][0][0]

def findDominantColorFromHistogram(hsv_img):

    # Calculate the histogram of the hue channel
    hue_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])

    # Find the bin with the highest frequency
    dominant_color_bin = hue_hist.argmax()

    # Convert the bin number to an HSV value
    dominant_color_hsv = np.array([[[dominant_color_bin, 255, 255]]], dtype=np.uint8)
    # dominant_color_rgb = cv2.cvtColor(dominant_color_hsv, cv2.COLOR_HSV2BGR)

    # print(f"The most dominant color in HSV format is: ",dominant_color_hsv)

    return dominant_color_hsv


def computeDominantHSVinGrid(framNum,frame, grid_params,csv_file,inputVideoFile,method):
    height, width = frame.shape[:2]
    aspect_ratio = float(width) / height
    
    grid_width = int(aspect_ratio * grid_params['cell_height'] * grid_params['cols'])
    grid_height = grid_params['cell_height'] * grid_params['rows']
    x_step = int(width / grid_params['cols'])
    y_step = int(height / grid_params['rows'])

    num_cells = grid_params['cols'] * grid_params['rows']
    
    # Loop over each cell in the grid
    cell_idx = 0
    hsv_colors_flat = []

    # set the number of jobs to use for parallel processing
    num_jobs = multiprocessing.cpu_count()
   

    for y in range(grid_params['rows']):
        
        for x in range(grid_params['cols']):
            x1 = x * x_step
            y1 = y * y_step
            x2 = min(x1 + x_step, width)
            y2 = min(y1 + y_step, height)
            
            grid_roi = frame[y1:y2, x1:x2]
            grid_roi_hsv= cv2.cvtColor(grid_roi, cv2.COLOR_BGR2HSV)

            processed_image = preprocess_image(grid_roi_hsv)

            if(method=="dominantHue"):
                dominant_color_hsv=findDominantColorFromHistogram(grid_roi_hsv)
                hsv_colors_flat.append(dominant_color_hsv[0][0][0])
            elif(method=="KMeans"):
                hsvVal = cluster_colors(processed_image, 1)
                hsv_colors_flat.append(hsvVal)
                   
            cell_idx += 1


    df = pd.DataFrame(data=[hsv_colors_flat], columns=[f"cell_{i}" for i in range(num_cells)])

    if(framNum<=2):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file,mode='a', index=False, header=False)

                    

def process_video(inputVideoFile,method="KMeans"):
    
    # Load the video 
    cap = cv2.VideoCapture(inputVideoFile)
    
    number_of_videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameNum = 1

    ret, frame = cap.read()

    grid_params = {'rows': 14, 'cols': 25, 'cell_width': 50, 'cell_height': 50}

    compflow = ComputeOpticalFLow(frame)

    csvFilePath="OutCSV/"+str(inputVideoFile).split('.')[0]+".csv"

    while(cap.isOpened()):

        print("\n\n Processing FrameNum: ",frameNum," of ", number_of_videoFrames)
        
        ret, frame_rgb = cap.read()
        
        if not ret: break

        frame_optical = compflow.compute(frame_rgb)

        frameNum = frameNum + 1

        key = cv2.waitKey(30) & 0xFF
        
        computeDominantHSVinGrid(frameNum,frame_optical, grid_params, csv_file=csvFilePath, inputVideoFile=inputVideoFile,method=method)

    cap.release()
    cv2.destroyAllWindows()

# Color Kmeans

def parse_arguments():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to the input video")
    ap.add_argument("--method",required=False,help="KMeans,dominantHue")
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = parse_arguments()

        
    process_video(args['path'],args['method'])

