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
from sklearn.cluster import KMeans

def cluster_colors(image, n_clusters,method="KMeans"):
    """
    Cluster the colors in an image using K-means clustering and return the
    resulting color bar image as a NumPy array.
    """

    # print(image.shape)

    # # # Filter out pixels with alpha value of 0
    # image = image[image[:, :, 3] > 0]



    flattened_image = image.reshape(image.shape[0] * image.shape[1], 3)

    if(method=="KMeans"):
        clt = KMeans(n_clusters=n_clusters,max_iter=10)
        clt.fit(flattened_image)
    elif(method=="FaissKMeans"):
        clt = FaissKMeans(n_clusters=n_clusters,max_iter=10)
        clt.fit(flattened_image)
        
    # get labels for all points
    labels = clt.predict(flattened_image)
    
    # print(clt.cluster_centers_[0])
    r0, g0, b0= np.rint(clt.cluster_centers_[0])   

    rgb0 = np.array([[[r0, g0, b0]]], dtype=np.uint8)
    

    # Convert RGB to HSV using OpenCV
    hsv0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2HSV)
        

    return hsv0[0][0][0]

def processVideo(inputVideoFile,csv_file):

    # Load the video 
    cap = cv2.VideoCapture(inputVideoFile)

    
    number_of_videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameNum = 1

    ret, frame = cap.read()
    hsv_colors_flat = []

    hsv_colors_flat.append(inputVideoFile)

    while(cap.isOpened()):

        print("\n\n Processing FrameNum: ",frameNum," of ", number_of_videoFrames)
        
        ret, frame_rgb = cap.read()
        
        if not ret: break
        hsvVal = cluster_colors(frame_rgb, 1)
        hsv_colors_flat.append(hsvVal)

        frameNum=frameNum+1

    print ("HSV vector",hsv_colors_flat)

    df = pd.DataFrame(data=[hsv_colors_flat])
    df.to_csv(csv_file,mode='a', index=False, header=False)


def parse_arguments():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to the input video")
    ap.add_argument("--csvFile",required=True, help="CSVFile to append data to")
    
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = parse_arguments()
    processVideo(args['video'],args['csvFile'])
    
