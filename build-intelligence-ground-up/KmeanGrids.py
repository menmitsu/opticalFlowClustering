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

from computeOpticalFlowModule import ComputeOpticalFLow

from sklearn.cluster import KMeans
from faiss_kmeans import FaissKMeans
import matplotlib.pyplot as plt

outputVideosDict={}
outputVideosOpticalFlowDict={}


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

def cluster_colors(image, n_clusters,method="KMeans"):
    """
    Cluster the colors in an image using K-means clustering and return the
    resulting color bar image as a NumPy array.
    """

    # print(image.shape)

    # # # Filter out pixels with alpha value of 0
    # image = image[image[:, :, 3] > 0]



    flattened_image = image.reshape(image.shape[0] * image.shape[1], 4)

    if(method=="KMeans"):
        clt = KMeans(n_clusters=n_clusters,max_iter=10)
        clt.fit(flattened_image)
    elif(method=="FaissKMeans"):
        clt = FaissKMeans(n_clusters=n_clusters,max_iter=10)
        clt.fit(flattened_image)
        
    # get labels for all points
    labels = clt.predict(flattened_image)
    
    # print(clt.cluster_centers_[0])
    r0, g0, b0, a0 = np.rint(clt.cluster_centers_[0])   

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

def drawGridCells(frame,frame_optical,grid_params):

    height, width = frame.shape[:2]

    x_step = int(width / grid_params['cols'])
    y_step = int(height / grid_params['rows'])

    cell_idx = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    for y in range(grid_params['rows']):
        for x in range(grid_params['cols']):
            x1 = x * x_step
            y1 = y * y_step
            x2 = min(x1 + x_step, width)
            y2 = min(y1 + y_step, height)
            
            grid_roi = frame[y1:y2, x1:x2]
            grid_optical_roi = frame_optical[y1:y2, x1:x2]

            outputVideosDict[cell_idx].write(grid_roi)
            outputVideosOpticalFlowDict[cell_idx].write(grid_optical_roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            text = f"{cell_idx}"

            x3 = (cell_idx % grid_params['cols']) * x_step
            y3 = (cell_idx // grid_params['cols']) * y_step + 10
        
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x3 + (x_step - text_size[0]) // 2
            text_y = y3 + (y_step - text_size[1]) // 2 + text_size[1]
            
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            


            cell_idx += 1



def initializeOutputVideoWriters(grid_params,videoFPS,inputVideoSize,datasetName):

    num_cells=grid_params['cols']*grid_params['rows']
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    video_size=(int(inputVideoSize[0] / grid_params['cols']),int(inputVideoSize[1] / grid_params['rows']) )
    print(video_size)

    isExist = os.path.exists("OutputVideos")
    if not os.path.exists('OutputVideos'):
        os.makedirs('OutputVideos')
    isExist = os.path.exists("OutputVideos/"+datasetName)
    if not os.path.exists("OutputVideos/"+datasetName):
        os.makedirs("OutputVideos/"+datasetName)

    for cell_index in range(num_cells):
        outputVideosDict[cell_index]=cv2.VideoWriter("OutputVideos/"+datasetName+"/"+str(cell_index)+".mp4",fourcc,videoFPS, video_size)
        outputVideosOpticalFlowDict[cell_index]=cv2.VideoWriter("OutputVideos/"+datasetName+"/"+str(cell_index)+"_optical.mp4",fourcc,videoFPS, video_size)

def computeOperationInGrid(framNum,frame, grid_params,csv_file,inputVideoFile,method):
    height, width = frame.shape[:2]
    
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

            elif(method=="FaissKMeans"):
                hsvVal=cluster_colors(processed_image,1,method="FaissKMeans")
                hsv_colors_flat.append(hsvVal)
                   
            cell_idx += 1


    df = pd.DataFrame(data=[hsv_colors_flat], columns=[f"cell_{i}" for i in range(num_cells)])

    if(framNum<=2):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file,mode='a', index=False, header=False)

def calculateGridParams(inputVideoCap,cell_width,cell_height):
    grid_params={}
   
    grid_params["cell_width"]=cell_width
    grid_params["cell_height"]=cell_height

    print(int(inputVideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    grid_params["cols"]=int(int(inputVideoCap.get(cv2.CAP_PROP_FRAME_WIDTH))/cell_width)
    grid_params["rows"]=int(inputVideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)/cell_height)

    return grid_params

def process_video(inputVideoFile,cell_width,cell_height,method="KMeans"):
    
    # Load the video 
    cap = cv2.VideoCapture(inputVideoFile)

    
    number_of_videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameNum = 1

    ret, frame = cap.read()

    # grid_params = {'rows': 14, 'cols': 25, 'cell_width': 50, 'cell_height': 50}

    grid_params=calculateGridParams(cap,cell_width,cell_height)

    print("Grid rows",grid_params['rows'])
    print("Grid cols",grid_params['cols'])

    compflow = ComputeOpticalFLow(frame)

    csvFilePath="OutCSV/"+str(inputVideoFile).split('.')[0]+".csv"

    initializeOutputVideoWriters(grid_params,cap.get(cv2.CAP_PROP_FPS),(int(cap.get(3)), int(cap.get(4))),os.path.basename(inputVideoFile))
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    while(cap.isOpened()):

        print("\n\n Processing FrameNum: ",frameNum," of ", number_of_videoFrames)
        
        ret, frame_rgb = cap.read()
        
        if not ret: break

        frame=frame_rgb.copy()

        frame_optical = compflow.compute(frame_rgb)

        

        key = cv2.waitKey(30) & 0xFF
       
        # computeOperationInGrid(frameNum,frame_optical, grid_params, csv_file=csvFilePath, inputVideoFile=inputVideoFile,method=method)
        drawGridCells(frame_rgb,frame_optical,grid_params)

        cv2.imshow("Image",frame_rgb)

        if(frameNum==1):
            cv2.imwrite("OutputVideos/"+os.path.basename(inputVideoFile)+"/grid.jpg",frame_rgb)

        frameNum = frameNum + 1

    cap.release()
    cv2.destroyAllWindows()



def parse_arguments():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to the input video")
    ap.add_argument("--method",required=False,help="KMeans,dominantHue")
    ap.add_argument("--show",required=False,help="Show image/vid")
    ap.add_argument("--cell_width",type=int,default=50,required=False,help="Cell Width")
    ap.add_argument("--cell_height",type=int,default=50,required=False,help="Cell Height")
    
    return vars(ap.parse_args())

if __name__ == "__main__":
    args = parse_arguments()
    

        
    process_video(args['path'],args['cell_width'],args['cell_height'],args['method'])

