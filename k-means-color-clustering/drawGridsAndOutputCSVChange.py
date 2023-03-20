import cv2
import numpy as np
import os
import pandas as pd
import csv
import matplotlib
import argparse
import colorsys

from computeOpticalFlowModule import ComputeOpticalFLow


def load_yolo_bounding_boxes(yolo_bounding_box_file):
    # Load the rows of numbers from the yolo text-file
    data = np.loadtxt(yolo_bounding_box_file)

    # Convert the data to integers and reshape it
    data = np.round(data).astype(np.int32)

    return data.reshape(-1, 11)


def draw_yolo_bounding_box(frame, selectedRow):
    rect = np.delete(selectedRow, np.s_[:3], axis=1) 
    rect = np.delete(rect, np.s_[7:], axis=1) 

    for rect_row in rect:
        cv2.rectangle(frame, (rect_row[0],rect_row[1]), (rect_row[0]+rect_row[2],rect_row[1]+rect_row[3]), (255, 255, 255), thickness=2)

        
def load_contours(inputVideoFile, frameNum, frame):
    # Load segmented contour file
    contour_filepath = "Contours/" + inputVideoFile + "/" + inputVideoFile + "_" + str(frameNum) + ".txt"

    if os.path.isfile(contour_filepath):
        with open(contour_filepath) as f:
            lines = f.readlines()

            for line in lines:
                points = np.fromstring(line, dtype=int, sep=' ')
                points = points.reshape(-1, 1)
                points = np.delete(points, 0,0)
                points = points.reshape(-1, 2)

                if len(points) > 0:
                    cv2.drawContours(frame, [points], -1, (255, 255, 255), thickness=2)
                    cv2.fillPoly(frame, pts =[points], color=(0,0,0))

def overlayGridAndComputeAvgColor(framNum,frame, grid_params,csv_file,inputVideoFile):
    height, width = frame.shape[:2]
    aspect_ratio = float(width) / height
    
    grid_width = int(aspect_ratio * grid_params['cell_height'] * grid_params['cols'])
    grid_height = grid_params['cell_height'] * grid_params['rows']
    x_step = int(width / grid_params['cols'])
    y_step = int(height / grid_params['rows'])

    num_cells = grid_params['cols'] * grid_params['rows']
    
    # Compute average RGB value for each grid cell
    avg_rgb_values = []

    # Create an empty list to store the RGB values for each grid cell
    master_list = []

     # Create an empty numpy array to store the average color of each cell
    avg_rgb_colors = np.zeros((num_cells, 3))
    avg_hsv_colors = np.zeros((num_cells, 1))

    avg_hsv_values=[]

    # Loop over each cell in the grid
    cell_idx = 0

    for y in range(grid_params['rows']):
        for x in range(grid_params['cols']):
            x1 = x * x_step
            y1 = y * y_step
            x2 = min(x1 + x_step, width)
            y2 = min(y1 + y_step, height)
            
            grid_roi = frame[y1:y2, x1:x2]
            grid_roi_hsv= cv2.cvtColor(grid_roi, cv2.COLOR_BGR2HSV)
       

            avg_rgb_value = np.mean(grid_roi, axis=(0,1)).astype(np.uint8)

            # Convert average RGB value to HSV
            avg_hsv_value = cv2.cvtColor(np.array([[avg_rgb_value]]), cv2.COLOR_BGR2HSV)[0, 0]


            avg_rgb_values.append(avg_rgb_value)
            avg_hsv_values.append(avg_hsv_value)
            # avg_hsv_values.append(matplotlib.colors.rgb_to_hsv(avg_rgb_value))

            # print("RGB:",avg_rgb_value," \nHSV:",avg_hsv_value)


            # Store the average color in the array
            avg_rgb_colors[cell_idx] = avg_rgb_value
            avg_hsv_colors[cell_idx] = avg_hsv_value[0]
            
            cell_idx += 1

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            tm = os.path.basename(inputVideoFile).split('.')[0]
            pat = f'OutImgs/{tm}/{str(framNum)}'
            cv2.imwrite(f'{pat}/{cell_idx}.png', grid_roi)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), np.mean(grid_roi, axis=(0,1)),-1)
            
    # Draw average RGB value in each grid cell
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    for i, avg_rgb_value in enumerate(avg_rgb_values):
        x = (i % grid_params['cols']) * x_step
        y = (i // grid_params['cols']) * y_step + 10
        
        text = f"({avg_rgb_value[0]:.0f}, {avg_rgb_value[1]:.0f}, {avg_rgb_value[2]:.0f})"
        text_hsv = f"({avg_hsv_value[0]:.0f}, {avg_hsv_value[1]:.0f}, {avg_hsv_value[2]:.0f})"

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + (x_step - text_size[0]) // 2
        text_y = y + (y_step - text_size[1]) // 2 + text_size[1]
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

     # Convert the numpy array to a pandas dataframe and write it to a CSV file
    avg_rgb_colors_flat = np.array([','.join(map(str, cell)) for cell in avg_rgb_colors])
    avg_hsv_colors_flat = np.array([','.join(map(str, cell)) for cell in avg_hsv_colors])
    

    df = pd.DataFrame(data=[avg_hsv_colors_flat], columns=[f"cell_{i}" for i in range(num_cells)])


    if(framNum<=2):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file,mode='a', index=False, header=False)

                    

def process_video(yolo_bounding_box_file, inputVideoFile, loadYoloBoxes=True,loadContours=True):
    
    if(loadYoloBoxes):
        # Load YOLO bounding boxes
        data = load_yolo_bounding_boxes(yolo_bounding_box_file)
        
    # Load a video 
    cap = cv2.VideoCapture(inputVideoFile)
    # cap_optical_flow=cv2.VideoCapture(inputVideoFile + "_optical"+ inputVideoFileExtension)

    outputVideoFileName=inputVideoFile+"_output.mp4"

    size = (int(cap.get(3)), int(cap.get(4)))
    
    outputVideo = cv2.VideoWriter(outputVideoFileName, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         cap.get(cv2.CAP_PROP_FPS), size)
    

    number_of_videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameNum = 1

    ret, frame = cap.read()

    paused = False
    showRGB=False
    showOverlay=True

    grid_params = {'rows': 14, 'cols': 25, 'cell_width': 50, 'cell_height': 50}
    compflow = ComputeOpticalFLow(frame)

    while(cap.isOpened()):
        if not paused:
            ret, frame_rgb = cap.read()
            # ret2,frame_optical=cap_optical_flow.read()
            frame_optical = compflow.compute(frame_rgb)

            if not ret:
                break

            frameNum = frameNum + 1

        # Make folder
        tm = os.path.basename(inputVideoFile).split('.')[0]
        dir_path = f'OutImgs/{tm}/{str(frameNum)}'
        if not os.path.exists(dir_path): os.makedirs(dir_path)

        if showRGB: frame=frame_rgb
        else: frame=frame_optical

        print("\n\n frameNum: ",frameNum)

        if(loadYoloBoxes):
            # Draw YOLO bounding boxes
            selectedRow = data[data[:, 0] == frameNum]
            # print(selectedRow)

            if np.any(selectedRow):
                draw_yolo_bounding_box(frame, selectedRow)

        if(loadContours):
            # Load and Draw Segmented contours
            load_contours(inputVideoFile, frameNum,frame)
            

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('o'):
            showOverlay= not showOverlay
        
        if key== ord('t'):
            showRGB= not showRGB
        # elif key == ord('a'):  # Left arrow key
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)-2)
        # elif key == ord('d'):  # Right arrow key
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+1)

        if(showOverlay):
            overlayGridAndComputeAvgColor(frameNum,frame, grid_params, csv_file="rgb_values.csv", inputVideoFile=inputVideoFile)

        if not paused:
            outputVideo.write(frame)
        
        # cv2.imshow("Image", frame)

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # Add a boolean flag for noyolo
    parser.add_argument('--noyolo', action='store_false', help='do not load yolo bounding boxes')
    # Add a boolean flag for nocontour
    parser.add_argument('--nocontour', action='store_false', help='do not use contour detection')
    parser.add_argument("--path", required=True, help="Path to the input video")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the noyolo flag is set
    if args.noyolo: print('noyolo flag is set')
    else: print('noyolo flag is not set')

    yolo_bounding_box_file = "yolo_labels.txt"
    # inputVideoFile = "video_lq"
    # inputVideoFileExtension = ".mp4"
    
    process_video(yolo_bounding_box_file, args.path, args.noyolo, args.nocontour)

# python drawGridsAndOutputCSV.py --noyolo --nocontour --path video_lq.mp4
# python -W ignore .\color_kmeans.py -d OutImgs\video_lq\ -c 1 -f add.csv

# python -W ignore 