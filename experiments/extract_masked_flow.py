from typing import List
import faiss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import argparse
import os
import pandas as pd
import matplotlib
from flow_masking_and_bounce_detection import get_masks
matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ComputeOpticalFLow:
    def __init__(self, firstframe):
        self.firstframe = firstframe
        self.width = self.firstframe.shape[1]
        self.height = self.firstframe.shape[0]

        self.outputImg = np.zeros([self.height, 2*self.width, 3],
                                  dtype=self.firstframe.dtype)
        self.mask = np.zeros_like(self.firstframe)
        self.mask[..., 1] = 255
        self.prev_gray = cv2.cvtColor(self.firstframe, cv2.COLOR_BGR2GRAY)

    def compute(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray,
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        self.mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        self.mask[..., 2] = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(self.mask, cv2.COLOR_HSV2BGR)
        self.prev_gray = gray

        return rgb
        

def compute_flow_and_mask_video(input_video_path, model, output_video_path, show_img=False):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 1
    output_video = None
    resolution = (1280, 720)
    bbox = ((0, 90), (640, 729))
    mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    cv2.rectangle(mask, bbox[0], bbox[1], (255, 255, 255), -1)

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        # cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), framerate, resolution)

    success, firstframe = cap.read()
    firstframe = cv2.resize(firstframe, resolution)
    firstframe = cv2.bitwise_and(firstframe, firstframe, mask=mask)

    compflow = ComputeOpticalFLow(firstframe)

    while success:
        success, frame = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff

        frame = cv2.resize(frame, resolution)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        opflowimg = compflow.compute(frame)

        masks = get_masks([frame], model)[0]
        for ctr in masks:
            cv2.drawContours(opflowimg, [ctr], -1, (0, 0, 0), cv2.FILLED)

        if show_img:
            cv2.imshow('Win', opflowimg)

        if output_video is not None:
            output_video.write(opflowimg)

        framecount = framecount + 1

    cap.release()
    if output_video is not None:
        print("Output video released.")
        output_video.release()


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input directory containing .mp4 videos and extracts the masked flow for each video.')

    parser.add_argument('--input_dir', type=str, default='', const="",
                        nargs='?', help='Input folder path with all the videos.')

    parser.add_argument('--output_dir', type=str, default=None, const="",
                        nargs='?', help='Output folder path for the extracted flow videos.')

    return parser.parse_args()


def main(args):
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO('yolov8x-seg.pt')
    model.fuse()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for file in os.listdir(args.input_dir):
        if(file.endswith('.mp4')):
            filename = os.path.splitext(file)[0]
            input_video_path = os.path.join(args.input_dir, file)
            output_video_path = os.path.join(
                args.output_dir, filename + "_flow.mp4")

            print("Processing: ", input_video_path)
            compute_flow_and_mask_video(input_video_path=input_video_path,
                                        model=model, output_video_path=output_video_path, show_img=False)
            print("Done!")

    print("Processed all files!")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
