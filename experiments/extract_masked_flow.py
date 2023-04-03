from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import argparse
import os
import pandas as pd
from flow_masking_and_bounce_detection import get_masks
from flow_masking_and_bounce_detection import create_mask
from flow_masking_and_bounce_detection import process_frame
from flow_masking_and_bounce_detection import ComputeOpticalFLow


def str2bool(v: str):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises error if v is
    anything else.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_flow_and_mask_video(input_video_path, model, output_video_path, process_flow=False, mask_flow=True):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 1
    output_video = None
    resolution = (1280, 720)
    mask = create_mask(resolution=resolution)

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), framerate, resolution)

    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask)
    compflow = ComputeOpticalFLow(firstframe)

    while success:
        success, frame = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff
        frame = process_frame(frame, resolution, mask)
        opflowimg = compflow.compute(frame)

        if mask_flow and model is not None:
            masks = get_masks([frame], model)[0]
            cv2.drawContours(opflowimg, masks, -1, (0, 0, 0), cv2.FILLED)

        if process_flow:
            opflowimg = process_flow(opflowimg)

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

    parser.add_argument("--mask_flow", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether to mask output optical flow using yolo segmentation model.")

    parser.add_argument("--process_optical_flow", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to process extracted optical flow.")

    return parser.parse_args()


def main(args):
    model = None
    if args.mask_flow:
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
            extract_flow_and_mask_video(input_video_path=input_video_path,
                                        model=model, output_video_path=output_video_path, process_flow=args.process_optical_flow, mask_flow=args.mask_flow)
            print("Done!")

    print("Processed all files!")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
