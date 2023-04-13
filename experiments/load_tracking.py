from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import argparse
import os
import pandas as pd
from flow_masking_and_bounce_detection import get_masks
from flow_masking_and_bounce_detection import process_frame
from flow_masking_and_bounce_detection import ComputeOpticalFLow
from flow_masking_and_bounce_detection import hconcat_frames
from flow_masking_and_bounce_detection import process_flow
from flow_masking_and_bounce_detection import play_pause
from flow_masking_and_bounce_detection import sliding_window
from flow_masking_and_bounce_detection import bgr_dilate
from flow_masking_and_bounce_detection import bgr_erode


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


def combine_images(imgs):
    combined_image = np.min(imgs, axis=0).astype(np.uint8)
    return combined_image


def process_combined_img(frame, og_frame, threshold_val=1):
    frame = bgr_erode(frame, iterations=2, kernal_size=9)
    frame = bgr_dilate(frame, iterations=2, kernal_size=15)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        frame_gray, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding rectangle
            cv2.rectangle(og_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame


def extract_flow_and_mask_video(input_video_path, model, output_video_path, process_optical_flow=False, mask_flow=True, show_img=False):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 1
    output_video = None
    resolution = (1280, 720)
    window_size = 4
    mask = None

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), framerate, (1920, 360))

    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask)
    compflow = ComputeOpticalFLow(firstframe)
    sliding_img_window = np.ones(
        (window_size, 720, 1280, 3), dtype=np.uint8)*255

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

        if process_optical_flow:
            opflowimg = process_flow(opflowimg)

        sliding_window(sliding_img_window, opflowimg)
        combined_img = combine_images(sliding_img_window)
        combined_img = process_combined_img(combined_img, frame)

        if show_img or output_video:
            output_frame = hconcat_frames(
                [frame, opflowimg, combined_img], resolution=(640, 360))

            if output_video is not None:
                output_video.write(output_frame)

            if show_img and play_pause(output_frame, key) == False:
                break

        framecount = framecount + 1

    cap.release()
    if output_video is not None:
        print("Output video released.")
        output_video.release()


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input directory containing .mp4 videos and extracts the masked flow for each video.')

    """For running on a single video, use --input_video and --output_video flags."""

    parser.add_argument('--input_video', type=str, default='', const="",
                        nargs='?', help='Input video path.')

    parser.add_argument('--output_video', type=str, default=None, const="",
                        nargs='?', help='Output video path. Use default value of None to not generate a output video.')

    """For running on a number of videos, use --input_dir and --output_dir flags."""

    parser.add_argument('--input_dir', type=str, default=None, const="",
                        nargs='?', help='Input folder path with all the videos.')

    parser.add_argument('--output_dir', type=str, default=None, const="",
                        nargs='?', help='Output folder path for the extracted flow videos. Use default value of None to not generate output videos.')

    parser.add_argument("--mask_flow", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether to mask optical flow using yolo segmentation model.")

    parser.add_argument("--process_optical_flow", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to process extracted optical flow.")

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    return parser.parse_args()


def main(args):
    model = None
    if args.mask_flow:
        print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = YOLO('yolov8x-seg.pt')
        model.fuse()

    if args.input_dir is not None:
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
                                            model=model, output_video_path=output_video_path, process_optical_flow=args.process_optical_flow, mask_flow=args.mask_flow, show_img=args.show_img)
                print("Done!")

        print("Processed all files!")

    else:
        extract_flow_and_mask_video(input_video_path=args.input_video, model=model, output_video_path=args.output_video,
                                    process_optical_flow=args.process_optical_flow, mask_flow=args.mask_flow, show_img=args.show_img)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
