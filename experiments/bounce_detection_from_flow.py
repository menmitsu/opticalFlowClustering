from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import pandas as pd
from flow_masking_and_bounce_detection import process_flow
from flow_masking_and_bounce_detection import detect_bounce_pattern
from flow_masking_and_bounce_detection import sliding_window
from flow_masking_and_bounce_detection import plot_graphs_and_create_img
from flow_masking_and_bounce_detection import hconcat_frames
from flow_masking_and_bounce_detection import play_pause


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
    combined_image = np.zeros(imgs[0].shape, dtype=np.uint8)
    for img in imgs:
        combined_image = np.maximum(combined_image, img)
    return combined_image


def process_combined_img(frame, threshold_val=5):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        frame_gray, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame

def detect_bounce_from_flow(input_video_path, output_video_path, show_img=False, scale_value=20, process_optical_flow=True):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 0
    output_video = None
    bounce_detected = False
    success = True

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), framerate, (1280, 640))  # NOTE: This VideoWriter may not work in linux environments

    window_size = 15
    bgr_values = [np.zeros(window_size, dtype=np.float32) for _ in range(3)]
    sliding_img_window = np.zeros((window_size, 640, 640, 3), dtype=np.uint8)

    if show_img or output_video:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    while success:
        success, opflowimg = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff
        if process_optical_flow:
            opflowimg = process_flow(opflowimg)

        """
        sliding_window(sliding_img_window, cv2.resize(opflowimg, (640, 640)))
        combined_img = combine_images(sliding_img_window)
        process_flow(combined_img)
        process_combined_img(combined_img)
        cv2.imwrite('test_combined/combined_img_' +
                    str(framecount) + ".png", combined_img)
        """

        for i in range(3):
            color_channel = opflowimg[:, :, i].mean()
            color_channel = color_channel*scale_value
            sliding_window(bgr_values[i], color_channel)

        bounce_detected_in_window = detect_bounce_pattern(bgr_values)
        bounce_detected = bounce_detected | bounce_detected_in_window

        if show_img or output_video:
            img = plot_graphs_and_create_img(ax, fig, bgr_values)
            output_frame = hconcat_frames([opflowimg, img])

            if show_img and play_pause(output_frame, key) == False:
                break

            if output_video is not None:
                output_video.write(output_frame)

        framecount = framecount + 1

    cap.release()
    if output_video is not None:
        print("Output video released.")
        output_video.release()

    return bounce_detected


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input flow video and detects bounce violations.')

    parser.add_argument('--input_video', type=str, default=None, const="",
                        nargs='?', help='Input optical flow video path.')

    parser.add_argument('--output_video', type=str, default=None, const="",
                        nargs='?', help='Output video path')

    parser.add_argument('--input_dir', type=str, default=None, const="",
                        nargs='?', help='Input folder containing mp4 optical flow videos.')

    parser.add_argument('--output_dir', type=str, default=None, const="",
                        nargs='?', help='Output folder path.')

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    parser.add_argument("--process_optical_flow", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    return parser.parse_args()


def main(args):
    if args.input_video is not None:
        print(detect_bounce_from_flow(input_video_path=args.input_video,
                                      output_video_path=args.output_video, show_img=args.show_img, process_optical_flow=args.process_optical_flow))

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)

    preds = []
    video_names = []

    if args.input_dir is not None:
        for subdir, dirs, files in os.walk(args.input_dir):
            for file in files:
                filepath = os.path.join(subdir, file)
                filename = os.path.splitext(os.path.basename(filepath))[0]
                if filename[-7:] == "optical" and filepath.endswith('.mp4'):
                    dirname = os.path.dirname(filepath)
                    output_filename = os.path.basename(
                        dirname) + "_" + filename
                    output_video_path = None

                    print('Processing file: ', filepath)
                    if args.output_dir is not None:
                        output_video_path = os.path.join(
                            args.output_dir, output_filename + "_out.mp4")
                        print('Output at: ', output_video_path)

                    pred = detect_bounce_from_flow(
                        input_video_path=filepath, output_video_path=output_video_path, show_img=args.show_img, scale_value=1, process_optical_flow=args.process_optical_flow)

                    video_names.append(output_filename)
                    preds.append(1 if pred else 0)
                    print('Done!')

        print('Processed all files!')
        print(preds)
        if args.output_dir is not None:
            output_results_csv_path = os.path.join(
                args.output_dir, 'results.csv')
            results_df = pd.DataFrame(
                {"video_name": video_names, "preds": preds})
            results_df.to_csv(output_results_csv_path, index=False)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
