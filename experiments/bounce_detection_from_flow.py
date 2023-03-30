from typing import List
import faiss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import argparse
import os
import pandas as pd
import matplotlib
from flow_masking_and_bounce_detection import process_flow
from flow_masking_and_bounce_detection import detect_bounce_pattern
from flow_masking_and_bounce_detection import sliding_window

matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def detect_bounce_from_flow(input_video_path, output_video_path, show_img=False):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 0
    output_video = None
    bounce_detected = False
    success = True
    scale_value = 20

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        # cv2.VideoWriter_fourcc('a', 'v', 'c', '1') # NOTE: This VideoWriter may not work in linux environments
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), framerate, (1280, 640))

    window_size = 15
    bgr_values = [np.zeros(window_size, dtype=np.float32), np.zeros(
        window_size, dtype=np.float32), np.zeros(window_size, dtype=np.float32)]

    if show_img or output_video:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    while success:
        success, opflowimg = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff

        opflowimg = process_flow(opflowimg)

        b, g, r = cv2.split(opflowimg)
        b_mean = np.mean(b)*scale_value
        g_mean = np.mean(g)*scale_value
        r_mean = np.mean(r)*scale_value
        sliding_window(bgr_values[0], b_mean)
        sliding_window(bgr_values[1], g_mean)
        sliding_window(bgr_values[2], r_mean)

        assert len(bgr_values[0]) == len(bgr_values[1]) and len(
            bgr_values[1]) == len(bgr_values[2])

        bounce_detected_in_window = detect_bounce_pattern(bgr_values)
        bounce_detected = bounce_detected | bounce_detected_in_window

        if show_img or output_video:
            if bounce_detected_in_window:
                cv2.putText(opflowimg, "BOUNCE DETECTED!", (200, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)

            cv2.putText(opflowimg, "r %2.f, " % r_mean + "g %2.f, " % g_mean + "b %2.f" %
                        b_mean, (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)
            ax.clear()
            ax.plot(bgr_values[0], color='blue')
            ax.plot(bgr_values[1], color='green')
            ax.plot(bgr_values[2], color='red')
            ax.set_ylim(0, 200)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            opflowimg = cv2.resize(opflowimg, (640, 640))
            img = cv2.resize(img, (640, 640))
            output_frame = cv2.hconcat([opflowimg, img])

            if show_img:
                if key == ord('p'):  # Use 'p' to pause and resume the video playback
                    while True:
                        key2 = cv2.waitKey(1) or 0xff
                        cv2.imshow('Win', output_frame)
                        if key2 == ord('p') or key2 == 27:
                            break

                cv2.imshow('Win', output_frame)

            if output_video is not None:
                output_video.write(output_frame)

            if key == 27:
                break

        framecount = framecount + 1

    cap.release()
    if output_video is not None:
        print("Output video released.")
        output_video.release()

    return bounce_detected


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input flow video and detects bounce violations.')

    parser.add_argument('--input_video', type=str, default='', const="",
                        nargs='?', help='Input video path.')

    parser.add_argument('--output', type=str, default=None, const="",
                        nargs='?', help='Output video path')

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    return parser.parse_args()


def main(args):
    print(detect_bounce_from_flow(input_video_path=args.input_video,
                                  output_video_path=args.output, show_img=args.show_img))


if __name__ == '__main__':
    args = get_arguments()
    main(args)
