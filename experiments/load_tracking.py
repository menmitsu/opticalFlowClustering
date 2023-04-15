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
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.tools.detections import Detections
from onemetric.cv.utils.iou import box_iou_batch

OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                          "kcf": cv2.TrackerKCF_create,
                          # "boosting": cv2.TrackerBoosting_create,
                          "mil": cv2.TrackerMIL_create,
                          # "tld": cv2.TrackerTLD_create,
                          # "medianflow": cv2.TrackerMedianFlow_create,
                          # "mosse": cv2.TrackerMOSSE_create
                          }


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


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


def process_combined_img(frame):
    frame = bgr_erode(frame, iterations=2, kernal_size=9)
    frame = bgr_dilate(frame, iterations=2, kernal_size=12)
    return frame


def yolobbox2bbox(x, y, w, h):
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return x1, y1, x2, y2


def get_xyxy_bboxes(frame, area_threshold=5000, threshold_val=1):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        frame_gray, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xyxy_bboxes_list = []
    merge_bbox_mask = np.zeros_like(frame_gray)

    for contour in contours:
        if cv2.contourArea(contour) > 7000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = yolobbox2bbox(x, y, w, h)
            xyxy_bboxes_list.append(np.array([x1, y1, x2, y2], dtype=np.int))

    return xyxy_bboxes_list


def extract_flow_and_mask_video(input_video_path, model, output_video_path, process_optical_flow=False, mask_flow=True, show_img=False):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    framecount = 1
    output_video = None
    resolution = (1280, 720)
    window_size = 3
    mask = None

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), framerate, (1920, 360))

    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask)
    compflow = ComputeOpticalFLow(firstframe)
    sliding_img_window = np.ones(
        (window_size, 720, 1280, 3), dtype=np.uint8)*255

    byte_tracker = BYTETracker(BYTETrackerArgs())
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # multi_object_cv_trackers = cv2.MultiTracker_create()
    # cv_tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
    cv_trackers = {}
    tracker_id_list = set()
    while success:
        success, frame = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff
        frame = process_frame(frame, resolution, mask)

        # fgmask = fgbg.apply(frame)
        # fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)[1]
        opflowimg = compflow.compute(frame)

        if mask_flow and model is not None:
            masks = get_masks([frame], model)[0]
            cv2.drawContours(opflowimg, masks, -1, (0, 0, 0), cv2.FILLED)
            # cv2.drawContours(fgmask, masks, -1, (0, 0, 0), cv2.FILLED)

        if process_optical_flow:
            opflowimg = process_flow(opflowimg)

        # cv2.imshow('bgs', fgmask)
        sliding_window(sliding_img_window, opflowimg)
        combined_img = combine_images(sliding_img_window)
        combined_img = process_combined_img(combined_img)
        xyxy_bboxes = get_xyxy_bboxes(combined_img)

        for tracker_id in list(cv_trackers.keys()):
            # deleting trackers which have not been updated for a long time
            if framecount - cv_trackers[tracker_id]['last_updated'] > 25:
                del cv_trackers[tracker_id]

        for tracker_id in cv_trackers:
            (tracker_success, bbox) = cv_trackers[tracker_id]['tracker'].update(
                frame)
            if tracker_success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), thickness=2)
                cv2.rectangle(combined_img, p1, p2, (0, 0, 255), thickness=2)

        detections = Detections(
            xyxy=np.asarray(xyxy_bboxes).reshape(-1, 4),
            confidence=np.ones(len(xyxy_bboxes), dtype=np.float),
            class_id=np.zeros(len(xyxy_bboxes), dtype=np.int)
        )
        tracks = byte_tracker.update(output_results=detections2boxes(
            detections=detections), img_info=frame.shape, img_size=frame.shape)
        tracker_id = match_detections_with_tracks(
            detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        tracker_mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=tracker_mask, inplace=True)

        for xyxy, confidence, class_id, tracker_id in detections:
            if tracker_id is None:
                continue
            cv_trackers[tracker_id] = {
                'tracker': OPENCV_OBJECT_TRACKERS['kcf'](), 'last_updated': framecount}
            x1, y1, x2, y2 = xyxy
            cv_trackers[tracker_id]['tracker'].init(
                frame, (x1, y1, x2-x1, y2-y1))

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
