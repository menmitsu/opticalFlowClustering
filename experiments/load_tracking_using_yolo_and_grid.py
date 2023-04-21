import time
import pandas as pd
from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import argparse
import os
from flow_masking_and_bounce_detection import process_frame
from flow_masking_and_bounce_detection import ComputeOpticalFLow
from flow_masking_and_bounce_detection import play_pause
from flow_masking_and_bounce_detection import hconcat_frames
import norfair
from norfair import Detection, Tracker, get_cutout
from norfair.filter import FilterPyKalmanFilterFactory
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DISTANCE_THRESHOLD_BBOX: float = 1
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                          "kcf": cv2.TrackerKCF_create,
                          "mil": cv2.TrackerMIL_create,
                          # "boosting": cv2.TrackerBoosting_create,
                          # "tld": cv2.TrackerTLD_create,
                          # "medianflow": cv2.TrackerMedianFlow_create,
                          # "mosse": cv2.TrackerMOSSE_create
                          }


class BboxVelocity:
    def __init__(self, x, y, dt=1):
        self.x = x / dt  # component along x direction
        self.y = y / dt  # component along y direction
        self.magnitude = (x*x + y*y)**0.5
        self.dt = dt

    def str(self):
        return f'{self.x}, {self.y}, {self.magnitude:.2f}'


@torch.no_grad()
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


def center(points):
    return [np.mean(np.array(points), axis=0)]


def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(
            snd_embedding, detection_fst.embedding, cv2.HISTCMP_CORREL
        )
        if distance < 0.5:
            return distance
    return 1


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        for boxes in yolo_detections:
            centroid = np.array(
                [boxes.xywh[0].item(), boxes.xywh[1].item()]
            )
            scores = np.array([boxes.conf[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(boxes.cls[-1].item()),
                )
            )

    elif track_points == "bbox":
        for boxes in yolo_detections:
            bbox = np.array(
                [
                    [boxes.xyxy[0][0].item(), boxes.xyxy[0][1].item()],
                    [boxes.xyxy[0][2].item(), boxes.xyxy[0][3].item()],
                ]
            )
            scores = np.array(
                [boxes.conf.item(), boxes.conf.item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(boxes.cls[-1].item())
                )
            )

    return norfair_detections


def mask_frame_using_grids(opflowimg, frame, grid_n=20, threshold_val=20):
    height, width = opflowimg.shape[:2]
    M = height // grid_n
    N = width // grid_n
    grids = []

    opflowimg_gray = cv2.cvtColor(opflowimg, cv2.COLOR_BGR2GRAY)
    _, opflowimg_threshold = cv2.threshold(
        opflowimg_gray, threshold_val, 255, cv2.THRESH_BINARY)

    mask = np.ones_like(frame)*255
    masked_frame = frame.copy()

    for y in range(0, height, M):
        for x in range(0, width, N):
            y1 = y + M
            x1 = x + N
            tile = opflowimg_threshold[y:y+M, x:x+N]
            if cv2.countNonZero(tile) == 0:
                masked_frame[y:y+M, x:x+N] = (0, 0, 0)
                mask[y:y+M, x:x+N] = (0, 0, 0)

    return masked_frame, mask


def centroid_xywh_bbox(bbox):
    # find centroid for bbox in x, y, w, h format
    return [(2*bbox[0] + bbox[2]) // 2, (2*bbox[1] + bbox[3]) // 2]


def calculate_velocity(current_bbox_dict, prev_bbox_dict):
    velocity_dict = {}
    for tracker_id in current_bbox_dict:
        if tracker_id in prev_bbox_dict:
            current_centroid = centroid_xywh_bbox(
                current_bbox_dict[tracker_id]['bbox'])
            prev_centroid = centroid_xywh_bbox(
                prev_bbox_dict[tracker_id]['bbox'])
            velocity_dict[tracker_id] = BboxVelocity(
                x=current_centroid[0] - prev_centroid[0], y=current_centroid[1] - prev_centroid[1])

    return velocity_dict


def load_tracking_using_yolo_and_grid(input_video_path: str, model, conf_limit=0.1, output_video_path: str = None, track_points='bbox', show_img=False) -> int:
    cap = cv2.VideoCapture(input_video_path)
    success = True
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = None
    resolution = (640, 640)
    framecount = 0

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), framerate, (1920, 640))  # Note: This codec may not work on Linux systems.

    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask=None)
    compflow = ComputeOpticalFLow(firstframe)
    cv_trackers = {}
    distance_function = "iou" if track_points == "bbox" else "euclidean"
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        initialization_delay=4,
        hit_counter_max=5,
        filter_factory=FilterPyKalmanFilterFactory(R=8.0, Q=0.08),
        distance_threshold=distance_threshold,
        past_detections_length=15,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=1,
        reid_hit_counter_max=500,
    )

    prev_bbox_dict = {}
    max_speed = 0
    max_x = 0
    max_y = 0
    rough_throw_detected = False
    skip_frames = 0

    while success:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, resolution, mask=None)
        opflowimg = compflow.compute(frame)
        masked_frame, grid_mask = mask_frame_using_grids(
            opflowimg, frame, grid_n=10, threshold_val=5)

        results = model(masked_frame, verbose=False)
        filtered_conf_idx = np.array(
            results[0].boxes.conf.cpu().numpy() >= conf_limit)
        results[0].boxes = results[0].boxes[filtered_conf_idx]

        for tracker_id in list(cv_trackers.keys()):
            # Deleting trackers that have not been updated in a while
            if framecount - cv_trackers[tracker_id]['last_updated'] > 3:
                del cv_trackers[tracker_id]

        bbox_dict = {}
        for tracker_id in cv_trackers:
            (tracker_success, bbox) = cv_trackers[tracker_id]['tracker'].update(
                frame)
            if tracker_success:
                bbox_dict[tracker_id] = {'bbox': bbox, 'pred': True}

        detections = yolo_detections_to_norfair_detections(
            results[0].boxes, track_points=track_points)

        # Skip frames when too many bboxes are detected.
        if len(detections) > 3:
            skip_frames = 5

        """
        Improves results when detections are of different colors. May not work well for our case.

        for detection in detections:
            for detection in detections:
                cut = get_cutout(detection.points, frame)
                if cut.shape[0] > 0 and cut.shape[1] > 0:
                    detection.embedding = get_hist(cut)
                else:
                    detection.embedding = None
        """

        tracked_objects = tracker.update(detections=detections)
        for obj in tracked_objects:
            bbox = obj.estimate
            height, width = frame.shape[:2]
            bbox_color = (0, 255, 0)
            p1 = (min(int(bbox[0][0]), width-1),
                  min(int(bbox[0][1]), height-1))
            p2 = (min(int(bbox[1][0]), width-1),
                  min(int(bbox[1][1]), height-1))
            w = int(p2[0] - p1[0])
            h = int(p2[1] - p1[1])
            tracker_id = int(obj.id)
            bbox_dict[tracker_id] = {
                'bbox': [p1[0], p1[1], w, h], 'pred': False}
            if w <= 5 or h <= 5:  # Skipping tracker initialization for bounding boxes, which are very small. Having a very small width or height could also raise an exception from OPENCV_TRACKERS
                continue
            cv_trackers[tracker_id] = {
                'tracker': OPENCV_OBJECT_TRACKERS['csrt'](), 'last_updated': framecount}
            cv_trackers[tracker_id]['tracker'].init(
                frame, (p1[0], p1[1], w, h))

        """
        Use this when using bytetracker or bot-sort from the ultralytics library

        for box in results[0].boxes:
            if box.is_track:
                tracker_id = int(box.id)
                cv_trackers[tracker_id] = {
                    'tracker': OPENCV_OBJECT_TRACKERS['csrt'](), 'last_updated': framecount}
                [x1, y1, x2, y2] = [int(i) for i in box.xyxy[0]]
                bbox_dict[tracker_id] = {
                    'bbox': [x1, y1, x2-x1, y2-y1], 'pred': False}
                cv_trackers[tracker_id]['tracker'].init(
                    frame, (x1, y1, x2-x1, y2-y1))
        """

        velocity_dict = calculate_velocity(bbox_dict, prev_bbox_dict)
        prev_bbox_dict = bbox_dict.copy()
        rough_throw_detected_in_frame = False

        for tracker_id in bbox_dict:
            bbox = bbox_dict[tracker_id]['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            w = int(p2[0] - p1[0])
            h = int(p2[1] - p1[1])

            if w <= 5 or h <= 5:  # Skipping bounding boxes, which are very small.
                continue

            bbox_color = (0, 0, 255)  # red bboxes are bboxes from yolo model
            if bbox_dict[tracker_id]['pred']:
                # blue bboxes are predicted bboxes from tracker (since no bbox was generated for this tracker_id by yolo model.)
                bbox_color = (255, 0, 0)
            cv2.putText(frame, str(
                tracker_id), (p1[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)

            if tracker_id in velocity_dict:
                max_speed = max(max_speed, velocity_dict[tracker_id].magnitude)
                max_x = max(abs(max_x), velocity_dict[tracker_id].x)
                max_y = max(abs(max_y), velocity_dict[tracker_id].y)
                if velocity_dict[tracker_id].magnitude >= 31 and skip_frames == 0:
                    rough_throw_detected = True
                    rough_throw_detected_in_frame = True
                    # green bboxes are bboxes for which a violation was detected.
                    bbox_color = (0, 255, 0)
                    cv2.putText(frame, 'Rough Throw Detected!', (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)

                cv2.putText(frame, velocity_dict[tracker_id].str(
                ), (p2[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255,), 2, 2)
            cv2.rectangle(frame, p1, p2, bbox_color, thickness=2)

        print(f'{max_x}, {max_y}, {max_speed}')
        if show_img or output_video is not None:
            key = cv2.waitKey(1) & 0xff
            res_plot = results[0].plot()
            cv2.putText(frame, f'{max_x}, {max_y}, {max_speed:.2f}', (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255,), 4, 2)
            output_frame = hconcat_frames([frame, res_plot, opflowimg])
            if output_video is not None:
                output_video.write(output_frame)
            if show_img and play_pause(output_frame, key) == False:
                break
            if rough_throw_detected_in_frame:
                cv2.waitKey(2000)

        framecount = framecount + 1
        if skip_frames > 0:
            skip_frames = skip_frames - 1

    cap.release()
    if output_video is not None:
        output_video.release()
    return rough_throw_detected


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input video and detects cfl throw violation.')

    """For running on a single video, use --input_video and --output_video flags."""

    parser.add_argument('--input_video', type=str, default=None, const="",
                        nargs='?', help='Input video path.')

    parser.add_argument('--output_video', type=str, default=None, const="",
                        nargs='?', help='Output video path. Use None to not generate a output video.')

    """For running on multiple videos, use --input_dir and --output_dir flags."""

    parser.add_argument('--input_dir', type=str, default=None, const="",
                        nargs='?', help='Input video directory.')

    parser.add_argument('--output_dir', type=str, default=None, const="",
                        nargs='?', help='Output video directory. Use None to not generate the output videos.')

    parser.add_argument('--model_path', type=str, default='', const="",
                        nargs='?', help='Path to yolov8 model.')

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    return parser.parse_args()


def main(args):
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO(args.model_path)
    model.fuse()

    if args.input_video is not None:
        print(load_tracking_using_yolo_and_grid(input_video_path=args.input_video,
                                                model=model, output_video_path=args.output_video, show_img=args.show_img))

    elif args.input_dir is not None:
        if args.output_dir is not None:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)

        filename_list = []
        rough_throw_detected_list = []

        for file in os.listdir(args.input_dir):
            if(file.endswith('.mp4')):
                filename = os.path.splitext(file)[0]
                input_video_path = os.path.join(args.input_dir, file)
                output_video_path = None
                if args.output_dir is not None:
                    output_video_path = os.path.join(
                        args.output_dir, filename + ".mp4")

                print("Processing: ", input_video_path)
                rough_throw_detected = load_tracking_using_yolo_and_grid(
                    input_video_path=input_video_path, model=model, output_video_path=output_video_path)

                filename_list.append(file)
                rough_throw_detected_list.append(
                    1 if rough_throw_detected else 0)
                print("Done!")

        print("Processed all files!")
        print("Outputs: ", rough_throw_detected_list)
        if args.output_dir is not None:
            output_results_csv_path = os.path.join(
                args.output_dir, 'results.csv')
            results_df = pd.DataFrame(
                {"filename": filename_list, "rough_throw_detected": rough_throw_detected_list})
            results_df.to_csv(output_results_csv_path, index=False)
            print('Generated results.csv!')


if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
