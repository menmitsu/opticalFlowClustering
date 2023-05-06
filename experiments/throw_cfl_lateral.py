import cv2
from throw_cfl import BboxVelocity, centroid_in_roi, centroid_xywh_bbox
from throw_cfl import embedding_distance, get_hist, mask_frame_using_grids
from throw_cfl import yolo_detections_to_norfair_detections, str2bool, centroid_in_roi
import json
from eval_metric import NumpyEncoder
import time
import pandas as pd
from typing import List
from ultralytics import YOLO
import numpy as np
import torch
import argparse
import os
from flow_masking_and_bounce_detection import process_frame, ComputeOpticalFLow
from flow_masking_and_bounce_detection import play_pause, hconcat_frames
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


def centroid_xyxy_bbox(bbox):
    # find centroid for bbox in format: x-top-left, y-top-left, x-bottom-right, y-bottom-right
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]


def get_human_bbox(human_detection_model, frame, roi=[233, 387, 397, 639], detections_limit=2):
    human_detection_result = human_detection_model(frame, verbose=False)
    bboxes = human_detection_result[0].boxes[human_detection_result[0].boxes.cls == 0]
    human_bbox = None
    bbox_cnt = 0
    if len(bboxes) > detections_limit:  # Too many person detected.
        return None

    for bbox in bboxes:
        if centroid_in_roi(centroid_xyxy_bbox(bbox.xyxy[0]), roi):
            human_bbox = bbox.xyxy.cpu().numpy()[0]
            bbox_cnt = bbox_cnt + 1

    if bbox_cnt == 1:
        return human_bbox
    return None


def throw_cfl_lateral(input_video_path: str, model, output_video_path: str = None, track_points='bbox', show_img=False, export_trajectory=False, threshold_distance=80) -> int:
    cap = cv2.VideoCapture(input_video_path)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = None
    resolution = (640, 640)
    framecount = 0
    human_detection_model = YOLO('yolov8x.pt')
    human_detection_model.fuse()

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), framerate, (1280, 640))  # Note: This codec may not work on Linux systems.

    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask=None)
    compflow = ComputeOpticalFLow(firstframe)
    tracker = Tracker(
        distance_function="iou" if track_points == "bbox" else "euclidean",
        initialization_delay=4,
        hit_counter_max=5,
        filter_factory=FilterPyKalmanFilterFactory(R=16.0, Q=0.04),
        distance_threshold=(DISTANCE_THRESHOLD_BBOX if track_points == "bbox"
                            else DISTANCE_THRESHOLD_CENTROID),
        past_detections_length=15,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=1,
        reid_hit_counter_max=500,
    )
    cv_trackers = {}
    prev_bbox_dict = {}
    min_x, max_x, min_y, max_y, min_z, max_z, max_speed = 0, 0, 0, 0, 0, 0, 0
    rough_throw_detected = False
    skip_frames = 0
    min_w, min_h = 15, 15
    trajectory_dict = {} if export_trajectory else None

    while success:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, resolution, mask=None)
        opflowimg = compflow.compute(frame)
        masked_frame, _ = mask_frame_using_grids(
            opflowimg, frame, grid_n=10, threshold_val=20)

        results = model(masked_frame, verbose=False)

        for box in results[0].boxes:  # Skipping frames when a shrink bag is detected
            if model.names[int(box.cls)] == 'shrink' and not centroid_in_roi(centroid_xyxy_bbox(box.xyxy[0]), roi=[181, 550, 350, 639]):
                skip_frames = 25
                print(f'Shrink bag detected, skipping next {skip_frames} frames.')
                break

        for tracker_id in list(cv_trackers.keys()):
            # Deleting trackers that have not been updated in a while
            if framecount - cv_trackers[tracker_id]['last_updated'] > 3:
                del cv_trackers[tracker_id]

        bbox_dict = {}
        for tracker_id in cv_trackers:
            (tracker_success, bbox) = cv_trackers[tracker_id]['tracker'].update(
                frame)

            if tracker_success:
                # bbox format: [x_top_left, y_top_left, width, height]
                bbox_dict[tracker_id] = {'bbox': bbox,
                                         'pred': True, 'z_depth': 0}

        detections = yolo_detections_to_norfair_detections(
            results[0].boxes, track_points=track_points)

        # Skip frames when too many bboxes are detected.
        if len(detections) != 1:
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
            p1 = (min(int(bbox[0][0]), width-1),
                  min(int(bbox[0][1]), height-1))
            p2 = (min(int(bbox[1][0]), width-1),
                  min(int(bbox[1][1]), height-1))
            w = int(p2[0] - p1[0])
            h = int(p2[1] - p1[1])

            tracker_id = int(obj.id)
            bbox_dict[tracker_id] = {
                'bbox': [p1[0], p1[1], w, h], 'pred': False, 'z_depth': 0}

            if w <= min_w or h <= min_h:  # Skipping tracker initialization for bounding boxes, which are very small. Having a very small width or height could also raise an exception from OPENCV_TRACKERS
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

        # velocity_dict = calculate_velocity(bbox_dict, prev_bbox_dict)
        # prev_bbox_dict = bbox_dict.copy()
        rough_throw_detected_in_frame = False
        human_xyxy_bbox = None

        if skip_frames == 0:
            human_xyxy_bbox = get_human_bbox(
                human_detection_model=human_detection_model, frame=frame)

        for tracker_id in bbox_dict:
            bbox = bbox_dict[tracker_id]['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            w = bbox[2]
            h = bbox[3]

            # red bboxes are bboxes from norfair tracker
            bbox_color = (0, 0, 255)
            if bbox_dict[tracker_id]['pred']:
                # blue bboxes are predicted bboxes from OPENCV_TRACKERS (since no bbox was generated for this tracker_id by norfair.)
                bbox_color = (255, 0, 0)

            # Skipping bounding boxes, which are very small
            if not (skip_frames > 0 or w <= min_w or h <= min_h):
                if trajectory_dict is not None:
                    if tracker_id not in trajectory_dict:
                        trajectory_dict[tracker_id] = {
                            'pred': 0, 'trajectory': []}
                    trajectory_dict[tracker_id]['trajectory'].append(bbox)

                if human_xyxy_bbox is not None:
                    centroid = centroid_xyxy_bbox(human_xyxy_bbox)
                    load_centroid = centroid_xywh_bbox(bbox)
                    distance = abs(centroid[0] - load_centroid[0])
                    if distance >= threshold_distance and load_centroid[1] >= 245 and w*h <= 25000 and not centroid_in_roi(centroid_xywh_bbox(bbox), roi=[181, 550, 350, 639]):
                        rough_throw_detected_in_frame = True
                        rough_throw_detected = True
                        print('Lateral Rough Throw Detected: ', distance)
                        print('BBox area: ', w * h)
                        cv2.putText(frame, 'Rough Throw Detected!', (5, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)

            cv2.putText(frame, str(tracker_id), (p1[0], p1[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255,), 4, 2)
            cv2.rectangle(frame, p1, p2, bbox_color, thickness=2)
            if human_xyxy_bbox is not None:
                human_p1 = (int(human_xyxy_bbox[0]), int(human_xyxy_bbox[1]))
                human_p2 = (int(human_xyxy_bbox[2]), int(human_xyxy_bbox[3]))
                cv2.rectangle(frame, human_p1, human_p2,
                              (0, 255, 255), thickness=2)

        # print(f'{min_x}, {max_x}, {min_y}, {max_y}, {max_speed}')
        if show_img or output_video is not None:
            key = cv2.waitKey(1) & 0xff
            res_plot = results[0].plot()
            cv2.putText(frame, f'{max_x}, {max_y}, {max_z:.1f}, {max_speed:.1f}', (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255,), 4, 2)
            output_frame = hconcat_frames(
                [frame, res_plot])
            if output_video is not None:
                output_video.write(output_frame)
            if show_img and play_pause(output_frame, key) == False:
                break

        framecount = framecount + 1
        if skip_frames > 0:
            skip_frames = skip_frames - 1

    cap.release()
    if output_video is not None:
        output_video.release()

    if export_trajectory:
        return rough_throw_detected, trajectory_dict
    else:
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

    parser.add_argument('--output_dir', type=str, default='', const="",
                        nargs='?', help='Output video directory. Use None to not generate the output videos.')

    parser.add_argument('--model_path', type=str, default='', const="",
                        nargs='?', help='Path to yolov8 model.')

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    parser.add_argument("--export_trajectory", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to export trajectory in json format. Make sure to have a output_dir.")

    return parser.parse_args()


@torch.no_grad()
def main(args):
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO(args.model_path)
    model.fuse()
    all_trajectory_dict = {} if args.export_trajectory else None

    if args.input_video is not None:
        if args.export_trajectory:
            result, trajectory_dict = throw_cfl_lateral(
                input_video_path=args.input_video, model=model, output_video_path=args.output_video,
                show_img=args.show_img, export_trajectory=args.export_trajectory)
            print('Result: ', result)
            if args.output_dir is not None:
                filename = os.path.splitext(
                    os.path.basename(args.input_video))[0]
                all_trajectory_dict[filename] = trajectory_dict
                output_json_path = os.path.join(
                    args.output_dir, 'trajectories.json')
                with open(output_json_path, 'w+', encoding='utf-8') as f:
                    json.dump(all_trajectory_dict, f,
                              ensure_ascii=False, indent=4, cls=NumpyEncoder)
                print('Exported trajectories!')

        else:
            result = throw_cfl_lateral(
                input_video_path=args.input_video, model=model, output_video_path=args.output_video,
                show_img=args.show_img, export_trajectory=args.export_trajectory)
            print('Result: ', result)

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
                if args.export_trajectory:
                    rough_throw_detected, current_trajectory_dict = throw_cfl_lateral(
                        input_video_path=input_video_path, model=model, output_video_path=output_video_path, show_img=args.show_img, export_trajectory=args.export_trajectory)
                    all_trajectory_dict[filename] = current_trajectory_dict

                else:
                    rough_throw_detected = throw_cfl_lateral(
                        input_video_path=input_video_path, model=model, output_video_path=output_video_path, show_img=args.show_img, export_trajectory=args.export_trajectory)

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
            if args.export_trajectory:
                output_json_path = os.path.join(
                    args.output_dir, 'trajectories.json')
                with open(output_json_path, 'w+', encoding='utf-8') as f:
                    json.dump(all_trajectory_dict, f,
                              ensure_ascii=False, indent=4, cls=NumpyEncoder)
                print('Exported trajectories!')


if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
