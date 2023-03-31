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


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1].flatten()


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


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def scale_contour(cnt, CONTOUR_SCALE_X=1, CONTOUR_SCALE_Y=1):
    """Scales a counter in both X and Y direction by a factor of CONTOUR_SCALE_X and CONTOUR_SCALE_Y respectively"""
    M = cv2.moments(cnt.astype(np.float32))

    cx = M['m10']/max(M['m00'],1e-10)
    cy = M['m01']/max(M['m00'],1e-10)

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = np.array([[x*CONTOUR_SCALE_X, y*CONTOUR_SCALE_Y]
                           for x, y in cnt_norm])
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


# Corresponding classes: 0 -> person, 58 -> plant
def get_masks(frames: List[np.array], model, classes=[0, 58], scale_contours=False) -> List[np.array]:
    # Returns a list of contours for objects of different classes found in each frame

    shape_y = frames[0].shape[0]
    shape_x = frames[0].shape[1]
    results = model(frames, verbose=False)
    cnts_list = []
    for result in results:
        boxes = result.boxes
        masks = None
        if result.masks is not None and result.masks.segments is not None:
            masks = result.masks.segments

        cnts = []
        if boxes is not None and masks is not None and len(boxes) == len(masks):
            for box, segment in zip(boxes, masks):
                if box.cls in classes and segment.size > 0:
                    segment[:, 0] *= shape_x
                    segment[:, 1] *= shape_y
                    if scale_contours:
                        segment = scale_contour(segment)
                    ctr = np.array(segment).reshape(
                        (-1, 1, 2)).astype(np.int32)
                    cnts.append(ctr)

        cnts_list.append(cnts)

    return cnts_list


def bgr_erode(img, iterations, kernal_size=5):
    # separate the color channels
    b, g, r = cv2.split(img)

    # define the structuring element
    kernel = np.ones((kernal_size, kernal_size), np.uint8)

    # perform erosion for each channel
    eroded_b = cv2.erode(b, kernel, iterations=iterations)
    eroded_g = cv2.erode(g, kernel, iterations=iterations)
    eroded_r = cv2.erode(r, kernel, iterations=iterations)

    # merge the eroded channels back to form the final image
    eroded_img = cv2.merge((eroded_b, eroded_g, eroded_r))
    return eroded_img


def bgr_dilate(img, iterations, kernal_size=5):
    # separate the color channels
    b, g, r = cv2.split(img)

    # define the structuring element
    kernel = np.ones((kernal_size, kernal_size), np.uint8)

    # perform dilation for each channel
    eroded_b = cv2.dilate(b, kernel, iterations=iterations)
    eroded_g = cv2.dilate(g, kernel, iterations=iterations)
    eroded_r = cv2.dilate(r, kernel, iterations=iterations)

    # merge the eroded channels back to form the final image
    eroded_img = cv2.merge((eroded_b, eroded_g, eroded_r))
    return eroded_img


def sliding_window(values, newvalue):
    values[:-1] = values[1:]
    values[-1] = newvalue


def remove_adjacent_duplicates(data):
    selection = np.ones(len(data), dtype=bool)
    selection[1:] = data[1:] != data[:-1]
    return data[selection]


def get_extremas(data):
    # Remove adjacent duplicates
    data_adj_dedup = remove_adjacent_duplicates(data)
    # Create a boolean array indicating whether each element is greater than its neighbors
    is_extrema = (data_adj_dedup[1:-1] > data_adj_dedup[:-2]
                  ) & (data_adj_dedup[1:-1] > data_adj_dedup[2:])
    # Find the indices of the extrema
    extrema_idxs = np.where(is_extrema)[0] + 1
    # Find the indices w.r.t data
    extrema_idxs_wrt_data = np.where(
        np.isin(data, data_adj_dedup[extrema_idxs]))[0]
    return extrema_idxs_wrt_data


def filter_peaks(max_idx, extremas_idx, peak_distance=10):
    selection = (extremas_idx > max_idx) & (
        extremas_idx < max_idx + peak_distance)
    return extremas_idx[selection]


def detect_bounce_pattern(bgr_values):
    extrema_idxs = []
    min_length = bgr_values[0].size

    for i in range(len(bgr_values)):
        extrema_idxs.append(get_extremas(bgr_values[i]))
        min_length = min(min_length, extrema_idxs[-1].size)

    if min_length > 0:
        g_max_index_wrt_extrema = np.argmax(bgr_values[1][extrema_idxs[1]])
        g_max_index = extrema_idxs[1][g_max_index_wrt_extrema]

        b_value_at_g = bgr_values[0][g_max_index]
        g_max_val = bgr_values[1][g_max_index]
        r_value_at_g = bgr_values[2][g_max_index]

        b_peak_idxs_after_g = filter_peaks(g_max_index, extrema_idxs[0])

        if b_peak_idxs_after_g.size > 0 and g_max_val - b_value_at_g > 20 and g_max_val > 30 and r_value_at_g > 25:

            max_b_peak_idx = b_peak_idxs_after_g[np.argmax(
                bgr_values[0][b_peak_idxs_after_g])]
            b_peak_after_g = bgr_values[0][max_b_peak_idx]
            g_value_after_g = bgr_values[1][max_b_peak_idx]
            r_peak_after_g = bgr_values[2][max_b_peak_idx]

            if (min(b_peak_after_g, r_peak_after_g) > 6
                and g_max_val > max(b_peak_after_g, r_peak_after_g)
                and abs(r_peak_after_g - b_peak_after_g) <= 25
                    and min(b_peak_after_g, r_peak_after_g) - g_value_after_g > 5):
                print("BOUNCE DETECTED!")
                print("b at g", b_value_at_g)
                print("g max", g_max_val)
                print("r at g", r_value_at_g)
                print("b after g", b_peak_after_g)
                print("g after g", g_value_after_g)
                print("r after g", r_peak_after_g)
                return True

    return False


def process_flow(frame, gamma=2, threshold_val=10):
    # Remove noise by thresholding and then perform erosion and dilation.
    frame[frame <= 35] = 0
    frame = bgr_erode(frame, iterations=8, kernal_size=6)
    frame = bgr_dilate(frame, iterations=4, kernal_size=10)

    # Sets the color of all the points inside the detected contours to their mean value and increases gamma
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        frame_gray, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        mask = np.zeros(threshold.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(frame, mask=mask)
        mean_val = np.uint8(mean_val[0:3])
        mean_val = adjust_gamma(mean_val, gamma).flatten()
        frame[mask == 255] = mean_val

    return frame


def plot_graphs_and_create_img(ax, fig, bgr_values):
    ax.clear()
    ax.plot(bgr_values[0], color='blue')
    ax.plot(bgr_values[1], color='green')
    ax.plot(bgr_values[2], color='red')
    ax.set_ylim(0, 200)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def create_mask(resolution, bbox=((0, 90), (640, 729))):
    mask = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)
    cv2.rectangle(mask, bbox[0], bbox[1], (255, 255, 255), -1)
    return mask


def play_pause(frame, key, window_name='Win'):
    if key == ord('p'):  # Use 'p' to pause and resume the video playback
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow(window_name, frame)
            if key2 == ord('p') or key2 == 27:
                break

    cv2.imshow(window_name, frame)
    if key == 27:
        return False


def process_frame(frame, resolution, mask):
    frame = cv2.resize(frame, resolution)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    return frame


def hconcat_frames(frames, resolution=(640, 640)):
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, resolution)
        resized_frames.append(resized_frame)

    concatenated_frame = cv2.hconcat(resized_frames)
    return concatenated_frame


def compute_flow_and_mask_video(input_video_path, model, output_video_path, show_img=False):
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    cap = cv2.VideoCapture(input_video_path)
    output_video = None
    bounce_detected = False
    scale_value = 20
    resolution = (1280, 720)

    if output_video_path is not None:
        print('Output at: ', output_video_path)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), framerate, (1920, 640))  # NOTE: This VideoWriter may not work in linux environments

    mask = create_mask(resolution=resolution)
    success, firstframe = cap.read()
    firstframe = process_frame(firstframe, resolution, mask)
    compflow = ComputeOpticalFLow(firstframe)

    window_size = 15
    bgr_values = [np.zeros(window_size, dtype=np.float32) for _ in range(3)]

    if show_img or output_video:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    while success:
        success, frame = cap.read()
        if not success:
            break

        key = cv2.waitKey(1) & 0xff
        frame = process_frame(frame, resolution, mask)

        opflowimg = compflow.compute(frame)
        masks = get_masks([frame], model)[0]
        cv2.drawContours(opflowimg, masks, -1, (0, 0, 0), cv2.FILLED)
        opflowimg = process_flow(opflowimg)

        for i in range(3):
            color_channel = opflowimg[:, :, i].mean()
            color_channel = color_channel*scale_value
            sliding_window(bgr_values[i], color_channel)

        bounce_detected_in_window = detect_bounce_pattern(bgr_values)
        bounce_detected = bounce_detected | bounce_detected_in_window

        if show_img or output_video:
            if bounce_detected_in_window:
                cv2.putText(frame, "BOUNCE DETECTED!", (280, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)

            cv2.putText(opflowimg, "r %2.f, " % bgr_values[2][-1] + "g %2.f, " % bgr_values[1][-1] + "b %2.f" %
                        bgr_values[0][-1], (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255,), 4, 2)
            img = plot_graphs_and_create_img(ax, fig, bgr_values)
            output_frame = hconcat_frames([frame, opflowimg, img])

            if output_video is not None:
                output_video.write(output_frame)

            if show_img and play_pause(output_frame, key) == False:
                break

    cap.release()
    if output_video is not None:
        output_video.release()
        print("Output video released.")

    return bounce_detected


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Takes an input video and detects bounce violations.')

    parser.add_argument('--input_video', type=str, default=None, const="",
                        nargs='?', help='Input video path.')

    parser.add_argument('--output_video', type=str, default=None, const="",
                        nargs='?', help='Output video path')

    parser.add_argument('--input_dir', type=str, default=None, const="",
                        nargs='?', help='Input folder containing mp4 videos.')

    parser.add_argument('--output_dir', type=str, default=None, const="",
                        nargs='?', help='Output folder path.')

    parser.add_argument("--show_img", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to show processed frames.")

    return parser.parse_args()


def main(args):
    print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = YOLO('yolov8x-seg.pt')
    model.fuse()

    if args.input_video is not None:
        print(compute_flow_and_mask_video(input_video_path=args.input_video,
                                          model=model, output_video_path=args.output_video,
                                          show_img=args.show_img))
    elif args.input_dir is not None:
        if args.output_dir is not None:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)

        for file in os.listdir(args.input_dir):
            if(file.endswith('.mp4')):
                filename = os.path.splitext(file)[0]
                input_video_path = os.path.join(args.input_dir, file)
                output_video_path = os.path.join(
                    args.output_dir, filename + "_output.mp4")

                print("Processing: ", input_video_path)
                pred = compute_flow_and_mask_video(input_video_path=input_video_path,
                                                   model=model, output_video_path=output_video_path, show_img=args.show_img)
                print("Done!")

        print("Processed all files!")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
