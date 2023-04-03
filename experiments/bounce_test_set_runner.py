from sklearn.metrics import classification_report
from flow_masking_and_bounce_detection import compute_flow_and_mask_video
from bounce_detection_from_flow import detect_bounce_from_flow
from typing import List
import argparse
import os
import pandas as pd
from eval_metric import create_eval_metric_csv


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


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Generates results for Bounce Test Set')

    parser.add_argument('--test_set_csv', type=str, default='', const="",
                        nargs='?', help='Test set csv path.')

    parser.add_argument('--clips_root_folder', type=str, default='', const="",
                        nargs='?', help='Test set clips folder path.')

    parser.add_argument('--output_folder', type=str, default='', const="",
                        nargs='?', help='Output folder for results, processed videos (if output_videos = True).')

    parser.add_argument("--output_videos", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to also output processed videos.")

    parser.add_argument("--flow_input", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether the input videos are flow videos.")

    return parser.parse_args()


def main(args):
    model = None
    if args.flow_input == False:
        print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = YOLO('yolov8x-seg.pt')
        model.fuse()

    if args.output_folder is not None:
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)

    df = pd.read_csv(args.test_set_csv)
    video_name = []
    labels = []
    preds = []

    for index, row in df.iterrows():
        if pd.isnull(row['Label']) or pd.isnull(row['Video_Name']):
            continue

        print("Processing file number: ", index, " named:", row['Video_Name'])

        input_video_path = os.path.join(
            args.clips_root_folder, row['Video_Name'] + "_output.mp4")
        output_video_path = None
        if args.output_videos:
            output_video_path = os.path.join(
                args.output_folder, row['Video_Name'] + "_output.mp4")

        pred = None
        if args.flow_input:
            pred = detect_bounce_from_flow(
                input_video_path=input_video_path, output_video_path=output_video_path, show_img=False, process_optical_flow=True)

        else:
            pred = compute_flow_and_mask_video(
                input_video_path=input_video_path, model=model, output_video_path=output_video_path, show_img=False)

        if output_video_path is not None:
            print('Output video stored at: ', output_video_path)

        video_name.append(row['Video_Name'])
        labels.append(1 if row['Label'] == 'Yes' else 0)
        preds.append(1 if pred else 0)

    print(classification_report(labels, preds, digits=4))

    if args.output_folder is not None:
        output_results_csv_path = os.path.join(
            args.output_folder, 'results.csv')
        results_df = pd.DataFrame(
            {"video_name": video_name, "labels": labels, "preds": preds})
        results_df.to_csv(output_results_csv_path, index=False)
        create_eval_metric_csv(labels, preds, args.output_folder)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
