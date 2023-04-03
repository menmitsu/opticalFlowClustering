from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import os
import pandas as pd


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
        description='Script to process Bounce Test Set results.')

    parser.add_argument('--csv', type=str, default='', const="",
                        nargs='?', help='results.csv path')

    return parser.parse_args()


def get_eval_metrics(labels, preds):
    assert len(labels) == len(preds)
    eval_dict = {}
    cm = confusion_matrix(labels, preds)
    eval_dict['confusion_matrix'] = cm
    cm = cm.diagonal()/cm.sum(axis=1)
    report = classification_report(labels, preds, digits=4, output_dict=True)
    for i in range(len(cm)):
        report_i = report[str(i)]
        report_i['accuracy'] = cm[i]
        eval_dict[str(i)] = report_i
        report.pop(str(i), None)

    eval_dict['overall_report'] = report
    return eval_dict


def create_eval_metric_csv(labels, preds, output_path=''):
    eval_dict = get_eval_metrics(labels, preds)
    overall_accuracy = eval_dict['overall_report']['accuracy']
    macro_avg = eval_dict['overall_report']['macro avg']

    results_dict = {'combination': 'Third_iteration', 'overall_accuracy': [], '0_accuracy': [
    ], '1_accuracy': [], 'macro_precision': [], 'macro_recall': [], 'macro_f1-score': []}
    results_dict['overall_accuracy'].append(overall_accuracy)
    results_dict['0_accuracy'].append(eval_dict['0']['accuracy'])
    results_dict['1_accuracy'].append(eval_dict['1']['accuracy'])
    results_dict['macro_precision'].append(macro_avg['precision'])
    results_dict['macro_recall'].append(macro_avg['recall'])
    results_dict['macro_f1-score'].append(macro_avg['f1-score'])

    output_csv_path = os.path.join(output_path, 'eval_metrics.csv')
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_csv_path, index=False)

    output_json_path = os.path.join(output_path, 'report.json')
    with open(output_json_path, 'w+', encoding='utf-8') as f:
        json.dump(eval_dict, f, ensure_ascii=False, indent=4)


def main(args):
    df = pd.read_csv(args.csv)
    labels = []
    preds = []

    for index, row in df.iterrows():
        if pd.isnull(row['labels']) or pd.isnull(row['preds']):
            continue

        labels.append(row['labels'])
        preds.append(row['preds'])

    create_eval_metric_csv(labels, preds)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
