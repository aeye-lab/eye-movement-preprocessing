from __future__ import annotations

import argparse
import os
import config.config as config

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='sbsat')
    parser.add_argument(
        '--sampling-rate', type=int, default=1000)    
    parser.add_argument('--detection-method', type=str, default='ivt')
    args = parser.parse_args()
    return args


def get_detection_params(detection_method):
    out_strings = []
    if detection_method == 'ivt':
        params = config.ivt_detection_params
        minimum_durations = params['minimum_duration']
        velocity_thresholds = params['velocity_threshold']
        for minimum_duration in minimum_durations:
            for velocity_threshold in velocity_thresholds:
                out_str = '--minimum-duration ' + str(minimum_duration) +\
                            ' --velocity-threshold ' + str(velocity_threshold)
                out_strings.append(out_str)
    elif detection_method == 'idt':
        params = config.idt_detection_params
        minimum_durations = params['minimum_duration']
        dispersion_thresholds = params['dispersion_threshold']
        for minimum_duration in minimum_durations:
            for dispersion_threshold in dispersion_thresholds:
                out_str = '--minimum-duration ' + str(minimum_duration) +\
                            ' --dispersion-threshold ' + str(dispersion_threshold)
                out_strings.append(out_str)
    elif detection_method == 'microsaccades':
        params = config.microsaccades_detection_params
        minimum_durations = params['minimum_duration']
        for minimum_duration in minimum_durations:
            out_str = '--minimum-duration ' + str(minimum_duration)
            out_strings.append(out_str)
    return out_strings
            
def get_datset_labels(dataset):
    if dataset == 'sbsat':
        labels = config.SBSAT_LABELS
    elif dataset == 'gazebase':
        labels = config.GAZBASE_LABELS
    return labels
            
def evaluate(args):
    dataset = args.dataset
    detection_method = args.detection_method
    sampling_rate = args.sampling_rate
    
    detection_params = get_detection_params(detection_method)
    label_columns = get_datset_labels(dataset)
    
    for label_column in label_columns:
        for detection_param in detection_params:
            exec_string = 'python train_classification_model.py --dataset ' + str(dataset) +\
                            ' --label-column ' + str(label_column) +\
                            ' --sampling-rate ' + str(sampling_rate) +\
                            ' --detection-method ' + str(detection_method) +\
                            ' ' + detection_param
            os.system(exec_string)

def main() -> int:
    args = get_argument_parser()
    evaluate(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())