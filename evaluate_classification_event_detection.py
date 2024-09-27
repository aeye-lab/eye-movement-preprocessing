from __future__ import annotations

import argparse
import os
import config.config as config
import utils.helpers as helpers

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='sbsat')
    parser.add_argument(
        '--sampling-rate', type=int, default=1000)    
    parser.add_argument('--detection-method', type=str, default='ivt')
    args = parser.parse_args()
    return args

            
def evaluate(args):
    dataset = args.dataset
    detection_method = args.detection_method
    sampling_rate = args.sampling_rate
    
    detection_params = helpers.get_detection_params(detection_method)
    label_columns = helpers.get_datset_labels(dataset)
    
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