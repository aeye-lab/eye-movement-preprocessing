from __future__ import annotations

import argparse
import os
import numpy as np
import config.config as config
import utils.helpers as helpers

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='sbsat')
    parser.add_argument('--detection-method', type=str, default='ivt')
    parser.add_argument('--flag-redo', type=int,default=0)
    args = parser.parse_args()
    return args

            
def evaluate(args):
    dataset = args.dataset
    detection_method = args.detection_method
    flag_redo = args.flag_redo
    
    if dataset == 'gazebasevr':
        detection_params = helpers.get_detection_params(detection_method, sampling_rate=250)
    elif dataset == 'hbn':
        detection_params = helpers.get_detection_params(detection_method, sampling_rate=120)
    elif dataset == 'gazeonfaces':
        detection_params = helpers.get_detection_params(detection_method, sampling_rate=60)
    elif dataset == 'gazegraph':
        detection_params = helpers.get_detection_params(detection_method, sampling_rate=30)    
    else:
        detection_params = helpers.get_detection_params(detection_method)
    label_columns = helpers.get_datset_labels(dataset)
    
    exec_strings = []
    for label_column in label_columns:
        for detection_param in detection_params:
            exec_string = 'python train_classification_model.py --dataset ' + str(dataset) +\
                            ' --label-column ' + str(label_column) +\
                            ' --detection-method ' + str(detection_method) +\
                            ' --flag-redo ' + str(flag_redo) +\
                            ' ' + detection_param
            exec_strings.append(exec_string)
            
    # shuffle and start
    np.random.shuffle(exec_strings)
    for exec_string in exec_strings:
        print('run ' + str(exec_string))
        os.system(exec_string)

def main() -> int:
    args = get_argument_parser()
    evaluate(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
