from __future__ import annotations

import argparse
import joblib
import os
import utils.helpers as helpers

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='PoTeC')
    parser.add_argument('--detection-method', type=str, default='ivt')
    parser.add_argument('--flag-redo', type=int,default=0)
    args = parser.parse_args()
    return args


def main() -> int:
    args = get_argument_parser()
    dataset = args.dataset
    detection_method = args.detection_method
    flag_redo = args.flag_redo
    detection_params = helpers.get_detection_params(args.detection_method)
    joblib.Parallel(n_jobs=100)(
        joblib.delayed(os.system)(
            f'python pla_preprocess.py --dataset {dataset} --detection-method {detection_method} {detection_param}'
        )
        for detection_param in detection_params
    )
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
