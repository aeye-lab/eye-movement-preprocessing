from __future__ import annotations

import os
import argparse
import glob

import pandas as pd
import numpy as np
from tqdm import tqdm

import utils.helpers as helpers



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EMTeC')
    parser.add_argument('--detection-method', type=str, default='ivt')
    args = parser.parse_args()
    detection_method = args.detection_method
    dataset = args.dataset
    detection_params = helpers.get_detection_params(detection_method)

    for detection_param in detection_params:
        detection_param_list = detection_param.split()
        assert len(detection_param_list) == 4
        detection_param_string = '_'.join(
            [
                detection_param_list[0][2:].replace('-', '_'),
                detection_param_list[1],
                detection_param_list[2][2:].replace('-', '_'),
                detection_param_list[3],
            ],
        )
        print(f'++ merging for {detection_param_string}')
        rms_dfs = list()

        if dataset == 'EMTeC':
            paths_to_rms = glob.glob(
                os.path.join(
                    'reading_measures',
                    dataset,
                    '*',
                    detection_param_string,
                    '*.csv',
                ),
            )
        elif dataset == 'PoTeC':
            paths_to_rms = glob.glob(
                os.path.join(
                    'reading_measures',
                    dataset,
                    '*',
                    '*',
                    detection_param_string,
                    '*.csv',
                ),
            )
        if len(paths_to_rms) == 0:
            print(f'++++ skipping due to no rms recorded')
            continue

        for path_to_rm in tqdm(paths_to_rms):
            rm_df = pd.read_csv(path_to_rm)
            rms_dfs.append(rm_df)

        all_rms = pd.concat(rms_dfs)

        # rename and drop columns
        all_rms = all_rms.rename(columns={'word_index': 'word_id'})

        # change order of columns
        if dataset == 'EMTeC':
            columns = [
                'subject_id', 'item_id', 'model', 'decoding_strategy', 'word_id', 'word', 'FFD', 'SFD',
                'FD', 'FPRT', 'FRT', 'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR',
                'FPReg', 'TRC_out', 'TRC_in', 'SL_in', 'SL_out', 'TFC',
            ]
        elif dataset == 'PoTeC':
            columns = [
                'subject_id', 'text_id', 'word', 'word_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT',
                'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out', 'TRC_in',
                'SL_in', 'SL_out', 'TFC',
            ]
        all_rms = all_rms[columns]

        save_dir = f'merged_rm_files/{dataset}/{detection_param_string}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'merged_rms.csv')
        print(f'++ saving to {save_path=}')
        all_rms.to_csv(save_path, sep='\t', index=False)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
