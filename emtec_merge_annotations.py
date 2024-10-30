import os
import argparse

import numpy as np
import pandas as pd

import utils.helpers as helpers

def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument('--detection-method', type=str, default='ivt')
    args = parser.parse_args()
    # read in the files

    path_to_stimuli = 'EMTeC/data/stimuli.csv'
    stimuli = pd.read_csv(path_to_stimuli, sep='\t')
    detection_params = helpers.get_detection_params(args.detection_method)
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
        path_to_word_level_annotations = 'EMTeC/annotation/word_level_annotations.csv'
        _path_to_rms = f'./merged_rm_files/EMTeC/{detection_param_string}/'
        path_to_rms = os.path.join(_path_to_rms, 'merged_rms.csv')

        annotations = pd.read_csv(path_to_word_level_annotations, sep='\t')
        try:
            rms = pd.read_csv(path_to_rms, sep='\t')
        except:
            continue

        # merge the annotations with the reading measures
        print(' --- merging annotations with reading measures')
        merged = pd.merge(rms, annotations, on=['item_id', 'model', 'decoding_strategy', 'word_id', 'word'], how='left')

        # merge the reading measures with the stimulus info type/task/subcategory
        print(' --- merging reading measures with stimulus info')
        merged = pd.merge(merged, stimuli[['item_id', 'model', 'decoding_strategy', 'type', 'task', 'subcategory']],
                          on=['item_id', 'model', 'decoding_strategy'])

        # save the merged files
        save_path = os.path.join(_path_to_rms, 'reading_measures.csv')
        print(f'saving {save_path=}', end='\n\n')
        merged.to_csv(save_path, sep='\t', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
