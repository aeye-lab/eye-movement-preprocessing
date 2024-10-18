import argparse
import os
import pymovements as pm
import pandas as pd
import numpy as np
from tqdm import tqdm
import utils.helpers as helpers


def compute_reading_measures(
        fixations_df,
        aoi_df,
) -> pd.DataFrame:
    """
    Computes reading measures from fixation sequences.
    :param fixations_df: pandas dataframe with columns 'index', 'duration', 'aoi', 'word_roi_str'
    :param aoi_df: pandas dataframe with columns 'word_index', 'word', and the aois of each word
    :return: pandas dataframe with reading measures
    """
    # append one extra dummy fixation to have the next fixation for the actual last fixation
    pd.concat(
        [
            fixations_df,
            pd.DataFrame(
                [[0 for _ in range(len(fixations_df.columns))]], columns=fixations_df.columns,
            ),
        ],
        ignore_index=True,
    )

    # XXX: fix off by one error (python vs human counting)
    aoi_df['aoi'] = aoi_df['aoi'] - 1
    # get the original words of the text and their word indices within the text
    text_aois = aoi_df['aoi'].tolist()
    text_strs = aoi_df['character'].tolist()

    # iterate over the words in that text
    word_dict = dict()
    for word_index, word in zip(text_aois, text_strs):
        word_row = {
            'word': word,
            'word_index': word_index,
            'FFD': 0,       # first-fixation duration
            'SFD': 0,       # single-fixation duration
            'FD': 0,        # first duration
            'FPRT': 0,      # first-pass reading time
            'FRT': 0,       # first-reading time
            'TFT': 0,       # total-fixation time
            'RRT': 0,       # re-reading time
            'RPD_inc': 0,   # inclusive regression-path duration
            'RPD_exc': 0,   # exclusive regression-path duration
            'RBRT': 0,      # right-bounded reading time
            'Fix': 0,       # fixation (binary)
            'FPF': 0,       # first-pass fixation (binary)
            'RR': 0,        # re-reading (binary)
            'FPReg': 0,     # first-pass regression (binary)
            'TRC_out': 0,   # total count of outgoing regressions
            'TRC_in': 0,    # total count of incoming regressions
            # 'LP': 0,        # landing position -- cannot have landing position because we don't work with character-based aois
            'SL_in': 0,     # incoming saccade length
            'SL_out': 0,    # outgoing saccade length
            'TFC': 0,       # total fixation count
        }

        word_dict[int(word_index)] = word_row

        right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

    for index, fixation in fixations_df.iterrows():

        # if aoi is not a number (i.e., it is coded as a missing value using '.'), continue
        try:
            aoi = int(fixation['aoi']) - 1
        except ValueError:
            continue

        # update variables
        last_fix_word_idx = cur_fix_word_idx

        cur_fix_word_idx = next_fix_word_idx
        cur_fix_dur = next_fix_dur
        if np.isnan(cur_fix_dur):
            continue

        next_fix_word_idx = aoi
        next_fix_dur = fixation['duration']

        # the 0 that we added as dummy fixation at the end of the fixations df
        if next_fix_dur == 0:
            # we set the idx to the idx of the actual last fixation such taht there is no error later
            next_fix_word_idx = cur_fix_word_idx

        if right_most_word < cur_fix_word_idx:
            right_most_word = cur_fix_word_idx

        if cur_fix_word_idx == -1:
            continue

        word_dict[cur_fix_word_idx]['TFT'] += int(cur_fix_dur)

        word_dict[cur_fix_word_idx]['TFC'] += 1

        if word_dict[cur_fix_word_idx]['FD'] == 0:
            word_dict[cur_fix_word_idx]['FD'] += int(cur_fix_dur)

        if right_most_word == cur_fix_word_idx:
            if word_dict[cur_fix_word_idx]['TRC_out'] == 0:
                word_dict[cur_fix_word_idx]['FPRT'] += int(cur_fix_dur)
                if last_fix_word_idx < cur_fix_word_idx:
                    word_dict[cur_fix_word_idx]['FFD'] += int(cur_fix_dur)
        else:
            if right_most_word < cur_fix_word_idx:
                print('error')
            word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

        if cur_fix_word_idx < last_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_in'] += 1
        if cur_fix_word_idx > next_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_out'] += 1
        if cur_fix_word_idx == right_most_word:
            word_dict[cur_fix_word_idx]['RBRT'] += int(cur_fix_dur)
        if (
            word_dict[cur_fix_word_idx]['FRT'] == 0 and
            (not next_fix_word_idx == cur_fix_word_idx or next_fix_dur == 0)
        ):
            word_dict[cur_fix_word_idx]['FRT'] = word_dict[cur_fix_word_idx]['TFT']
        if word_dict[cur_fix_word_idx]['SL_in'] == 0:
            word_dict[cur_fix_word_idx]['SL_in'] = cur_fix_word_idx - last_fix_word_idx
        if word_dict[cur_fix_word_idx]['SL_out'] == 0:
            word_dict[cur_fix_word_idx]['SL_out'] = next_fix_word_idx - cur_fix_word_idx

    # Compute the remaining reading measures from the ones computed above
    for word_indices, word_rm in sorted(word_dict.items()):
        if word_rm['FFD'] == word_rm['FPRT']:
            word_rm['SFD'] = word_rm['FFD']
        word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
        word_rm['FPF'] = int(word_rm['FFD'] > 0)
        word_rm['RR'] = int(word_rm['RRT'] > 0)
        word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
        word_rm['Fix'] = int(word_rm['TFT'] > 0)
        word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']

        # if it is the first word, we create the df (index of first word is 0)
        if word_indices == 0:
            rm_df = pd.DataFrame([word_rm])
        else:
            rm_df = pd.concat([rm_df, pd.DataFrame([word_rm])])

    return rm_df


def _pla_preprocess(args):
    dataset_name = args.dataset
    detection_method = args.detection_method

    dataset = pm.Dataset(dataset_name, path=f'data/{dataset_name}')
    try:
        dataset.load(
            # subset={'subject_id': 5}
        )
    except:
        dataset.download()
        dataset.load(
            # subset={'subject_id': 5}
        )
    dataset.pix2deg()
    dataset.pos2vel()
    minimum_duration = args.minimum_duration
    dispersion_threshold = args.dispersion_threshold
    velocity_threshold = args.velocity_threshold

    if detection_method == 'ivt':
        detection_params = {'minimum_duration': minimum_duration,
                            'velocity_threshold': velocity_threshold,
                        }
    elif detection_method == 'idt':
        detection_params = {'minimum_duration': minimum_duration,
                            'dispersion_threshold': dispersion_threshold,
                        }
    detection_param_string = ''
    for key in detection_params:
        detection_param_string += str(key) + '_' + str(detection_params[key]) + '_'
    detection_param_string = detection_param_string[0:len(detection_param_string)-1]
    dataset.detect(detection_method, **detection_params)
    dataset.compute_event_properties(("location", {'position_column':"pixel"}))

    for event_idx in tqdm(range(len(dataset.events))):
        if dataset_name == 'PoTeC':
            tmp_df = dataset.events[event_idx]
            if tmp_df.frame.is_empty():
                print('+ skip due to empty DF')
                continue
            text_id = tmp_df['text_id'][0]
            aoi_text_stimulus = pm.stimulus.text.from_file(
                f'./potec/stimuli/word_aoi_texts/word_aoi_{text_id}.tsv',
                aoi_column='character',
                start_x_column='start_x',
                start_y_column='start_y',
                end_x_column='end_x',
                end_y_column='end_y',
                page_column='page',
                custom_read_kwargs={'separator': '\t'},
            )
        dataset.events[event_idx].map_to_aois(aoi_text_stimulus)

    for _fix_file in dataset.events:
        if _fix_file.frame.is_empty():
            print('+ skip due to empty DF')
            continue
        fixations_df = _fix_file.frame.to_pandas()

        text_id = fixations_df.iloc[0]['text_id']
        subject_id = int(fixations_df.iloc[0]['subject_id'])
        if dataset_name == 'PoTeC':
            aoi_df = pd.read_csv(f'./potec/stimuli/word_aoi_texts/word_aoi_{text_id}.tsv', delimiter='\t')
            save_basepath = os.path.join(
                'reading_measures', dataset_name,
                str(subject_id), str(text_id),
                detection_param_string,
            )
            rm_filename = f'{subject_id}-{text_id}-reading_measures.csv'
        path_save_rm_file = os.path.join(save_basepath, rm_filename)

        if args.check_file_exists:
            if os.path.isfile(path_save_rm_file):
                print(f'\t--- file {path_save_rm_file} already exists. skipping.')
                continue

        os.makedirs(save_basepath, exist_ok=True)

        print(f'++ processing file {path_save_rm_file}')

        rm_df = compute_reading_measures(
            fixations_df=fixations_df,
            aoi_df=aoi_df,
        )

        rm_df['subject_id'] = subject_id
        rm_df['text_id'] = text_id

        rm_df.to_csv(path_save_rm_file, index=False)

    return 0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-file-exists', action='store_true')
    parser.add_argument('--dataset', type=str, default='PoTeC')
    parser.add_argument('--detection-method', type=str, default='ivt')
    parser.add_argument(
        '--minimum-duration', type=int,
        default=100,
    )
    parser.add_argument(
        '--dispersion-threshold', type=float,
        default=1.0,
    )
    parser.add_argument(
        '--velocity-threshold', type=float,
        default=20.0,
    )
    args = parser.parse_args()
    return _pla_preprocess(args)

if __name__ == '__main__':
    raise SystemExit(main())
