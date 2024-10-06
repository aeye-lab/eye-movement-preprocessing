from __future__ import annotations

import argparse
import sys
# to be changed after release!!!
sys.path.append('/mnt/mlshare/prasse/aeye_git/pymovements/src/')

import os
import numpy as np
import polars as pl
import joblib
import pymovements as pm
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics

import preprocessing.feature_extraction as feature_extraction
import config.config as config

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='sbsat')
    parser.add_argument(
        '--label-column', type=str, default='native',
        # choices=['acc', 'difficulty', 'subj_acc', 'native', 'task_name', 'familarity'],
    )
    parser.add_argument(
        '--save-dir', type=str,
        default='results/',
    )
    parser.add_argument(
        '--sampling-rate', type=int, default=1000)
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
    parser.add_argument('--detection-method', type=str, default='ivt')
    args = parser.parse_args()
    return args
    


def get_feature_matrix(dataset,
                    sampling_rate,
                    blink_threshold,
                    blink_window_size,
                    blink_min_duration,
                    blink_velocity_threshold,
                    feature_aggregations,
                    detection_method,
                    label_grouping,
                    instance_grouping,
                    splitting_criterion,
                    max_len=None,
                    ):
                        
    event_name_dict = config.event_name_dict
    event_name_code_dict = config.event_name_code_dict
    detection_method_default_event = config.detection_method_default_event    
                        
    num_add = 1000
    group_names = []
    splitting_names = []
    iter_counter = 0

    for i in tqdm(np.arange(len(dataset.gaze))):
        cur_gaze_df = dataset.gaze[i]
        try:
            cur_gaze_df.unnest()
        except:
            print('Warning: maybe already unnested')
        cur_gaze_df = cur_gaze_df.frame
        if 'position_xl' in cur_gaze_df.columns:
            cur_gaze_df =  cur_gaze_df.rename({'position_xl':'position_x',
                                'position_yl':'position_y',
                                'velocity_xl':'velocity_x',
                                'velocity_yl':'velocity_y',
                               })
        cur_event_df = dataset.events[i].frame

        # add events to gaze df
        # initialize event_type as None
        event_type = np.array([event_name_code_dict[detection_method_default_event[detection_method]] for _ in range(cur_gaze_df.shape[0])], dtype=np.int32)
        for event_id in range(cur_event_df.shape[0]):
            cur_event = cur_event_df[event_id]
            cur_onset_time = cur_event_df[event_id]['onset'][0]
            cur_offset_time = cur_event_df[event_id]['offset'][0]
            if 'index' in cur_gaze_df.columns:
                cur_onset_id = cur_gaze_df.filter(pl.col('time') == cur_onset_time)['index'][0]
                cur_offset_id = cur_gaze_df.filter(pl.col('time') == cur_offset_time)['index'][0]
            else:
                cur_onset_id = cur_gaze_df.with_row_index().filter(pl.col('time').cast(int) == cur_onset_time)['index'][0]
                cur_offset_id = cur_gaze_df.with_row_index().filter(pl.col('time').cast(int) == cur_offset_time)['index'][0]
            event_type[cur_onset_id:cur_offset_id] = event_name_code_dict[event_name_dict[cur_event_df[event_id]['name'][0]]]

        if 'postion_x' in cur_gaze_df.columns:
            pos_x = np.array(cur_gaze_df['position_x'].is_null())
        else:
            pos_x = np.zeros([cur_gaze_df.shape[0],])
        if 'velocity_x' in cur_gaze_df.columns:
            vel_x = np.array(cur_gaze_df['velocity_x'].is_null())
        else:
            vel_x = np.zeros([cur_gaze_df.shape[0],])
        if 'pixel_x' in cur_gaze_df.columns:
            pix_x = np.array(cur_gaze_df['pixel_x'].is_null())
        else:
            pix_x = np.zeros([cur_gaze_df.shape[0],])
        null_ids = np.logical_or(pos_x,
                             np.logical_or(vel_x,
                            np.array(pix_x)))
        non_ids = np.where(null_ids)[0]     
        event_type[non_ids] = -1
        cur_gaze_df = cur_gaze_df.with_columns(pl.Series(name="event_type", values=event_type))
        
        for name, data in cur_gaze_df.group_by(instance_grouping):
            if max_len is not None:
                if data.shape[0] > max_len:
                    data = data[0:max_len,:]
            # extract features
            combined_features, combined_feature_names = feature_extraction.compute_features(data,
                            sampling_rate,
                            blink_threshold,
                            blink_window_size,
                            blink_min_duration,
                            blink_velocity_threshold,
                            feature_aggregations,
                            )
            if iter_counter == 0:
                feature_matrix = np.zeros([num_add, len(combined_features)])
                group_names = []
            while feature_matrix.shape[0] <= iter_counter:
                feature_matrix = np.concatenate([feature_matrix, np.zeros([num_add, len(combined_features)])], axis=0)
            label_tuple = []
            for jj in range(len(label_grouping)):
                label_tuple.append(str(data[label_grouping[jj]][0]))
            label_tuple = '_'.join(label_tuple)
            group_names.append(label_tuple)
            splitting_names.append(data[splitting_criterion][0])
            feature_matrix[iter_counter,:] = combined_features
            iter_counter += 1
    feature_matrix = feature_matrix[0:iter_counter,:]
    feature_matrix[np.isnan(feature_matrix)] = 0.0
    return feature_matrix, group_names, splitting_names


def evaluate_model(args):
    dataset_name = args.dataset
    label_column = args.label_column
    save_dir = args.save_dir
    detection_method = args.detection_method    
    result_prefix = detection_method
    #sampling_rate = args.sampling_rate    
    
    # detection method params
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
    elif detection_method == 'microsaccades':
        detection_params = {'minimum_duration': minimum_duration,
                        }
    
    detection_param_string = ''
    for key in detection_params:
        detection_param_string += str(key) + '_' + str(detection_params[key]) + '_'
    detection_param_string = detection_param_string[0:len(detection_param_string)-1]
    
    # load config data    
    event_name_dict = config.event_name_dict
    event_name_code_dict = config.event_name_code_dict
    detection_method_default_event = config.detection_method_default_event
    feature_aggregations = config.feature_aggregations
    blink_threshold = config.blink_threshold
    blink_window_size = config.blink_window_size
    blink_min_duration = config.blink_min_duration
    blink_velocity_threshold = config.blink_velocity_threshold
    
    param_grid = config.param_grid
    grid_search_verbosity = config.grid_search_verbosity
    n_splits = config.n_splits
        
    
    if dataset_name == 'sbsat':
        label_grouping = config.SBSAT_LABEL_GROUPING
        instance_grouping = config.SBSAT_INSTANCE_GROUPING
        splitting_criterion = config.SBSAT_SPLITTING_CRITERION        
        label_path = config.SBSAT_LABEL_PATH
        max_len = config.SBSAT_MAXLEN
        
        # load labels
        label_df = pl.read_csv(label_path)
        
        print(' === Loading data ===')
        label_mean = np.mean(label_df[label_column].to_numpy())
        
        dataset = pm.Dataset("SBSAT", path='data/SBSAT')
        try:
            dataset.load(
                #subset={'subject_id':[1,2,3,4,5,6,7,8,9,10]}
                )
        except:
            dataset.download()
            dataset.load(
                #subset={'subject_id':[1,2,3,4,5,6,7,8,9,10]}
            )

        sampling_rate = dataset.definition.experiment.sampling_rate
        deleted_instances = 0
        instance_count = 0
        # Preprocessing
        # delete screens with errors (time difference not constant)
        for i in tqdm(np.arange(len(dataset.gaze))):
            cur_gaze_df = dataset.gaze[i].frame.with_row_index()
            delete_ids = []
            for name, data in cur_gaze_df.group_by(instance_grouping):
                timesteps_diff = np.diff(list(data['time']))
                number_steps = len(np.unique(timesteps_diff))
                if number_steps > 1:
                    delete_ids += list(data['index'])
                    deleted_instances+= 1
                instance_count += 1
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_row_index().filter(~pl.col("index").is_in(delete_ids))            
        print(' === Evaluating model ===')
        print('    deleted instances: ' + str(deleted_instances) +\
                ' (' + str(np.round(deleted_instances/instance_count*100.,decimals=2)) + '%)')
        
        
        
        # transform pixel coordinates to degrees of visual angle
        dataset.pix2deg()
        
        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len
                                )
        
        # create label
        label = list(label_df[label_column])
        label_group_list = dict()
        label_dict = dict()
        for group in label_grouping:
            label_group_list[group] = list(label_df[group])
        for i in range(label_df.shape[0]):
            cur_tuple = '_'.join([str(label_group_list[a][i]) for a in label_group_list])
            label_dict[cur_tuple] = label[i]
        
        y = []
        subjects = []
        for i in range(len(group_names)):
            c_tuple = group_names[i]
            c_subject = splitting_names[i]
            y.append(label_dict[c_tuple])
            subjects.append(c_subject)

        y = np.array(y)
        subjects = np.array(subjects)
        
        # binarize the label
        bin_label = np.zeros([len(y),])
        bin_label[y >= label_mean] = 1.
        y = bin_label
    elif dataset_name == 'gazebase':
        label_grouping = config.GAZEBASE_LABEL_GROUPING
        instance_grouping = config.GAZEBASE_INSTANCE_GROUPING
        splitting_criterion = config.GAZEBASE_SPLITTING_CRITERION
        max_len = config.GAZEBASE_MAXLEN
        
        dataset = pm.Dataset("GazeBase", path='data/GazeBase')
        try:
            dataset.load(subset = {#'subject_id':[1,2,3,4,5,6,7,8,9,10],
                                   'task_name': ['BLG', 'FXS', 'HSS', 'RAN', 'TEX', 'VD1'],
                                   'round_id': [1],
                                   'session_id': [1],
                                  })
        except:
            dataset.download()
            dataset.load(subset = {#'subject_id':[1,2,3,4],
                                   'task_name': ['BLG', 'FXS', 'HSS', 'RAN', 'TEX', 'VD1'],
                                   'round_id': [1],
                                   'session_id': [1],
                                  })

        sampling_rate = dataset.definition.experiment.sampling_rate
        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len,
                                )
        
        from sklearn.preprocessing import LabelEncoder
        label_names = np.array(group_names)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_names)                
        subjects = np.array(splitting_names)
    elif dataset_name == 'gazebasevr':
        label_grouping = config.GAZEBASEVR_LABEL_GROUPING
        instance_grouping = config.GAZEBASEVR_INSTANCE_GROUPING
        splitting_criterion = config.GAZEBASEVR_SPLITTING_CRITERION
        max_len = config.GAZEBASEVR_MAXLEN
        
        dataset = pm.Dataset("GazeBaseVR", path='data/GazeBaseVR')
        try:
            dataset.load(subset = {#'subject_id':[1,2,3,4,5,6,7,8,9,10],
                                           'task_name': ['1_VRG', '4_TEX', '2_PUR', '3_VID', '5_RAN'],
                                           'round_id': [1],
                                           'session_id': [1],
                                          })
        except:
            dataset.download()
            dataset.load(subset = {#'subject_id':[1,2,3,4,5,6,7,8,9,10],
                                           'task_name': ['1_VRG', '4_TEX', '2_PUR', '3_VID', '5_RAN'],
                                           'round_id': [1],
                                           'session_id': [1],
                                          })

        # replace timesteps -> change to int (FIX for IDT algorithm)
        for i in tqdm(np.arange(len(dataset.gaze))):
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_columns(pl.Series(name="time", values=[j for j in range(dataset.gaze[i].frame.shape[0])]))

        
        sampling_rate = dataset.definition.experiment.sampling_rate
        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len,
                                )
        
        from sklearn.preprocessing import LabelEncoder
        label_names = np.array(group_names)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_names)                
        subjects = np.array(splitting_names)
    elif args.dataset == 'copco':
        label_path = config.COPCO_LABEL_PATH
        label_grouping = config.COPCO_LABEL_GROUPING
        instance_grouping = config.COPCO_INSTANCE_GROUPING
        splitting_criterion = config.COPCO_SPLITTING_CRITERION
        max_len = config.COPCO_MAXLEN
        
        # load labels
        label_df = pl.read_csv(config.COPCO_LABEL_PATH)
        label_df = label_df.with_columns(
            pl.when(
                # L1 without dyslexia
                (pl.col('native_language').is_in({'Danish'})) & (pl.col('dyslexia').is_in({'no'})))
                    .then(0)
                # L1 with dyslexia
                .when((pl.col('native_language').is_in({'Danish'})) & (pl.col('dyslexia').is_in({'yes'})))
                    .then(1)
                # L2
                .otherwise(2)
            .alias('classes')
        )
        label_df = label_df.rename(
                    {'subj': 'subject_id',
                    'comprehension_accuracy': 'acc_numeric',                  # text comprehension
                    'score_reading_comprehension_test': 'subj_acc_numeric',   #general reading comprehension
                    }
                )

        # Text Comprehension
        acc_mean = np.mean(list(label_df['acc_numeric']))
        label_df = label_df.with_columns(
            pl.when(
                # L1 without dyslexia
                (pl.col('acc_numeric') >= acc_mean))
                .then(1)        
                .otherwise(0)
            .alias('acc')
        )

        # General Reading Comprehension
        subj_acc_mean = np.nanmean(np.array(list(label_df['subj_acc_numeric']), dtype=np.float32))
        label_df = label_df.with_columns(
            pl.when(
                # L1 without dyslexia
                (pl.col('subj_acc_numeric') >= subj_acc_mean))        
                .then(1)
                .when((pl.col('subj_acc_numeric').is_null()))
                    .then(-1)
                .otherwise(0)
            .alias('subj_acc')
        )

        use_participants_df   = label_df.filter(pl.col(label_column) != -1)
        use_participants = list(use_participants_df['subject_id'])
        use_label = list(use_participants_df[label_column])
        subject_label_dict = dict()
        for i in range(len(use_participants)):
            part = use_participants[i]
            lab = use_label[i]
            subject_label_dict[int(part.replace('P',''))] = lab
        
        
        dataset = pm.Dataset("CopCo", path='data/CopCo')

        # BEGIN hack
        try:
            dataset.load(preprocessed=False,
                    )
        except:
            pass

        dataset.definition.time_column = 'time'
        dataset.definition.trial_columns = ['subject_id','speech_id','paragraph_id','trial_id']
        dataset.definition.time_unit = 'ms'
        dataset.definition.pixel_columns = ['x_right', 'y_right']

        import glob
        csv_dir = '/mnt/mlshare/prasse/aeye_git/eye-movement-preprocessing/data/CopCo/csvs/'
        csv_files = glob.glob(csv_dir + '*.csv')
        subject_id = []
        filepath = []
        for file in csv_files:
            subject_id.append(file.split('/')[-1].replace('P','').replace('.csv',''))
            filepath.append(file)
        out_df = pl.DataFrame({'subject_id':subject_id,
                               'filepath':filepath
                              }
                )
        dataset.fileinfo['gaze'] = out_df

        dataset.definition.custom_read_kwargs['gaze'] = {
                        'schema_overrides': {
                            'time': pl.Int64,
                            'x_right': pl.Float32,
                            'y_right': pl.Float32,
                            'pupil_right': pl.Float32,
                        },
                        'separator': ',',
                    }

        dataset.definition.filename_format_schema_overrides['gaze'] = {
                        'subject_id': int,
                        'speech_id': int,
                        'paragraph_id': int,
                        'trial_id': int,
                    }        

        try:
            dataset.load_gaze_files(preprocessed=False,
                    )
        except:
            dataset.download()
            dataset.load_gaze_files(preprocessed=False,
                    )

        sampling_rate = dataset.definition.experiment.sampling_rate
        deleted_instances = 0
        instance_count = 0
        # Preprocessing
        # delete screens with errors (time difference not constant)
        for i in tqdm(np.arange(len(dataset.gaze))):
            cur_gaze_df = dataset.gaze[i].frame.with_row_index()
            delete_ids = []
            for name, data in cur_gaze_df.group_by(instance_grouping):
                timesteps_diff = np.diff(list(data['time']))
                number_steps = len(np.unique(timesteps_diff))
                if number_steps > 1:
                    delete_ids += list(data['index'])
                    deleted_instances+= 1
                instance_count += 1
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_row_index().filter(~pl.col("index").is_in(delete_ids))            
        print(' === Evaluating model ===')
        print('    deleted instances: ' + str(deleted_instances) +\
                ' (' + str(np.round(deleted_instances/instance_count*100.,decimals=2)) + '%)')
        # END hack
        
        # transform pixel coordinates to degrees of visual angle
        dataset.pix2deg()

        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len,
                                )
        
        
        # create label
        y = []
        subjects = []
        use_ids = []
        for i in range(feature_matrix.shape[0]):
            c_sub = int(splitting_names[i])
            if c_sub in subject_label_dict:
                y.append(subject_label_dict[c_sub])
                subjects.append(int(c_sub))
                use_ids.append(i)
        y = np.array(y)
        subjects = np.array(subjects)
        feature_matrix = feature_matrix[use_ids]
        group_names = np.array(group_names)[use_ids]
        splitting_names = np.array(splitting_names)[use_ids]
    elif args.dataset == 'potec':
        label_grouping = config.POTEC_LABEL_GROUPING
        instance_grouping = config.POTEC_INSTANCE_GROUPING
        splitting_criterion = config.POTEC_SPLITTING_CRITERION        
        label_path = config.POTEC_LABEL_PATH
        max_len = config.POTEC_MAXLEN

        # load labels
        label_df = pl.read_csv(label_path, separator='\t')
        
        # create labels
        reader_ids = list(label_df['reader_id'])
        reader_domains = list(label_df['reader_domain'])
        reader_domain_dict = dict()
        for i in range(len(reader_domains)):
            reader_domain_dict[reader_ids[i]] = reader_domains[i]
        
        print(' === Loading data ===')
        dataset = pm.Dataset("PoTeC", path='data/PoTeC')
        try:
            dataset.load(
                subset={'subject_id':reader_ids,
                }#,105]}
            )
        except:
            dataset.download()
            dataset.load(
                subset={'subject_id':reader_ids
                }
            )

        sampling_rate = dataset.definition.experiment.sampling_rate
        deleted_instances = 0
        instance_count = 0
        # Preprocessing
        # delete screens with errors (time difference not constant)
        for i in tqdm(np.arange(len(dataset.gaze))):
            cur_gaze_df = dataset.gaze[i].frame.with_row_index()
            delete_ids = []
            for name, data in cur_gaze_df.group_by(instance_grouping):
                timesteps_diff = np.diff(list(data['time']))
                number_steps = len(np.unique(timesteps_diff))
                if number_steps > 1:
                    delete_ids += list(data['index'])
                    deleted_instances+= 1
                instance_count += 1
            dataset.gaze[i].frame = dataset.gaze[i].frame.with_row_index().filter(~pl.col("index").is_in(delete_ids))            
        print(' === Evaluating model ===')
        print('    deleted instances: ' + str(deleted_instances) +\
                ' (' + str(np.round(deleted_instances/instance_count*100.,decimals=2)) + '%)')



        # transform pixel coordinates to degrees of visual angle
        dataset.pix2deg()

        # transform positional data to velocity data
        dataset.pos2vel()

        # detect events
        dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len,
                                )
        # create label
        if label_column == 'familarity':
            y = []
            for i in range(len(group_names)):
                split_name = group_names[i].split('_')
                cur_user = int(split_name[0])
                cur_text = split_name[1]
                if   reader_domain_dict[cur_user] == 'physics' and cur_text.startswith('p'):
                    y.append(1)
                elif reader_domain_dict[cur_user] == 'biology' and cur_text.startswith('b'):
                    y.append(1)
                elif reader_domain_dict[cur_user] == 'biology' and cur_text.startswith('p'):
                    y.append(0)
                elif reader_domain_dict[cur_user] == 'physics' and cur_text.startswith('b'):
                    y.append(0)
                else:
                    raise RuntimeError('Error: no conversion to label possible')
        else:
            raise RuntimeError('Error: label not implemented')
        subjects = splitting_names
    elif dataset_name == 'hbn':
        label_grouping = config.HBN_LABEL_GROUPING
        instance_grouping = config.HBN_INSTANCE_GROUPING
        splitting_criterion = config.HBN_SPLITTING_CRITERION        
        label_path = config.HBN_LABEL_PATH
        max_len = config.HBN_MAXLEN
        
        # load labels
        label_df = pl.read_csv(label_path, separator='\t')
        
        print(' === Loading data ===')
        
        dataset = pm.Dataset("HBN", path='data/HBN')

        try:
            dataset.load(
                # subset={'subject_id': ['NDARAJ807UYR', 'NDARMF939FNX', 'NDARZZ740MLM', 'NDARZZ740ML']},
                subset={'subject_id': label_df['Patient_ID'].to_list()}
            )
        except:
            dataset.download()
            dataset.load(
                # subset={'subject_id': ['NDARAJ807UYR', 'NDARMF939FNX', 'NDARZZ740MLM', 'NDARZZ740ML']},
                subset={'subject_id': label_df['Patient_ID'].to_list()}
            )

        sampling_rate = dataset.definition.experiment.sampling_rate
        print(' === Evaluating model ===')
        # transform pixel coordinates to degrees of visual angle
        dataset.pix2deg()
        
        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        try:
            dataset.detect(detection_method, **detection_params)
        except:
            for gaze_df_idx in range(len(dataset.gaze)):
                dataset.gaze[gaze_df_idx].frame = dataset.gaze[gaze_df_idx].frame.with_columns(
                    pl.col('time').cast(pl.Int32),
                )
            dataset.detect(detection_method, **detection_params)
        
        # create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len
                                )
        y = []
        subjects = []
        for i in range(len(group_names)):
            c_tuple = group_names[i]
            c_subject = splitting_names[i]
            y.append(label_df.filter(pl.col('Patient_ID') == c_subject)['label'][0])
            subjects.append(c_subject)

        y = np.array(y)
        subjects = np.array(subjects)
    elif dataset_name == 'gazeonfaces':
        label_grouping = config.GAZEONFACES_LABEL_GROUPING
        instance_grouping = config.GAZEONFACES_INSTANCE_GROUPING
        splitting_criterion = config.GAZEONFACES_SPLITTING_CRITERION
        max_len = config.GAZEONFACES_MAXLEN
        label_path = config.GAZEONFACES_LABEL_PATH

# load labels
        label_df = pl.read_csv(label_path)

        dataset = pm.Dataset("GazeOnFaces", path='data/GazeOnFaces')
        try:
            dataset.load(
                # subset = {'sub_id':[1,2,3,4,5,6,7,8,9,10]},
            )
        except:
            dataset.download()
            dataset.load(
                # subset = {'sub_id':[1,2,3,4]},
            )

        sampling_rate = dataset.definition.experiment.sampling_rate
# transform positional data to velocity data
        dataset.pix2deg()
        dataset.pos2vel()

# detect events
        dataset.detect(detection_method, **detection_params)

# create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                instance_grouping,
                                splitting_criterion,
                                max_len,
                                )

        from sklearn.preprocessing import LabelEncoder
        label_names = np.array(group_names)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_names)
        subjects = np.array(splitting_names)
    elif args.dataset == 'gazegraph':
        label_grouping = config.GAZEGRAPH_LABEL_GROUPING
        instance_grouping = config.GAZEGRAPH_INSTANCE_GROUPING
        splitting_criterion = config.GAZEGRAPH_SPLITTING_CRITERION
        max_len = None

        dataset = pm.Dataset("GazeGraph", path='data/GazeGraph')
        try:
            dataset.load(
#subset = {'subject_id':[1,2,3,4,5,6,7,8,9,10]},
            )
        except:
            dataset.download()
            dataset.load(
# subset = {#'subject_id':[1,2,3,4]},
            )

        sampling_rate = dataset.definition.experiment.sampling_rate
# transform positional data to velocity data
        dataset.pix2deg()
        dataset.pos2vel()

# detect events
        try:
            dataset.detect(detection_method, **detection_params)
        except:
            for gaze_df_idx in range(len(dataset.gaze)):
                dataset.gaze[gaze_df_idx].frame = dataset.gaze[gaze_df_idx].frame.with_columns(
                    pl.col('time').cast(pl.Int32),
                )

# create features
        feature_matrix, group_names, splitting_names = get_feature_matrix(dataset,
            sampling_rate,
            blink_threshold,
            blink_window_size,
            blink_min_duration,
            blink_velocity_threshold,
            feature_aggregations,
            detection_method,
            label_grouping,
            instance_grouping,
            splitting_criterion,
            max_len,
        )

        from sklearn.preprocessing import LabelEncoder
        label_names = np.array(group_names)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(label_names)                
        subjects = np.array(splitting_names)
    else:
        raise RuntimeError('Error: not implemented')
    
    y = np.array(y)
    print(' === Evaluating model ===')
    print(' === Number of subjects: ' + str(len(np.unique(subjects))) + ' === ')
    # split by subjects
    group_kfold = GroupKFold(n_splits=n_splits)
    aucs = []
    accs = []
    for i, (train_index, test_index) in enumerate(group_kfold.split(feature_matrix, y, subjects)):
        X_train = feature_matrix[train_index]
        y_train = y[train_index]
        X_test = feature_matrix[test_index]
        y_test = y[test_index]
        
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)]   = 0
        
        # rf
        rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid, verbose=grid_search_verbosity,
        )
        rf.fit(X_train, y_train)

        best_parameters = rf.best_params_
        pred_proba = rf.predict_proba(X_test)
        predictions = rf.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        accs.append(acc)
        
        if len(np.unique(y_train)) == 2:        
            fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba[:,1], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            print('AUC in fold ' + str(i+1) + ': ' + str(auc))
        joblib.dump({'y_test':y_test,
                     'pred':pred_proba},
                    save_dir + '/' + dataset_name + '_' + label_column + '_' + result_prefix +\
                                '_' + detection_param_string + '_fold_' + str(i) + '.joblib',
                    compress=3, protocol=2)
    
    if len(aucs) > 0:
        res_df = pl.DataFrame({'fold':np.arange(len(aucs)),
                      'auc': aucs})
    else:
        res_df = pl.DataFrame({'fold':np.arange(len(accs)),
                      'acc': accs})
    res_df.write_csv(save_dir + '/' + dataset_name + '_' + label_column + '_' + result_prefix +\
                    '_' + detection_param_string + '.csv')
    
    
    
def main() -> int:
    args = get_argument_parser()
    evaluate_model(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
