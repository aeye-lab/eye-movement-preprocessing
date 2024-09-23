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
from sklearn import metrics

import preprocessing.feature_extraction as feature_extraction
import config.config as config

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='sbsat')
    parser.add_argument('--label-grouping', '-s', type=str, default='book_name,subject_id')
    parser.add_argument(
        '--label-column', type=str, default='native',
        choices=['acc', 'difficulty', 'subj_acc', 'native'],
    )
    parser.add_argument(
        '--save-dir', type=str,
        default='results/',
    )
    parser.add_argument(
        '--sampling-rate', type=int, default=1000)
        
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
                    ):
                        
    event_name_dict = config.event_name_dict
    event_name_code_dict = config.event_name_code_dict
    detection_method_default_event = config.detection_method_default_event    
                        
    num_add = 1000
    group_names = []
    iter_counter = 0

    for i in tqdm(np.arange(len(dataset.gaze))):
        #print('i: ' + str(i))
        cur_gaze_df = dataset.gaze[i]
        try:
            cur_gaze_df.unnest()
        except:
            print('Warning: maybe already unnested')
        cur_gaze_df = cur_gaze_df.frame
        cur_event_df = dataset.events[i].frame

        # add events to gaze df
        # initialize event_type as None
        event_type = np.array([event_name_code_dict[detection_method_default_event[detection_method]] for _ in range(cur_gaze_df.shape[0])], dtype=np.int32)
        for event_id in range(cur_event_df.shape[0]):
            cur_event = cur_event_df[event_id]
            cur_onset_time = cur_event_df[event_id]['onset'][0]
            cur_offset_time = cur_event_df[event_id]['offset'][0]
            cur_onset_id = cur_gaze_df.with_row_index().filter(pl.col('time') == cur_onset_time)['index'][0]
            cur_offset_id = cur_gaze_df.with_row_index().filter(pl.col('time') == cur_offset_time)['index'][0]
            event_type[cur_onset_id:cur_offset_id] = event_name_code_dict[event_name_dict[cur_event_df[event_id]['name'][0]]]
        
        non_ids = np.where(cur_gaze_df['velocity_x'].is_null())[0]
        event_type[non_ids] = -1
        cur_gaze_df = cur_gaze_df.with_columns(pl.Series(name="event_type", values=event_type))
        
        for name, data in cur_gaze_df.group_by(label_grouping):
            #print('name: ' + str(name))
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
            group_names.append(name)
            feature_matrix[iter_counter,:] = combined_features
            iter_counter += 1
    feature_matrix = feature_matrix[0:iter_counter,:]
    feature_matrix[np.isnan(feature_matrix)] = 0.0
    return feature_matrix, group_names


def evaluate_model(args):
    label_grouping = list(args.label_grouping.split(','))
    dataset = args.dataset
    label_column = args.label_column
    save_dir = args.save_dir
    detection_method = args.detection_method    
    result_prefix = detection_method
    sampling_rate = args.sampling_rate
    
    # load config data
    label_path = config.SBSAT_LABEL_PATH
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
    
    # load labels
    label_df = pl.read_csv(label_path)
    
    
    print(' === Loading data ===')
    label_mean = np.mean(label_df[label_column].to_numpy())
    
    if args.dataset == 'sbsat':
        dataset = pm.Dataset("SBSAT", path='data/SBSAT')
        try:
            dataset.load()
        except:
            dataset.download()
            dataset.load()
        
        # transform pixel coordinates to degrees of visual angle
        dataset.pix2deg()
        
        # transform positional data to velocity data
        dataset.pos2vel()
        
        # detect events
        dataset.detect(detection_method)
        
        # create features
        feature_matrix, group_names = get_feature_matrix(dataset,
                                sampling_rate,
                                blink_threshold,
                                blink_window_size,
                                blink_min_duration,
                                blink_velocity_threshold,
                                feature_aggregations,
                                detection_method,
                                label_grouping,
                                )
        
        # create label
        label = list(label_df[label_column])
        label_group_list = dict()
        label_dict = dict()
        for group in label_grouping:
            label_group_list[group] = list(label_df[group])
        for i in range(label_df.shape[0]):
            cur_tuple = tuple(label_group_list[a][i] for a in label_group_list)
            label_dict[cur_tuple] = label[i]
        
        y = []
        subjects = []
        for c_tuple in group_names:
            y.append(label_dict[c_tuple])
            subjects.append(c_tuple[1])

        y = np.array(y)
        subjects = np.array(subjects)
        
        # binarize the label
        bin_label = np.zeros([len(y),])
        bin_label[y >= label_mean] = 1.
        y = bin_label
    else:
        raise RuntimeError('Error: not implemented')
    
    print(' === Evaluating model ===')
    # split by subjects
    group_kfold = GroupKFold(n_splits=n_splits)
    aucs = []
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
        
        fpr, tpr, _ = metrics.roc_curve(y_test, pred_proba[:,1], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        joblib.dump({'y_test':y_test,
                     'pred':pred_proba[:,1]},
                    save_dir + label_column + '_' + result_prefix + '_fold_' + str(i) + '.joblib',
                    compress=3, protocol=2)
        print('AUC in fold ' + str(i+1) + ': ' + str(auc))
    
    
    aucs = np.random.random([5,])
    res_df = pl.DataFrame({'fold':np.arange(len(aucs)),
                  'auc': aucs})
    res_df.write_csv(save_dir + '/' + label_column + '_' + result_prefix + '.csv')
    
    
    
def main() -> int:
    args = get_argument_parser()
    evaluate_model(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())