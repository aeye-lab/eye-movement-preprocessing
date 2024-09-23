SBSAT_LABEL_PATH = 'data/18sat_labels.csv'

# we allow 3 event_types: 'Fixation', 'Saccade', or 'None'
event_name_dict = {
                    'fixation': 'Fixation',
                    'saccade' : 'Saccade',
                  }


event_name_code_dict = {
                'Fixation': 0,
                'Saccade': 1,
                None: -1,
                }

detection_method_default_event = {
                    'ivt': 'Saccade',
                    'idt': 'Saccade',
                    'microsaccades': 'Fixation',
                                 }
                                 
feature_aggregations = ['mean', 'std', 'median', 'skew', 'kurtosis']

blink_threshold=0.6
blink_window_size=100
blink_min_duration=1
blink_velocity_threshold=0.1



# learning params
param_grid={
        'n_estimators': [500, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [2, 4, 8, 16, 32, None],
        'criterion': ['entropy'],
        'n_jobs': [-1],
    }
grid_search_verbosity = 10
n_splits = 5