# SBSAT DATA
SBSAT_LABEL_PATH = 'data/18sat_labels.csv'
SBSAT_LABEL_GROUPING = ['book_name', 'subject_id']
SBSAT_INSTANCE_GROUPING = ['book_name', 'subject_id', 'screen_id']
SBSAT_SPLITTING_CRITERION = 'subject_id'
SBSAT_LABELS = ['acc', 'difficulty', 'subj_acc', 'native']
SBSAT_MAXLEN = None

HBN_LABEL_PATH = 'data/sub_sel_classif.csv'
HBN_LABEL_GROUPING = ['subject_id']
HBN_INSTANCE_GROUPING = ['subject_id', 'video_id']
HBN_SPLITTING_CRITERION = 'subject_id'
HBN_LABELS = ['label']
HBN_MAXLEN = None


# GAZEBASE DATA
GAZEBASE_LABEL_GROUPING = ['task_name']
GAZEBASE_INSTANCE_GROUPING = ['round_id', 'subject_id','session_id','task_name']
GAZEBASE_SPLITTING_CRITERION = 'subject_id'
GAZBASE_LABELS = ['task_name']
GAZEBASE_MAXLEN = 10000

# GAZEBASEVR DATA
GAZEBASEVR_LABEL_GROUPING = ['task_name']
GAZEBASEVR_INSTANCE_GROUPING = ['round_id', 'subject_id','session_id','task_name']
GAZEBASEVR_SPLITTING_CRITERION = 'subject_id'
GAZBASEVR_LABELS = ['task_name']
GAZEBASEVR_MAXLEN = 2500

# COPCO
COPCO_LABEL_PATH = 'data/participant_stats.csv'
COPCO_LABEL_GROUPING = ['subject_id']
COPCO_INSTANCE_GROUPING = ['subject_id', 'speech_id', 'paragraph_id']
COPCO_SPLITTING_CRITERION = 'subject_id'
COPCO_LABELS = ['acc', 'subj_acc', 'classes', 'l1vsl2', 'dyslexia']
COPCO_MAXLEN = None

# POTEC
POTEC_LABEL_PATH = 'data/potec_label.tsv'
POTEC_LABEL_GROUPING = ['subject_id', 'text_id']
POTEC_INSTANCE_GROUPING = ['subject_id', 'text_id']
POTEC_SPLITTING_CRITERION = 'subject_id'
POTEC_LABELS = ['familarity'] # familarity: label == 1, if text from own field
POTEC_MAXLEN = None

# GAZEONFACES DATA
GAZEONFACES_LABEL_PATH = 'data/observer_info'
GAZEONFACES_LABEL_GROUPING = ['sub_id']
GAZEONFACES_INSTANCE_GROUPING = ["sub_id", "trial_id"]
GAZEONFACES_SPLITTING_CRITERION = 'sub_id'
GAZEONFACES_LABELS = ['gender']
GAZEONFACES_MAXLEN = 10000

# GAZEGRAPH DATA
GAZEGRAPH_LABEL_GROUPING = ['task']
GAZEGRAPH_INSTANCE_GROUPING = ['subject_id', 'task']
GAZEGRAPH_SPLITTING_CRITERION = 'subject_id'
GAZEGRAPH_LABELS = ['task']

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
        'max_depth': [32, 64, 128, None],
        'criterion': ['entropy'],
        'n_jobs': [-1],
    }
grid_search_verbosity = 10
n_splits = 5



microsaccades_detection_params = {
                        'minimum_duration': [5, 10, 15, 20]
                        }
idt_detection_params = {
                        'minimum_duration': [10, 50, 100, 200],
                        'dispersion_threshold': [0.5, 1.0, 1.5],
                       }
ivt_detection_params = {
                        'minimum_duration': [10, 50, 100, 200],
                        'velocity_threshold': [5., 10., 20., 30., 40.],
                       }
