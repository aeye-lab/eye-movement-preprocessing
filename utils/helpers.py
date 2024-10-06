import os
import config.config as config


def write_json_file(contents, destination_filepath):
    """
    Writes dictionary Contents into json file
    :param contents: source dictionary
    :param destination_filepath: file path to write json file
    :return: 1
    """
    import json
    with open(destination_filepath, 'w') as fp:
        json.dump(contents, fp)
    return 1


def load_json_file(source_filepath):
    """
    Loads and returns content of a json file
    :param source_filepath: source file path
    :return: contents of json file
    """
    import json
    F = open(source_filepath)
    file = F.read()
    F.close()
    content = json.loads(file)
    return content


def get_detection_params(detection_method):
    out_strings = []
    if detection_method == 'ivt':
        params = config.ivt_detection_params
        minimum_durations = params['minimum_duration']
        velocity_thresholds = params['velocity_threshold']
        for minimum_duration in minimum_durations:
            for velocity_threshold in velocity_thresholds:
                out_str = '--minimum-duration ' + str(minimum_duration) +\
                            ' --velocity-threshold ' + str(velocity_threshold)
                out_strings.append(out_str)
    elif detection_method == 'idt':
        params = config.idt_detection_params
        minimum_durations = params['minimum_duration']
        dispersion_thresholds = params['dispersion_threshold']
        for minimum_duration in minimum_durations:
            for dispersion_threshold in dispersion_thresholds:
                out_str = '--minimum-duration ' + str(minimum_duration) +\
                            ' --dispersion-threshold ' + str(dispersion_threshold)
                out_strings.append(out_str)
    elif detection_method == 'microsaccades':
        params = config.microsaccades_detection_params
        minimum_durations = params['minimum_duration']
        for minimum_duration in minimum_durations:
            out_str = '--minimum-duration ' + str(minimum_duration)
            out_strings.append(out_str)
    return out_strings
            
def get_datset_labels(dataset):
    if dataset == 'sbsat':
        labels = config.SBSAT_LABELS
    elif dataset == 'gazebase':
        labels = config.GAZBASE_LABELS
    elif dataset == 'gazebasevr':
        labels = config.GAZBASEVR_LABELS
    elif dataset == 'potec':
        labels = config.POTEC_LABELS
    elif dataset == 'copco':
        labels = config.COPCO_LABELS
    elif dataset == 'copco':
        labels = config.COPCO_LABELS
    elif dataset == 'hbn':
        labels = config.HBN_LABELS
    elif dataset == 'gazeonfaces':
        labels = config.GAZEONFACES_LABELS
    else:
        raise NotImplementedError()
    return labels
