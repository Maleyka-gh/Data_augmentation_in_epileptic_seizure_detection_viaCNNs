import os
import json

import numpy as np
import pandas as pd

def sort_list(vals, reverse):
    num_nans = vals.count('NaN')
    non_nan_list = [x for x in vals if not isinstance(x, str)]
    non_nan_list.sort(reverse=reverse)
    if num_nans > 0:
        non_nan_list = non_nan_list + (['NaN'] * num_nans)
    return non_nan_list


def is_sort_reverse(key):
    if 'false_negative_rate' in key or 'false_positive_rate' in key or 'false_discovery_rate' in key \
            or 'false_omission_rate' in key or 'error_rate' in key:
        return False
    else:
        return True

def remove_nan_info(vals):
    res = []
    for val in vals:
        if not isinstance(val, str):
            res.append(val)
        elif val == "All NaN":
            res.append('NaN')
        elif val[-3:] == 'NaN':
            res.append(float(val[:-8]))
    return res

def compute_places(values, sort_reverse):
    orig = values.copy()
    orig = remove_nan_info(values)
    values = sort_list(orig, reverse=sort_reverse)
    result = []
    for elem in orig:
        idx = values.index(elem) + 1
        result.append(idx)
    return result

def compute_placements(dicts):
    placements = dict()
    for key, values in dicts.items():
        places = compute_places(values.copy(), sort_reverse=is_sort_reverse(key))
        placements[key] = places
    return placements

def remove_num_of_samples(dicts):
    for k in list(dicts.keys()):
        if 'tp_' in k or 'fp_' in k or 'tn_' in k or 'fn_' in k:
            del dicts[k]
    return dicts

def remove_keras_results(dicts):
    for k in list(dicts.keys()):
        if 'ZZZ' in k:
            del dicts[k]
    return dicts

def remove_keys(dict, keep):
    if keep:
        for k in list(dict.keys()):
            if not any(elem in k for elem in keep):
                del dict[k]
    return dict

def load_dicts(files, name):
    dicts = dict()
    json_files = []
    for f in files:
        with open(f + "/" + name, 'r') as json_file:
            temp = json.load(json_file)
            json_files.append(temp)
    for k in temp.keys():
        dicts[k] = [d[k] for d in json_files]
    return dicts

if __name__ == "__main__":
    """todo"""
    directory = "/data/WORKModel/19-07-2022_22-34-22/CNN_meisel_Multi_FINAL/"
    list_of_subs = [directory + subdir for subdir in os.listdir(directory)]
    dicts = load_dicts(list_of_subs, 'result.json')
    dicts = remove_keys(dicts, keep=['mean', 'median'])
    dicts = remove_keras_results(dicts)
    dicts = remove_num_of_samples(dicts)
    placements = compute_placements(dicts)

    # to pandas
    dicts = pd.DataFrame(dicts).T
    placements = pd.DataFrame(placements).T

    dicts.to_csv(directory+"results_overview.csv")
    placements.to_csv(directory + "placements_overview.csv")

    print('YOLO')