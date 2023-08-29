from collections import defaultdict
import json
import os
import numpy as np
from pathlib import Path
from Helper.evaluation import Evaluation


class ResultComputer:
    def __init__(self, folder_name):
        folder_name = Path(str(folder_name + "/"))
        methods = [f.path for f in os.scandir(folder_name) if f.is_dir()]  # noqa
        for m in methods:
            self.path = os.path.join(folder_name, str(m))
            orig_result = self.collect_results()
            result = self.compute_statistical_features(orig_result)
            result.update(orig_result)
            # save
            result = Evaluation.make_json_serializable(result)
            with open(self.path + '/result.json', 'w') as fp:
                json.dump(result, fp)

    def collect_results(self):
        """
        Collects all results (k-fold CV) in a dictionary of lists.
        :return: dictionary of lists
        """
        # following keys are in the result_dict but shouldn't be used for the result over all K's from k-fold-cv
        delete_keys = ['k_fold_train', 'k_fold_test', 'class_weight', 'resampled', 'imbalance',
                       'Scaler (normalize data)', 'Num_Classes', 'Shape_train_ACC_X_Batch-Rows-Width',
                       'Shape_train_ACC_Y_Batch-Rows-Width', 'Shape_train_ACC_Z_Batch-Rows-Width',
                       'Shape_train_HR_Batch-Rows-Width', 'Shape_train_TEMP_Batch-Rows-Width',
                       'Shape_valid_ACC_X_Batch-Rows-Width', 'Shape_valid_ACC_Y_Batch-Rows-Width',
                       'Shape_valid_ACC_Z_Batch-Rows-Width', 'Shape_valid_HR_Batch-Rows-Width',
                       'Shape_valid_TEMP_Batch-Rows-Width', 'epochs', 'batch_size', 'Callbacks', 'optimizer', 'loss',
                       'metrics', 'validation_split', 'run_eagerly', 'verbose', 'KERAS_RESULTS',
                       'condition_positive_samples', 'condition_positive_seizures', 'condition_negative_samples']
        orig_result = defaultdict(list)
        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                if "result" in file:
                    with open(subdir + "/" + file, 'r') as f:
                        temp_result = json.load(f)
                        # first only store the original KERAS_Results:
                        keras_results = temp_result['KERAS_RESULTS']
                        for key, val in keras_results.items():
                            orig_result['ZZZ_keras_result_' + key].append(val)
                        # now the other results:
                        for key, val in temp_result.items():
                            if key not in delete_keys:
                                orig_result[key].append(val)
        return orig_result

    @staticmethod
    def compute_statistical_features(d):
        """
        Computes statistical features like mean...
        :param d: dictionary with list of values
        :return: new dictionary with statistical values.
        """
        result = dict()
        for key, val in d.items():
            nans = val.count('NaN')
            val = [elem for elem in val if elem != 'NaN']
            if val and key != "num_recognized_seizures":
                if nans == 0:
                    result[key + "_mean"] = np.mean(val)
                    result[key + "_median"] = np.median(val)
                    result[key + "_std"] = np.std(val)
                    result[key + "_var"] = np.var(val)
                    result[key + "_max"] = np.max(val)
                    result[key + "_min"] = np.min(val)
                else:
                    result[key + "_mean"] = str(np.nanmean(val)) + " + " + str(nans) + " NaN"
                    result[key + "_median"] = str(np.nanmedian(val)) + " + " + str(nans) + " NaN"
                    result[key + "_std"] = str(np.nanstd(val)) + " + " + str(nans) + " NaN"
                    result[key + "_var"] = str(np.nanvar(val)) + " + " + str(nans) + " NaN"
                    result[key + "_max"] = str(np.nanmax(val)) + " + " + str(nans) + " NaN"
                    result[key + "_min"] = str(np.nanmin(val)) + " + " + str(nans) + " NaN"
            else:
                result[key + "_mean"] = "All NaN"
                result[key + "_median"] = "All NaN"
                result[key + "_std"] = "All NaN"
                result[key + "_var"] = "All NaN"
                result[key + "_max"] = "All NaN"
                result[key + "_min"] = "All NaN"
        return result
