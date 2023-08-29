import copy
import os
from Nets_Orig_MultiInput import cnn_meisel_old, cnn_meisel_old_bfce, cnn_meisel_old_th_mov, cnn_meisel_old_bfce_th_mov, \
    cnn_meisel_bfce_th_mov, cnn_meisel_th_mov, cnn_meisel_bfce
from Nets_Orig_MultiInput import cnn_meisel
from Helper import Investigation_train
from copy import deepcopy
from datetime import datetime
from sklearn.utils import class_weight
import pandas as pd
import sklearn.utils
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import config
import seizures as seizure_collection
from Helper.compute_mean_results import ResultComputer
import random
from pandas.tseries.frequencies import to_offset
import numpy as np

# from new_code.Code.Learn.Nets_Orig_MultiInput import cnn_keras, cnn_meisel_old, cnn_tang, resnet_git, cnn_own, cnn_fawaz
# from new_code.Code.Learn.Nets_Orig_SingleInput import cnn_keras, cnn_meisel, cnn_tang, resnet_git, cnn_own, cnn_fawaz
# from new_code.Code.Learn.Nets_Change_MultiInput import cnn_meisel_tuner, cnn_meisel_tuned, cnn_meisel_TL_ACC_PREWEIGHTS, cnn_meisel_TL_ALL_PREWEIGHTSTOPMODEL, cnn_meisel_TL_ACC_PREWEIGHTSTOPMODEL, cnn_meisel_TL_ALL_PREWEIGHTS

"""the following part of code is added to this  file by me """

import pandas as pd

from CreateOrigDataFrames import create_dict_for_patient
import os


# import new_code.Code.DataPreparation.compute_data as cd


def create_list_of_pats():
    """This methods creates a list of all available patients by iterating through MOND_DATA//Data"""
    l_temp = list()
    for filename in os.listdir(config.DATA_PATH):
        if filename[0:2] == "BN":
            l_temp.append(filename)
    return l_temp


"""code added until here """


class Windowing:  # noqa
    """
    This Windowing class contains all time deltas for preparing the data.
    """

    def __init__(self):
        self.resampler_time = pd.Timedelta(seconds=1)
        self.window_length = pd.Timedelta(seconds=10)
        self.label_add_time = pd.Timedelta(seconds=10)
        self.label_offset = self.window_length + self.label_add_time
        self.expected_num_of_entries = self.window_length.seconds / self.resampler_time.seconds


def compute_class_weight(y_data):
    """
    Computes class weights for imbalanced data.
    :return: class weights
    """
    weights = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)
    return dict(enumerate(weights))


def truncates_to_ndarrays(feat, label, windowing):
    """
    Creates the features and labels in ndarrays splits of chosen size. First feat and label are resampled. Every new
    section is checked if the length is equal to the expected length. If so, the section is added to a list. The list
    is converted to a ndarray at the end.
    :param feat: dataframe with timestamp indices of the features.
    :param label: dataframe with one column (label)
    :param windowing: windowing class
    :return: 2 ndarrays, features and labels of the same size
    """

    # resampling and window length
    resampled_feat = feat.resample(windowing.window_length)
    resampled_label = label.resample(windowing.window_length)

    if resampled_feat.ax.equals(resampled_label.ax):  # check if label and values are the same
        feat_list = list()
        label_list = list()
        for i, (timestamp, df) in enumerate(resampled_feat):
            if df.shape[0] == windowing.expected_num_of_entries:  # only same size of truncates
                feat_list.append(df)
        for i, (timestamp, df) in enumerate(resampled_label):
            if df.shape[0] == windowing.expected_num_of_entries:
                label_list.append(int(df.any(axis=None)))
        features = np.array(feat_list)
        labels = np.array(label_list)
        return features, labels
    else:
        raise Exception("Indices of labels and values are not the same")


def create_label_df(key, val, windowing):
    """
    This method generates the labels corresponding to the values under consideration of the chosen offset before
    and after the classified seizures.
    :param key: Name of the patient, used to select the correct seizures of the seizure class
    :param val: Dataframe of the features with timestamp indices, only used for the timestamp indices.
    :param windowing: Windowing class, containing the offset values for the label
    :return: label -> boolean dataframe with timestamp index for every timestamp, which is also in val.
    """
    seizures = seizure_collection.get_seizures_to_use()[key]
    label = pd.DataFrame(index=val.index, data=0, columns=['label'])
    offset = windowing.label_offset
    for s in seizures:
        start = s['start'] - offset
        end = s['end'] + offset
        label[start:end]['label'] = 1
    return label


def resample(df, rule):
    """
    Resamples the dataframe df. Without nan
    :param df: Pandas dataframe with Timestamp index
    :param rule: Timedelta is the rule for resampling
    :return: resampled dataframe
    """
    offset = pd.Timedelta(seconds=rule.total_seconds() / 2)
    df = df.resample(rule=rule).mean()
    df.index = df.index + to_offset(offset)
    return df.dropna()


def is_in_inner_list(value, matrix):
    """
    Checks if value is in matrix or in any list inside matrix.
    :param value: any element
    :param matrix: a list or a list of lists
    :return: Boolean
    """
    if value in matrix:
        return True
    for list_elem in matrix:
        if value in list_elem:
            return True
    return False


def load_dataframes(pat_set, required_feats, windowing):
    """
    This method loads all dataframes used in set which are needed (required_feats). The data can be resampled.
    :param pat_set: list of all required patients
    :param required_feats: list of required features
    :param windowing: Class containing time of resampling
    :return: dictionary with key patient and value a concatenated dataframe of all features
    """
    res = dict()
    for pat in pat_set:
        feats_per_patient = list()
        for file in os.listdir(config.LEARN_DF + "/" + pat):
            if is_in_inner_list(file, required_feats):
                df = pd.read_pickle(config.LEARN_DF + "/" + pat + "/" + file)
                if windowing.resampler_time:
                    df = resample(df, windowing.resampler_time)
                feats_per_patient.append(df)
        df = pd.concat(feats_per_patient, axis=1)
        res[pat] = df
    return res


def is_epitect(device):
    """
    Checks is device is in-ear-sensor
    :param device: device name
    :return: boolean
    """
    return device == "epi_o2" or device == "epi_green"


def is_feat_used(feat):
    """
    Checks if the feat is a feat to be used
    :param feat: name of feature
    :return: boolean
    """
    return "acc" in feat or "temperature" in feat or "heart_rate" in feat


def create_feat_pat_list():
    """returns two lists. One contains all possible patient names; the other one contains all possible features."""
    list_of_feats = list(create_dict_for_patient().keys())
    list_of_feats.insert(0, "events")
    list_of_pats = create_list_of_pats()
    return list_of_pats, list_of_feats


def save_learn_df(pat_list):
    """
    Creates combined dataframes of every feature per patient and store them into config.LEARN_DF. Iterates through the
    patient list and loads the original file. Checks if this file is from the in-ear-sensor and combines the several
    observations per feature.
    :param pat_list: list of patient names
    """
    for pat in pat_list:
        data = dict()
        path = config.RAW_DF_FOLDER + "/" + pat + "/"
        meta = pd.read_pickle(path + "meta")
        for file in os.listdir(path):
            if file == "meta" or file == "events":
                continue
            num, feat = file[:8], file[9:]
            if is_feat_used(feat) and is_epitect(meta.at[num, 'device_id']):
                curr_df = pd.read_pickle(path + file)
                if feat not in data.keys():
                    data[feat] = pd.DataFrame()
                if any(curr_df.index.duplicated()):
                    curr_df = curr_df[~curr_df.index.duplicated(keep='first')]
                data[feat] = pd.concat([data[feat], curr_df], axis=0)  # noqa
        os.mkdir(config.LEARN_DF + "/" + pat)
        for key, value in data.items():
            value.to_pickle(config.LEARN_DF + "/" + pat + "/" + key)


def shuffle_dict(x, y):
    """
    Shuffles the data of the dictionary x and np array y.
    :param x: data in dictionary
    :param y: labels
    :return: x, y in shuffled
    """
    indices = [i for i in range(0, y.shape[0])]
    random.shuffle(indices)
    y = y[indices]
    for iteration, (key, val) in enumerate(x.items()):
        x[key] = val[indices]
    return x, y


def shuffle(x, y):
    """
    Shuffles the data x and y.
    :param x: data
    :param y: label
    :return: x, y
    """
    if isinstance(x, dict):
        return shuffle_dict(x, y)
    shape_3d = x.shape
    x = np.reshape(x, (shape_3d[0], shape_3d[1] * shape_3d[2]))
    x, y = sklearn.utils.shuffle(x, y)
    shape_2d = x.shape
    x = np.reshape(x, (shape_2d[0], shape_3d[1], shape_3d[2]))
    return x, y


def normalize_dict(x, scaler, fit):
    """
    Normalizes a dictionary. The input data x is a dictionary.
    For every input a single scaler is used! So first the required amount of deep copies of the scaler is produced.
    Then every value of x is normalized.
    :param x: input data a dictionary
    :param scaler: scaler object
    :param fit: boolean if the scaler is fitted on the data (train) or not (test)
    :return: scaled data as a dictionary
    """
    if not isinstance(scaler, list):
        scaler = [deepcopy(scaler) for i in range(0, len(x))]  # noqa
    for iteration, (key, val) in enumerate(x.items()):
        x[key], scaler[iteration] = normalize(val, scaler[iteration], fit)
    return x, scaler


def normalize(x, scaler, fit):
    """
    Normalizes the data using a defined scaler. If fit: normalize train; else normalize test.
    :param x: data
    :param scaler: scaler used to normalize
    :param fit: boolean, if train or test
    :return: normalized data
    """
    if isinstance(x, dict):
        return normalize_dict(x, scaler, fit)
    shape_3d = x.shape
    x = np.reshape(x, (shape_3d[0] * shape_3d[1], shape_3d[2]))
    if fit:
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)
    x = np.reshape(x, (shape_3d[0], shape_3d[1], shape_3d[2]))
    return x, scaler


def repeat_data(data_in, axis=1, style='max'):
    """
    This method repeats the data so that the size of data_in along the  axis is the same.
    :param data_in: input data (dictionary of different numpy arrays)
    :param axis: repeat along this axis.
    :param style: max for enlarge to maximum; min for shrinkage to minimum.
    :return: repeated / reshaped data.
    """
    if style != 'max':
        print('ERROR')
        return False
    # print(data_in.items())
    max_expansion = max([val.shape[axis] for key, val in data_in.items()])
    for key, val in data_in.items():
        repeats = max_expansion / val.shape[axis]
        data_in[key] = np.repeat(val, repeats=repeats, axis=axis)
    return data_in


def conflate_data(data_in):
    """
    This method receives a dictionary of numpy arrays. It reshapes the numpy arrays into the same size
    and concatenates them.
    :param data_in: Dictionary
    :return: list of keys of dictionary; new concatenated np array
    """
    # the dimensions of input has to be equal.
    data = repeat_data(data_in)
    # now conflate data of dictionaries
    names = list(data.keys())
    values = list(data.values())
    values = np.concatenate(values, 2)
    return values, names


def over_sample_data_dict(x, y):
    """
    Over samples the data in a dictionary. Therefore the values of dictionary are reshaped to same size.
    Than the under sampling is accomplished; only the index of the under sampling is used to extract the original
    data later.
    :param x: dictionary with data!
    :param y: label
    :return: x, y, indices
    """

    data, names = conflate_data(x.copy())

    _, _, indices = over_sample_data(data, y)
    indices.sort()

    for key, val in x.items():
        x[key] = val[indices]
    y = y[indices]
    return x, y, indices


def over_sample_data(x, y):
    """
    Over samples the data. Reshaping is necessary to 2d array.
    :param x: data
    :param y: label
    :return: x, y, indices
    """
    if isinstance(x, dict):
        return over_sample_data_dict(x, y)
    shape_3d = x.shape
    x = np.reshape(x, (shape_3d[0], shape_3d[1] * shape_3d[2]))
    under_sample = RandomOverSampler(sampling_strategy='auto')
    x, y = under_sample.fit_resample(x, y)
    indices = under_sample.sample_indices_
    shape_2d = x.shape
    x = np.reshape(x, (shape_2d[0], shape_3d[1], shape_3d[2]))
    return x, y, indices


def under_sample_data_dict(x, y):
    """
    Under samples the data in a dictionary. Therefore the values of dictionary are reshaped to same size.
    Than the under sampling is accomplished; only the index of the under sampling is used to extract the original
    data later.
    :param x: dictionary with data!
    :param y: label
    :return: x, y, indices
    """

    data, names = conflate_data(x.copy())

    _, _, indices = under_sample_data(data, y)
    indices.sort()

    for key, val in x.items():
        x[key] = val[indices]
    y = y[indices]
    return x, y, indices


### use undersampling  to reduce non-seizures the certain degree, donâ€™t balance both classes
# def under_sample_data(x, y):
#     """
#     Under samples the data. Reshaping is necessary to 2d array.
#     :param x: data
#     :param y: label
#     :return: x, y, indices
#     """
#     if isinstance(x, dict):
#         return under_sample_data_dict(x, y)
#     shape_3d = x.shape
#     x = np.reshape(x, (shape_3d[0], shape_3d[1] * shape_3d[2]))
#     # Create an instance of RandomUnderSampler with the desired ratio
#     ratio = {1: np.sum(y), 0: 5*np.sum(y)}  # ratio of minority to majority class
#     under_sample = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
#     x, y = under_sample.fit_resample(x, y)
#     indices = under_sample.sample_indices_
#     shape_2d = x.shape
#     x = np.reshape(x, (shape_2d[0], shape_3d[1], shape_3d[2]))
#     return x, y, indices


def under_sample_data(x, y):
    """
    Under samples the data. Reshaping is necessary to 2d array.
    :param x: data
    :param y: label
    :return: x, y, indices
    """
    if isinstance(x, dict):
        return under_sample_data_dict(x, y)
    shape_3d = x.shape
    x = np.reshape(x, (shape_3d[0], shape_3d[1] * shape_3d[2]))
    under_sample = RandomUnderSampler(sampling_strategy='majority')
    x, y = under_sample.fit_resample(x, y)
    indices = under_sample.sample_indices_
    shape_2d = x.shape
    x = np.reshape(x, (shape_2d[0], shape_3d[1], shape_3d[2]))
    return x, y, indices


def load_all_numpy_data(pat_set, path=config.LEARN_10_NoOverlap):
    """
    This method loads all computed numpy arrays. It creates two np arrays (features and labels) and fills them with the
    concatenated numpy arrays.
    :param pat_set: List of required patients
    :return: 3 np arrays (feats, labels, timestamps)
    """
    features = dict()
    labels = None
    times = dict()
    first_iter = True  # label, feature
    for file in sorted(os.listdir(path)):
        # print(file)
        if file[0:6] in pat_set:

            if "times" in file:
                curr_times = np.array(pd.read_pickle(os.path.join(path, file)))
                times[file[0:6]] = curr_times
                continue
            curr_np = np.load(os.path.join(path, file))
            if "label" in file:
                # print(file)
                if first_iter:
                    labels = curr_np
                    first_iter = False
                else:
                    # print(file)
                    labels = np.concatenate([labels, curr_np])
            else:
                # print(file)
                if 'run' in file:
                    name = file[6:-4]
                else:
                    name = file[7:-10]
                if name not in features:
                    features[name] = curr_np
                else:
                    features[name] = np.concatenate([features[name], curr_np])
    return features, labels, times


def create_path(technique):
    """
    Creates path for saving model and plots...
    :return: path
    """
    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + technique
    path = os.path.join(config.MODEL, time_stamp)
    os.makedirs(path)
    return path


def seizures_in(names, seizures_of_patients):
    """
    Computes how many seizures are in the list of patient names l.
    :param names: List of names of patient
    :param seizures_of_patients: Dictionary of num of seizures (value) per patient (key)
    :return: integer; num of seizures of those patients.
    """
    s = 0
    for pat in names:
        s += seizures_of_patients[pat]
    return s


def create_k_fold_cross_iter(k_, data):
    """
    This method creates two lists (one for training, for test).
    Each list contains lists with patient numbers.
    training [i] + test [i] -> all patients in data.
    :param k_: Num of lists in training and test
    :param data: list of all patients
    :return: two lists.
    """
    all_seizures = seizure_collection.get_seizures_to_use()
    seizures_per_patient = {pat: len(seizures) for pat, seizures in all_seizures.items() if pat in data}
    num_of_seizures = sum(seizures_per_patient.values())
    seizures_per_iter_in_test = num_of_seizures // k_
    train_set = []
    test_set = []
    random.shuffle(data)
    for i in range(0, k_):
        test_temp = []
        train_temp = []
        for elem in data:
            if (not test_temp or seizures_in(test_temp, seizures_per_patient) < seizures_per_iter_in_test) \
                    and not any(elem in sublist for sublist in test_set):
                test_temp.append(elem)
            else:
                train_temp.append(elem)
        test_set.append(test_temp)
        train_set.append(train_temp)
    return train_set, test_set


def hard_coded_test_set():
    """
    Hard coded test set as a dictionary. Key: Patient name. Value: Num of seizures of patient.
    :return: dictionary of test set patients.
    """
    # return {"BN_123": 7, "BN_046": 1, "BN_082": 3, "BN_141": 2, "BN_169": 2}
    # return {'BN_011': 3, 'BN_012': 1, 'BN_031': 1, 'BN_103': 2, 'BN_107': 1, 'BN_160': 4, 'BN_166': 1, 'BN_167': 9, 'BN_017': 1}
    return {'BN_011': 3, 'BN_012': 1, 'BN_103': 2, 'BN_107': 1, 'BN_160': 4, 'BN_166': 1, 'BN_167': 9, 'BN_017': 1}


def create_sets(min_train=0.7):
    """
    Creates train, valid and test set for learning. min_train of all 110 seizures are in the train set.
    Test set is hard coded and contains 15 seizures of 5 patients. train and valid set are filled randomly per patient.
    :param min_train: minimum of percentage for num of seizures in train set.
    :return: 3 lists of patient names.
    """
    all_seizures = seizure_collection.get_seizures_to_use()  # 110 seizures in 46 patients
    num_of_seizures = {pat: len(seizures) for pat, seizures in all_seizures.items()}
    total_seizures = sum(num_of_seizures.values())

    hard_coded_test = hard_coded_test_set()
    train_set = {}
    num_of_seizures = {key: value for key, value in num_of_seizures.items() if key not in hard_coded_test}
    while True:  # pythonic do-while loop
        random_key = random.choice(list(num_of_seizures.keys()))
        train_set[random_key] = num_of_seizures[random_key]
        num_of_seizures.pop(random_key)
        if sum(train_set.values()) >= min_train * total_seizures:  # break condition
            return list(train_set.keys()), list(num_of_seizures.keys()), list(hard_coded_test.keys())


if __name__ == "__main__":
    """First there are three sets created. Also the params for the windowing and features are determined.
    The data is loaded and then used for the CNN."""

    """"
     In "main"  only the following functions are called : 

     * create_k_fold_cross_iter 
     * load_all_numpy_data 
     * under_sample_data
     * normalize

      """

    # list_of_pats, list_of_feats = create_feat_pat_list()
    # save_learn_df(list_of_pats)

    train, valid, test = create_sets()

    k_fold_train, k_fold_test = create_k_fold_cross_iter(k_=5,
                                                         data=train + valid + test)  # we added hard coded test set as well

    # print('traning', k_fold_train)
    # print('validation', valid)
    # print('testdata', k_fold_test)
    # tensorflow_test:
    print('###########################################################################################################')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # switch off GPU
    import tensorflow as tf

    print(tf.__version__)
    print(tf.test.is_built_with_cuda())
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('###########################################################################################################')

    result = dict()
    result['k_fold_train'] = k_fold_train
    result['k_fold_test'] = k_fold_test

    result_path = create_path('k5fold_acc_addexp')  # hard coded test set included
    print(result_path)

    # ONLY UNDER SAMPLING
    for k, (train, valid) in enumerate(zip(k_fold_train, k_fold_test)):
        # load numpy files
        print(k)
        train_x, train_y, train_times = load_all_numpy_data(train, '/data/optimized/Numpy_Learn_10s0Overlap_acc')
        Investigation_train.trainset_investigation(train_x, train_y, train_times, result_path, k=k)
        cw = None  # compute_class_weight(train_y)
        valid_x, valid_y, valid_times = load_all_numpy_data(valid, '/data/optimized/Numpy_Learn_10s0Overlap_acc')
        result['resampled'] = 'Original; acc has 50 Hz, rest 1 Hz.'
        # SAMPLING TRAIN DATA

        print('classes in dataset /  labels : ', np.unique(train_y))
        print("Number of windows with seizure is : ", np.count_nonzero(train_y == 1))

        # print('train data',train_x)

        # print('Before under_sampling , label array : ', np.shape(train_y))
        # print('temperature and heart rate data :', train_x['temperature'].shape)
        # print('acceleration data ' , train_x['acc_x'].shape)

        train_x, train_y, _ = under_sample_data(train_x, train_y)  # noqa

        # print('After under_sampling, label array :',np.shape(train_y))
        # print('temperature and heart rate data :' , train_x['temperature'].shape)
        # print('acceleration data ', train_x['acc_x'].shape)

        result['imbalance'] = 'under sampling'

        # NORMALIZE DATA
        m_scaler = StandardScaler()  # noqa
        train_x, m_scaler = normalize(train_x, m_scaler, fit=True)
        valid_x, m_scaler = normalize(valid_x, m_scaler, fit=False)  # noqa

        # SHUFFLE TRAIN DATA
        train_x, train_y = shuffle(train_x, train_y)
        Investigation_train.plot_train_distribute(result_path, train_x, train_y, k=k, name='train')
        Investigation_train.plot_train_distribute(result_path, valid_x, valid_y, k=k, name='test')

        print(result)

        cnn_meisel.CNN(result, train_x, train_y, valid_x, valid_y, valid_times,
                       result_path, k, cw=cw, name='FINAL')

        #
        # cnn_meisel_bfce.CNN(result, train_x, train_y, valid_x, valid_y, valid_times,
        #                    result_path, k, cw=cw, name='FINAL_bfce')
        #
        #
        #
        # cnn_meisel_th_mov.CNN(result, train_x, train_y, valid_x, valid_y, valid_times,
        #                         result_path, k, cw=cw, name='FINAL_th_mov')
        #
        #
        # cnn_meisel_bfce_th_mov.CNN(result, train_x, train_y, valid_x, valid_y, valid_times,
        #                           result_path, k, cw=cw, name='FINAL_th_mov_bfce')

        # # ONLY PRE WEIGHTS
        # try:
        #     Code.Learn.Nets_Change_SingleInput.cnn_meisel_TL_ACC_PREWEIGHTS.CNN(result, train_x, train_y, valid_x,
        #                                                                              valid_y, valid_times, result_path,
        #                                                                             k, cw=cw, name="ACC_Preweights")
        # except:
        #     pass
        #
        # try:
        #     Code.Learn.Nets_Change_MultiInput.cnn_meisel_TL_ALL_PREWEIGHTS.CNN(result, train_x, train_y, valid_x,
        #                                                                             valid_y, valid_times, result_path,
        #                                                                            k, cw=cw, name="ALL_Preweights")
        # except:
        #     pass
        #
        # # Pre Weights and Top Model
        # try:
        #     Code.Learn.Nets_Change_MultiInput.cnn_meisel_TL_ACC_PREWEIGHTSTOPMODEL.CNN(result, train_x, train_y, valid_x,
        #                                                                          valid_y, valid_times, result_path,
        #                                                                          k, cw=cw, name="ACC_Preweights_TopModel")
        # except:
        #     pass
        #
        # try:
        #     Code.Learn.Nets_Change_MultiInput.cnn_meisel_TL_ALL_PREWEIGHTSTOPMODEL.CNN(result, train_x, train_y, valid_x,
        #                                                                          valid_y, valid_times, result_path,
        #                                                                          k, cw=cw, name="ALL_Preweights_TopModel")
        # except:
        #     pass
        #
        # # Standard :)
        # try:
        #     Code.Learn.Nets_Change_MultiInput.cnn_meisel_tuned.CNN(result, train_x, train_y, valid_x,
        #                                                                          valid_y, valid_times, result_path,
        #                                                                          k, cw=cw, name="Standard")
        # except:
        #     pass

    ResultComputer(result_path)
    print('FINISHED :)')

    '''
    for item in train_x.values():
        print(type(item))
        x1 = np.reshape(item, (len(item)*10, item.shape[2]))
        df = pd.DataFrame(x1)
        print(df)


   # df.plot(kind='line' , xlabel='Numbers' , ylabels='')
    # Plot from CSV
    import matplotlib.pyplot as plt
    cols = df.columns
    figure,ax1 = plt.subplots()
    ax1.plot(df[cols[0]])
    plt.show()'''



