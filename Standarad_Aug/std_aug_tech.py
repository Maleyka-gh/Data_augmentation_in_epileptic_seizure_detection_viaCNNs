#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import sys
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat


sys.path.insert(0, r"C:/Users/seyidova/Desktop/new_code/Code")
import shutil

import seizures

path = r"/data/augmentation/"

LearnDf_path = r'/data/optimized/dataframes_learn'


def create_aug_path(technique):
    temp = path + "/" + "aug_" + technique

    if "aug_" + technique in os.listdir(path):

        shutil.rmtree(temp, ignore_errors=True)
        os.mkdir(temp)

    else:
        os.mkdir(temp)

    return str(temp)


def create_numpy_path(technique):
    temp = path + "/" + "numpy_" + technique

    if "numpy_" + technique in os.listdir(path):
        shutil.rmtree(temp, ignore_errors=True)
        os.mkdir(temp)

    else:
        os.mkdir(temp)

    return str(temp)


def create_data_for_aug_path():
    temp = path + "/" + "data_for_aug"

    if "data_for_aug" in os.listdir(path):
        shutil.rmtree(temp, ignore_errors=True)
        os.mkdir(temp)
    else:
        os.mkdir(temp)

    return str(temp)


# In[2]:


class Windowing:
    """
    This Windowing class contains all time deltas for preparing the data.
    """

    def __init__(self, window_length_sec=10, label_additional_time=10):
        self.window_length = pd.Timedelta(seconds=window_length_sec)
        self.label_add_time = pd.Timedelta(seconds=label_additional_time)
        self.label_offset = self.window_length + self.label_add_time


# In[3]:


def Extract_Data_for_Aug(path):
    dic1 = seizures.get_motor_seizures()

    path1 = create_data_for_aug_path()

    for patfolder in os.listdir(path):
        if patfolder in dic1.keys():

            os.mkdir(path1 + "/" + patfolder)

            for files in os.listdir(path + "/" + patfolder):

                dataframe = pd.read_pickle(path + "/" + patfolder + "/" + files)

                for k, v in dic1.items():
                    if k == patfolder:
                        t = 0
                        for x in v:
                            t += 1
                            ets = dataframe.loc[str(x["start"]):str(x["end"])]
                            if (ets.shape[0] != 0):
                                if str(t) in os.listdir(path1 + "/" + patfolder):
                                    ets.to_pickle(path1 + "/" + patfolder + "/" + str(t) + "/" + files)
                                else:
                                    os.mkdir(path1 + "/" + patfolder + "/" + str(t))
                                    ets.to_pickle(path1 + "/" + patfolder + "/" + str(t) + "/" + files)
    return path1


# In[5]:


# dfa_path = Extract_Data_for_Aug(LearnDf_path)


# In[6]:


# os.listdir(dfa_path)


# In[111]:


def DA_Rotation(data_dict):
    df_empty = pd.DataFrame()
    for k, v in data_dict.items():
        if "acc" in k:
            df_empty = pd.concat([df_empty, v], axis=1)
    X = df_empty.values
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    output = np.matmul(X, axangle2mat(axis, angle))
    output_df = pd.DataFrame(output, index=df_empty.index, columns=df_empty.columns)  # rotated acc_x , acc_y, acc_z
    final_dict = data_dict
    for column in output_df.columns:
        final_dict[column] = pd.DataFrame(output_df[
                                              column])  # replacing acc_x , acc_y , acc_z columns with rotated ones, temp, heart rate were already in final_dict.

    return final_dict


# In[112]:


def DA_Jitter(data_dict, sigma=0.02):
    new_data = dict()
    for k, v in data_dict.items():
        Noise = np.random.normal(loc=0, scale=sigma, size=v.shape)
        new_data[k] = pd.DataFrame(v.values + Noise, index=v.index, columns=v.columns).dropna()
    return new_data


# In[7]:


def DA_Permutation(data_dict, nPerm=5, minSegLength=10):
    df = pd.DataFrame()

    final_dict = dict()

    for v in data_dict.values():
        df = pd.concat([df, v], axis=1)

    X = df.values

    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)

    new_df = pd.DataFrame(X_new, index=df.index, columns=df.columns)

    for feat in df.columns:
        final_dict[feat] = pd.DataFrame(new_df[feat]).dropna()

    return final_dict


## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DA_MagWarp(data_dict, sigma):
    df_empty = pd.DataFrame()
    for k, v in data_dict.items():
        if "acc" in k:
            df_empty = pd.concat([df_empty, v], axis=1)
    X = df_empty.values
    c = GenerateRandomCurves(X, sigma)
    output_df = pd.DataFrame(X * c, index=df_empty.index, columns=df_empty.columns)
    final_dict = data_dict
    for column in output_df.columns:
        final_dict[column] = pd.DataFrame(output_df[
                                              column])  # replacing acc_x , acc_y , acc_z columns with rotated ones, temp, heart rate were already in final_dict.

    return final_dict


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0], (X.shape[0] - 1) / tt_cum[-1, 1], (X.shape[0] - 1) / tt_cum[-1, 2]]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def DA_TimeWarp(data_dict, sigma=0.2):
    df_empty = pd.DataFrame()
    for k, v in data_dict.items():
        if "acc" in k:
            df_empty = pd.concat([df_empty, v], axis=1)
    X = df_empty.values
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])

    output_df = pd.DataFrame(X_new, index=df_empty.index, columns=df_empty.columns)
    final_dict = data_dict
    for column in output_df.columns:
        final_dict[column] = pd.DataFrame(output_df[column])
    return final_dict


# In[8]:


def all_feat_available(path):
    count = 0
    for item in os.listdir(path):
        if is_feat_used(item):
            count += 1
    if count == 3:
        return True
    else:
        return False


# In[9]:


def is_feat_used(feat):
    """
    Checks if the feat is a feat to be used
    :param feat: name of feature
    :return: boolean
    """
    # return "acc" in feat or "temperature" in feat or "heart_rate" in feat
    return "acc" in feat


# In[10]:


def frequency_of(feat):
    """
    Returns frequency of a feature based on its name.
    :param feat: name of integer
    :return: int freq
    """
    if "acc" in feat:
        return 50
    else:
        return 1


# In[11]:


def load_data_in_dict(path):
    data_dict = dict()
    for files in os.listdir(path):
        df = pd.read_pickle(path + "/" + files)
        # freq = frequency_of(files)
        # df_reampled = df.resample(pd.Timedelta(seconds= 1/freq)).mean()
        data_dict[files] = df
    return data_dict


# In[12]:


def Do_Augmentatio(datapath, technique: str):
    num_new_sample = 1
    path = create_aug_path(technique)

    for pat_folder in os.listdir(datapath):
        os.mkdir(path + "/" + pat_folder)
        for event in os.listdir(datapath + "/" + pat_folder):
            if all_feat_available(datapath + "/" + pat_folder + "/" + event):
                os.mkdir(path + "/" + pat_folder + "/" + event)
                data_dict = load_data_in_dict(datapath + "/" + pat_folder + "/" + event)
                for i in range(num_new_sample):
                    sp = path + "/" + pat_folder + "/" + event + "/" + "sample" + str(i)
                    os.mkdir(sp)
                    if technique == "Jitter":
                        generated_data = DA_Jitter(data_dict)
                    elif technique == "Permutation":
                        generated_data = DA_Permutation(data_dict)
                    elif technique == "Rotation":
                        generated_data = DA_Rotation(data_dict)
                    elif technique == "Rot_Per":
                        generated_data = DA_Rotation(DA_Permutation(data_dict))
                    elif technique == "Per_TimeWarp":
                        generated_data = DA_Permutation(DA_TimeWarp(data_dict, sigma=0.2))
                    elif technique == "Rot_TimeWarp":
                        generated_data = DA_Rotation(DA_TimeWarp(data_dict, sigma=0.2))
                    elif technique == "Rot_Per_TimeWarp":
                        generated_data = DA_Rotation(DA_Permutation(DA_TimeWarp(data_dict, sigma=0.2)))
                    elif technique == "TimeWarp":
                        generated_data = DA_TimeWarp(data_dict, sigma=0.2)
                    elif technique == "MagWarp":
                        generated_data = DA_MagWarp(data_dict, sigma=0.2)
                    else:
                        shutil.rmtree(path)
                        return
                    for k, v in generated_data.items():
                        v.to_pickle(sp + "/" + k)
    return path


dfa_path = "/data/augmentation/data_for_aug_acc"
# In[39]:


# augDpath_Rot_Per= Do_Augmentatio(dfa_path,"Permutation")


# In[40]:


#### Combination of techniques
# augDpath_Per_TimeWarp= Do_Augmentatio(dfa_path,"Per_TimeWarp")
# augDpath_Rot_TimeWarp= Do_Augmentatio(dfa_path,"Rot_TimeWarp")
# augDpath_Rot_Per= Do_Augmentatio(dfa_path,"Rot_Per")
augDpath_Rot_Per_TimeWarp = Do_Augmentatio(dfa_path, "Rot_Per_TimeWarp")


# augDpath_jitter = Do_Augmentatio(dfa_path,"Jitter")


# In[13]:


# augDpath_Permutation = Do_Augmentatio(dfa_path,"Permutation")

# augDpath_TimeWarp = Do_Augmentatio(dfa_path,"TimeWarp")

# augDpath_Rotation= Do_Augmentatio(dfa_path,"Rotation")


# augDpath_MagWarp = Do_Augmentatio(dfa_path,"MagWarp")
# In[14]:


def correct_size_of_df(df, target_value):
    """
    sometimes interpolated Dataframes are too small or too large. Here the last entries are either removed or doubled,
    so that we receive the expected size (target_value)
    :param df: interpolated Dataframe
    :param target_value: expected num of entries in df
    :return: Boolean if the result is okay or not, and adjusted df.
    """
    # we need to cut off the dataframe tail
    while df.shape[0] > target_value:
        df = df.iloc[:-1]
    # we have to append the last value again. but not too often!
    i = 0
    while df.shape[0] < target_value:
        new_data = df.values[-1]
        new_index = (df.index[-1] - df.index[-2]) + df.index[-1]
        temp_df = pd.DataFrame(data=new_data, index=[
            new_index], columns=df.columns.to_list())
        df = df.append(temp_df)
        i = i + 1
        if i > 5:
            return False, df
    return True, df


def interpolate(df, freq, expected_values, method="linear"):
    """
    This method interpolates missing values. First the dataframe is resampled, so that the missing timestamps are
    created. After that the values for these new timestamps are interpolated and pasted. Some errors in the size of the
    new dataframe are still possible: The correct size is created by cropping of the last value or duplicate the last
    value of the dataframe.
    :param df: Dataframe with the data to interpolate
    :param freq: Frequency; so we can compute the num of missing values
    :param expected_values: Num of samples we expect after interpolation
    :param method: Method of interpolation
    :return: Interpolated DataFrame.
    """
    df = pd.DataFrame(df)
    df = df.resample(rule=pd.Timedelta(seconds=1 / freq)).mean()
    # todo think about handling of a max value of nans one behind one another. Limit in next call is possible.
    df = df.interpolate(method=method)  # , limit=5)
    is_ok, df = correct_size_of_df(df, expected_values)
    return is_ok, df


def frequency_of(feat):
    """
    Returns frequency of a feature based on its name.
    :param feat: name of integer
    :return: int freq
    """
    if "acc" in feat:
        return 50
    else:
        return 1


def get_freq_and_num_of(feat, window_length):
    """
    Computes the frequency of feat and the expected num of values considering the window_length and frequency.
    :param feat: Name of feature
    :param window_length: Length of window as pd.Timedelta
    :return: frequency and expected values.
    """
    freq = frequency_of(feat)
    expected_values = window_length.seconds * freq
    return freq, expected_values


def is_num_values_ok(real_num, target_num, k=0.1):
    """
    Checks if the available number real_num is okay: So if the num is not smaller than
    k=0.15 (15%) of target_num.
    :param real_num: A value which has to be checked
    :param target_num: The comparison value
    :param k: Puffer around target_num
    :return: boolean if real num is in the area below target_num
    """
    threshold = target_num - (k * target_num)
    if real_num >= threshold:
        return True
    return False


def put_data_in_np_format(data, window_settings):
    result_data = dict()
    label = 1
    for df in data:
        curr_df = df.dropna()
        result = curr_df.to_numpy()
        result_data[df.columns[0]] = result
    return result_data, label


def compute_next_start(time, indices):
    """
    Find the next date time (so its either =time or later than time) which is really in indices.
    :param time: starting time
    :param indices: list of time points
    :return: returns the next fitting time inside indices.
    """
    try:
        time = indices[np.searchsorted(indices, time)]
    except IndexError:
        a = indices[-1]
        return a
    return time


def do_window(path, Elements, class_labels, window_settings):
    df = pd.DataFrame()

    for feature in os.listdir(path):
        temp = pd.read_pickle(path + "/" + feature)
        df = pd.concat([df, temp], axis=1)

    normal_step = window_settings.window_length
    small_step = pd.Timedelta(seconds=1)
    current_start = df.index[0]  # initial

    while True:  # pythonic do-while
        data = list()
        trunc_df = df.truncate(
            before=current_start, after=current_start + window_settings.window_length)
        next_step = normal_step  # for computation of the next section
        for column in trunc_df:
            freq, expected_num = get_freq_and_num_of(
                column, window_settings.window_length)
            curr_ser = trunc_df[column]

            # check if the num of values are enough
            if not is_num_values_ok(curr_ser.dropna().shape[0], expected_num):
                data = list()  # empty list if one feature is not ok
                # the next step is smaller, because the truncate was not successful
                next_step = small_step
                break  # this truncate does not contain enough values; so we don't need the other columns to check

            # interpolate missing values
            is_ok, curr_ser = interpolate(curr_ser.dropna(), freq, expected_num)

            # too many values interpolated
            if not is_ok:
                data = list()
                next_step = small_step
                break
            data.append(curr_ser)

        if data:
            data, label = put_data_in_np_format(data, window_settings)
            class_labels.append(label)
            for k, v in data.items():
                Elements[k].append(v)

        current_start = compute_next_start(current_start + next_step, df.index)
        if current_start == df.index[-1]:  # if next start point is the last entry return the result
            return Elements, class_labels


# In[15]:


from collections import defaultdict


def compute_windows(augDpath, technique: str, window_settings):
    np_path = create_numpy_path(technique)
    for pat in os.listdir(augDpath):
        Elements = defaultdict(list)
        label = list()
        for events in os.listdir(augDpath + "/" + pat):
            for sample in os.listdir(augDpath + "/" + pat + "/" + events):
                Elements, label = do_window(augDpath + "/" + pat + "/" + events + "/" + sample, Elements, label,
                                            window_settings)
        for k, v in Elements.items():
            Elements[k] = np.stack(v, axis=0)
            np.save(os.path.join(np_path, pat + "_" + k + "_feats"), Elements[k])
        if len(label) != 0:
            label = np.array(label)
            np.save(os.path.join(np_path, pat + "_label"), label)


# In[16]:


w = Windowing(window_length_sec=10, label_additional_time=0)
# compute_windows(augDpath_jitter,"Jitter",w)


# In[17]:


# compute_windows(augDpath_Rotation,"Rotation",w)
# compute_windows(augDpath_Permutation,"Permutation",w)
# compute_windows(augDpath_TimeWarp,"TimeWarp",w)
# compute_windows(augDpath_MagWarp,"MagWarp",w)


### Combination of techniques
# compute_windows(augDpath_Per_TimeWarp,"Per_TimeWarp",w)
# compute_windows(augDpath_Rot_TimeWarp,"Rot_TimeWarp",w)
# compute_windows(augDpath_Rot_Per,"Rot_Per",w)
compute_windows(augDpath_Rot_Per_TimeWarp, "Rot_Per_TimeWarp", w)





