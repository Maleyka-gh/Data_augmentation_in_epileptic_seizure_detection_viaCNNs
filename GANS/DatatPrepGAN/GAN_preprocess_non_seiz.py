#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import os
import numpy as np
import sys
from scipy.interpolate import CubicSpline  # for warping

# sys.path.insert(0,r"C:\Users\seyidova\Desktop\new_code\Code")
import shutil

import seizures

# Output directory path
path = r"/data/Data_prep_for_gan/"

# Lean_df_patha
LearnDf_path = r"/data/optimized/dataframes_learn_acc"


def create_numpy_path(technique):
    temp = path + "/" + "numpy_" + technique

    if "numpy_" + technique in os.listdir(path):
        shutil.rmtree(temp, ignore_errors=True)
        os.mkdir(temp)

    else:
        os.mkdir(temp)

    return str(temp)


def create_data_for_aug_path():
    temp = path + "/" + "GAN_non_seiz_data"

    if "GAN_non_seiz_data" in os.listdir(path):
        shutil.rmtree(temp, ignore_errors=True)
        os.mkdir(temp)
    else:
        os.mkdir(temp)

    return str(temp)


# In[41]:


class Windowing:
    """
    This Windowing class contains all time deltas for preparing the data.
    """

    def __init__(self, window_length_sec=10, label_additional_time=5):
        self.window_length = pd.Timedelta(seconds=window_length_sec)
        self.label_add_time = pd.Timedelta(seconds=label_additional_time)
        self.label_offset = self.window_length + self.label_add_time


# In[42]:


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
                        for x in v:
                            if dataframe[str(x["start"]):str(x["end"])].empty:
                                continue
                            else:
                                temp = dataframe[str(x["start"]):str(x["end"])].index
                                cond = dataframe.index.isin(temp)
                                dataframe = dataframe[~cond]
                dataframe.to_pickle(path1 + "/" + patfolder + "/" + files)
    return path1


# In[43]:


# dfa_path = Extract_Data_for_Aug(LearnDf_path)


# In[44]:


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
    if freq == 50:
        expected_values = 512
    else:
        expected_values = 16
    return freq, expected_values


def is_num_values_ok(real_num, target_num, k=0):
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
    label = 0
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
        if any(temp.index.duplicated()):
            temp = temp[~temp.index.duplicated(keep='first')]
        df = pd.concat([df, temp], axis=1)
        df = df.sort_index()
    print(df)

    normal_step = window_settings.window_length
    small_step = pd.Timedelta(seconds=1)
    current_start = df.index[0]  # initial

    while True:  # pythonic do-while
        data = list()
        trunc_df = df.truncate(before=current_start, after=current_start + window_settings.window_length)
        next_step = normal_step  # for computation of the next section/ 50 percent overlap
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


# In[45]:


# augDpath = dfa_path
# li_pat =['BN_017', 'BN_041', 'BN_046', 'BN_057', 'BN_071', 'BN_082', 'BN_083', 'BN_084', 'BN_086', 'BN_100', 'BN_103', 'BN_106', 'BN_107', 'BN_113', 'BN_123', 'BN_138', 'BN_141', 'BN_146', 'BN_159', 'BN_160', 'BN_166', 'BN_167', 'BN_169', 'BN_170', 'BN_179', 'BN_186']
# li_pat = ['BN_070']
from collections import defaultdict


def compute_windows(augDpath, technique: str, window_settings):
    np_path = create_numpy_path(technique)
    for pat in os.listdir(augDpath):

        Elements = defaultdict(list)
        label = list()
        Elements, label = do_window(augDpath + "/" + pat, Elements, label, window_settings)
        for k, v in Elements.items():
            Elements[k] = np.stack(v, axis=0)
            np.save(os.path.join(np_path, pat + "_" + k + "_feats"), Elements[k])
        if len(label) != 0:
            label = np.array(label)
            np.save(os.path.join(np_path, pat + "_label"), label)


# In[48]:


w = Windowing(window_length_sec=10.24, label_additional_time=5)

# In[ ]:


compute_windows('/data/Data_prep_for_gan/GAN_non_seiz_data', "epl_non_seiz_512_k0", w)

# In[ ]:




