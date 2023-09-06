#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import random
import pandas as pd

# path = r'/data/Data_prep_for_gan/numpy_tts_seiz_numpy_0overlap/'
def permutation(x, max_segments=10, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


# In[79]:


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def create_pat_list(path):
    pat_list = []
    for file in sorted(os.listdir(path)):
        if file[0:6] in pat_list or 'read' in file:
            continue
        else:
            pat_list.append(file[0:6])
    return pat_list


def load_all_numpy_data(pat_set, path):
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

def create_path(path,tech):
    p = os.path.join(path, 'numpy_' + tech)
    os.mkdir(p)
    return p

# path = r'C:\Users\seyidova\Desktop\augmentation\numpy_epl_GAN_Seiz_numpy_acc_only_500'
dir_path = r'/data/augmentation'
path = r'/data/Data_prep_for_gan/numpy_tts_seiz_numpy_0overlap'

patient_list = create_pat_list(path=path)
save_path = create_path(dir_path,tech='tech')

print(patient_list)
for patient in patient_list:
    print(patient)
    flag = True
    feat, _, _ = load_all_numpy_data(patient, path)
    if 'temperature' in feat.keys():
        flag = True
    else:
        flag = False

    feat_acc = np.concatenate([feat['acc_x'], feat['acc_y'], feat['acc_z']], axis=2)
    if flag:
        temp = feat['temperature']
        hr = feat['heart_rate']

    num_of_samples = 8
    acc_x, acc_y, acc_z, temperature, heart_rate = [], [], [], [], []
    for i in range(num_of_samples):
        aug_data = window_slice(feat_acc)
        acc_x.append(np.expand_dims(aug_data[:, :, 0], axis=2))
        acc_y.append(np.expand_dims(aug_data[:, :, 1], axis=2))
        acc_z.append(np.expand_dims(aug_data[:, :, 2], axis=2))
        if flag:
            temperature.append(temp)
            heart_rate.append(hr)

    label = np.ones(acc_x[0].shape[0] * num_of_samples)  # num of windows in original * num of times of samples
    np.save(os.path.join(save_path, patient + '_acc_x_' + 'feats.npy'), np.concatenate(acc_x, axis=0))
    np.save(os.path.join(save_path, patient + '_acc_y_' + 'feats.npy'), np.concatenate(acc_y, axis=0))
    np.save(os.path.join(save_path, patient + '_acc_z_' + 'feats.npy'), np.concatenate(acc_z, axis=0))
    np.save(os.path.join(save_path, patient + '_label.npy'), label)
    if flag:
        np.save(os.path.join(save_path, patient + '_temperature_' + 'feats.npy'), np.concatenate(temperature, axis=0))
        np.save(os.path.join(save_path, patient + '_heart_rate_' + 'feats.npy'), np.concatenate(heart_rate, axis=0))


# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_x.npy', np.expand_dims(x, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_y.npy', np.expand_dims(y, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_z.npy', np.expand_dims(z, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\labels', label)

# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_x.npy', np.expand_dims(x, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_y.npy', np.expand_dims(y, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\acc_z.npy', np.expand_dims(z, axis=2))
# np.save(r'C:\Users\seyidova\Desktop\augmentation\mw_26x\labels', label)