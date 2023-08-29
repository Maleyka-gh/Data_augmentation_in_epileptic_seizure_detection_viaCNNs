#!/usr/bin/env python
# coding: utf-8

# - In tts gan synthetic data is generated from random noise. We use three different overlap 0% , 50% and 70% data to train the model. Then we use these models to generate the synthetic data.We compare synthetic data with both train and test real seizures.
# - test set are the pastients did not attend in training
# - train set did attend in training
# -  2 points to consider:
# --- bets metrics model are saved where generated synhtetic seizure has a best match with a real test seizure
# --- since train patients are used in training, synthetic seizures should look more similar to train set real seizures- last checkpoint(may be)
#
#
#

# In[54]:


from dataLoader import *
from torch.utils import data

# In[55]:


train_set = Mond_data_loader(data_path=r'C:\Users\seyidova\Desktop\tts_gan\numpy_tts_seiz_numpy_0overlap',
                             is_normalize=False, data_mode='Train')

# In[56]:


real_data_loader = data.DataLoader(train_set, batch_size=1, num_workers=1,
                                   shuffle=True)  # pytorch built in library to prepare data

# #### Real data / in np format

# In[57]:


import matplotlib.pyplot as plt
import numpy as np

real_seiz = []

for i, (real_sig, label) in enumerate(real_data_loader):
    real_sig = real_sig.cpu().detach().numpy()
    sig = real_sig.reshape(real_sig.shape[1], real_sig.shape[3])
    sig = np.transpose(sig, (1, 0))
    #     sig1 = real_sig.reshape(real_sig.shape[3], real_sig.shape[1])

    if label[0] == 1:
        real_seiz.append(sig)
# real_seiz = np.array(real_seiz)
print(real_seiz[0].shape)
real = np.stack(real_seiz, axis=0)
real.shape


# real_seiz[0]==real[0]


# ### 0% overlap model

# #### Synthetic data

# In[68]:


def prep_syn(path):
    acc_x = np.load(os.path.join(path, 'run_1_acc_x.npy'))
    acc_y = np.load(os.path.join(path, 'run_1_acc_y.npy'))
    acc_z = np.load(os.path.join(path, 'run_1_acc_z.npy'))
    acc_x2 = np.load(os.path.join(path, 'run_2_acc_x.npy'))
    acc_y2 = np.load(os.path.join(path, 'run_2_acc_y.npy'))
    acc_z2 = np.load(os.path.join(path, 'run_2_acc_z.npy'))
    all_syn = np.concatenate([acc_x, acc_y, acc_z], axis=2)
    all_syn2 = np.concatenate([acc_x2, acc_y2, acc_z2], axis=2)
    syn = np.concatenate([all_syn, all_syn2], axis=0)
    print(syn.shape)
    return syn


syn = prep_syn(r'C:\Users\seyidova\Desktop\tts_gan\tts_70overlap100%')

# In[ ]:


# ### pca

# In[69]:


from visualizationMetrics import visualization

visualization(real, syn[:585, :, :], 'pca', 'Running-pca')

# ### t-sne

# In[70]:


visualization(real, syn[:585, :, :], 'tsne', 'Mond-data-tsne')

# ###  DTW calculation 3D

# #### Calculate dtw similarity btw  100 real train and synthetic seizures

# In[71]:


from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# from dtaidistance import dtw_ndim


# Define a function to calculate the euclidean distance between two 3-dimensional points
def euclidean_3d(point1, point2):
    return euclidean(point1, point2)


# Initialize a list to store the DTW distance between each real and synthetic time series
# dtw_distance = []


# for i in range(len(real_seiz)):
# #     for j in range(len(syn)):
#     distance= dtw_ndim.distance(real_time_series[i], synthetic_time_series[i])
#     dtw_distance.append(distance)
# print("DTW distance between real and synthetic time series:", np.mean(dtw_distance))

real_100 = real[:88, :, :]
syn_100 = syn[:88, :, :]


def cal_dtw(r_100, s_100):
    dtw_distance = []
    # Iterate through all the real and synthetic time series and calculate the DTW distance
    for i in range(len(r_100)):
        best_similarity = 1000
        for j in range(len(s_100)):
            distance, path = fastdtw(r_100[i], s_100[j], dist=euclidean_3d)
            if distance < best_similarity:
                best_similarity = distance
        dtw_distance.append(best_similarity)
    return np.mean(dtw_distance)


# #### Calculate best match for each sample and add to the list and take mean of this listÂ¶

# In[72]:


dtw_mean_train = cal_dtw(real_100, syn_100)
print(dtw_mean_train)

# #### Cos similarity calculation channelwise

# In[10]:


from math import sqrt


def square_rooted(x):  # noqa

    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def calc_feat(data):
    feat_list = []
    feat_list.append(np.median(data))
    feat_list.append(np.mean(data))
    feat_list.append(np.std(data))
    feat_list.append(np.var(data))
    feat_list.append(np.sqrt(np.mean(data ** 2)))
    feat_list.append(np.max(data))
    feat_list.append(np.min(data))
    return feat_list


def calc_feat_vec(syn, real_seiz):
    rx_feat, ry_feat, rz_feat = [], [], []
    sx_feat, sy_feat, sz_feat = [], [], []

    for i in syn:  # Iterate over the total number of syn sample(whole batch)
        sx_feat.append(calc_feat(i[:, 0]))
        sy_feat.append(calc_feat(i[:, 1]))
        sz_feat.append(calc_feat(i[:, 2]))

    for i in real_seiz:  # Iterate over the total number of real sample(whole batch)
        rx_feat.append(calc_feat(i[:, 0]))
        ry_feat.append(calc_feat(i[:, 1]))
        rz_feat.append(calc_feat(i[:, 2]))
    return rx_feat, ry_feat, rz_feat, sx_feat, sy_feat, sz_feat


def calc_cs(real_feat, syn_feat):
    list_cos_sim = []
    for a in real_feat:
        best_similarity = -1
        for b in syn_feat:
            cos_sim = cosine_similarity(a, b)
            list_cos_sim.append(cos_sim)
            if cos_sim > best_similarity:
                best_similarity = cos_sim
        list_cos_sim.append(best_similarity)
    avg_cos_sim = sum(list_cos_sim) / len(list_cos_sim)
    return avg_cos_sim


# In[11]:


rx_feat, ry_feat, rz_feat, sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100, syn_100)

x_similarity = calc_cs(rx_feat, sx_feat)
y_similarity = calc_cs(ry_feat, sy_feat)
z_similarity = calc_cs(rz_feat, sz_feat)
mean_similarity = np.mean([x_similarity, y_similarity, z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ", x_similarity)
print("Avg_cosine_Similarity for acc_y  :  ", y_similarity)
print("Avg_cosine_Similarity for acc_z  :  ", z_similarity)
print("Avg_cosine_Similarity for acc  :  ", mean_similarity)


# ### MSE calculation

# In[12]:


def compute_similarity(real, syn):
    mse = np.mean((real - syn) ** 2)
    return mse


def compute_mse(syn, real_seiz):
    best_mse = []
    for synthetic_sample in syn:
        best_match = None
        best_similarity = 1000
        for real_sample in real_seiz:
            similarity = compute_similarity(synthetic_sample,
                                            real_sample)  # Implement this function to compute the similarity between the two samples
            if similarity < best_similarity:
                best_similarity = similarity
        best_mse.append(best_similarity)
    final_mean_mse = np.mean(best_mse)
    return final_mean_mse


# In[13]:


mse_mean_train = compute_mse(real_100, syn_100)
print(mse_mean_train)

# ### 50% overlap model

# #### Synthetic data

# In[14]:


syn = prep_syn(r'C:\Users\seyidova\Desktop\tts_gan\tts_50overlap100%')

# #### pca

# In[15]:


from visualizationMetrics import visualization

visualization(real, syn[:585, :, :], 'pca', 'Running-pca')

# #### t-sne

# In[16]:


visualization(real, syn[:585, :, :], 'tsne', 'Mond-data-tsne')

# ###  DTW calculation 3D

# #### Calculate dtw similarity btw 100 real train and synthetic seizures

# In[17]:


real_100 = real[:585, :, :]
syn_100 = syn[:585, :, :]

# In[18]:


# dtw_mean_train= cal_dtw(real_100,syn_100)
# print(dtw_mean_train)


# #### Cos similarity calculation channelwise

# In[19]:


rx_feat, ry_feat, rz_feat, sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100, syn_100)

x_similarity = calc_cs(rx_feat, sx_feat)
y_similarity = calc_cs(ry_feat, sy_feat)
z_similarity = calc_cs(rz_feat, sz_feat)
mean_similarity = np.mean([x_similarity, y_similarity, z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ", x_similarity)
print("Avg_cosine_Similarity for acc_y  :  ", y_similarity)
print("Avg_cosine_Similarity for acc_z  :  ", z_similarity)
print("Avg_cosine_Similarity for acc  :  ", mean_similarity)

# ### MSE calculation

# In[20]:


mse_mean_train = compute_mse(real_100, syn_100)
print(mse_mean_train)

# ### 70% overlap model

# #### Synthetic data

# In[21]:


syn = prep_syn(r'C:\Users\seyidova\Desktop\tts_gan\tts_70overlap100%')

# #### pca

# In[22]:


from visualizationMetrics import visualization

visualization(real, syn[:585, :, :], 'pca', 'Running-pca')

# #### t-sne

# In[23]:


visualization(real, syn[:585, :, :], 'tsne', 'Mond-data-tsne')

# ###  DTW calculation 3D

# #### Calculate dtw similarity btw 100 real train and synthetic seizures

# In[24]:


real_100 = real[:585, :, :]
syn_100 = syn[:585, :, :]

# In[25]:


# dtw_mean_train= cal_dtw(real_100,syn_100)
# print(dtw_mean_train)


# #### Cos similarity calculation channelwise

# In[26]:


rx_feat, ry_feat, rz_feat, sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100, syn_100)

x_similarity = calc_cs(rx_feat, sx_feat)
y_similarity = calc_cs(ry_feat, sy_feat)
z_similarity = calc_cs(rz_feat, sz_feat)
mean_similarity = np.mean([x_similarity, y_similarity, z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ", x_similarity)
print("Avg_cosine_Similarity for acc_y  :  ", y_similarity)
print("Avg_cosine_Similarity for acc_z  :  ", z_similarity)
print("Avg_cosine_Similarity for acc  :  ", mean_similarity)

# ### MSE calculation

# In[27]:


mse_mean_train = compute_mse(real_100, syn_100)
print(mse_mean_train)

# In[ ]:




