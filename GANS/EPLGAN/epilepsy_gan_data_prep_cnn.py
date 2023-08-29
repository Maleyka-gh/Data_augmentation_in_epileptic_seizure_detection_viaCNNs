#!/usr/bin/env python
# coding: utf-8

# ### Prepare synthetic data for CNN

# 1. Compare train set patients' real seizures & train set patients' synthetic seizures from each overlap model
# 2. Compare hard coded test set(excluded) real 88 seizures & synthetic seizures from each overlap model

# ## 1.  Compare train set patients' real seizures & train set patients' synthetic seizures from each overlap model
# ###  model - 0% overlap
#
#

# In[1]:


import os
import numpy as np
import pandas as pd
import random
# from scipy.io import savemat ,loadmat
import scipy.io as sio


# In[3]:


# synthetic seizures generated from 0 overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_trainpats_from70overlap'
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)


# In[4]:


x_seiz.shape


# In[10]:


# saving 10% of data from 0% overlap model
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_0overlap_4cnn\acc_x.npy",x_seiz[:633,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_0overlap_4cnn\acc_y.npy",y_seiz[:633,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_0overlap_4cnn\acc_z.npy",z_seiz[:633,:,:])


# In[11]:


# saving labels 10%
# labels = np.ones(80)
# # saving labels 30%
# labels = np.ones(238)
# # saving labels 50%
# labels = np.ones(396)
# # saving labels 80%
# labels = np.ones(633)
# # saving labels 100%
# labels = np.ones(791)


# In[12]:


# np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_0overlap_4cnn\labels.npy",labels)


# In[5]:


seiz_numpy_path = r'C:\Users\seyidova\Desktop\Epilepsygan\numpy_tts_seiz_numpy_0overlap'
hard_test= ['BN_011','BN_012','BN_031','BN_017','BN_103','BN_107','BN_160','BN_166','BN_167']


# In[6]:


def create_list_of_pat(path):
    '''
    This funtion will create the list of patients for which we have the numpy data available
    Arguments:
        path (string) :  where the numpy data is present
    Returns
        pat_list (list)  :  list of patients
    '''
    pat_list = list()  #empty list for storing patients numbers
    for file in sorted(os.listdir(path)):
        pat_name = file[0:6]   #exteacting patient number form the filename
        if pat_name not in hard_test:
            if pat_name in pat_list: #Check if the name already exist in list
                continue   #if name already exist then move to the next file
            else: #else appending the name to the pat_list
                pat_list.append(pat_name) #appending to the pat_list
    return pat_list


# In[40]:


pat_set = create_list_of_pat(seiz_numpy_path)


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
        #print(file)
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


# In[7]:


# real seizures of train patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_791 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
syn_791= np.concatenate([x_seiz[:791,:,:],y_seiz[:791,:,:],z_seiz[:791,:,:]],axis=2)


# In[8]:


np.unique(real_791[2] == real_791[2,:,:])
real_791.shape


# ### pca

# In[9]:


from visualizationMetrics import visualization
visualization(real_791, syn_791[:585,:,:], 'pca', 'Running-pca')


# ### t-sne

# In[10]:


visualization(real_791, syn_791[:585,:,:], 'tsne', 'Mond-data-tsne')


# ### DTW & fast DTW calculation 3D

# ### Calculate dtw similarity btw  100 real and synthetic trainpats seizures

# In[19]:


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

real_100=real_791[:585,:,:]
syn_100=syn_791[:585,:,:]

def cal_dtw(r_100,s_100):
    dtw_distance = []
    # Iterate through all the real and synthetic time series and calculate the DTW distance
    for i in range(len(r_100)):
        best_similarity=1000
        for j in range(len(s_100)):
            distance, path = fastdtw(r_100[i], s_100[j], dist=euclidean_3d)
            if distance < best_similarity:
                    best_similarity = distance
        dtw_distance.append(best_similarity)
    return np.mean(dtw_distance)

# print("Fast DTW distance between real and synthetic time series:", np.mean(dtw_distance))


# #### Calculate best match for each sample and add to the list and take mean of this list

# In[20]:


# dtw_mean_train= cal_dtw(real_100,syn_100)
# print(dtw_mean_train)


# ### Cos similarity calculation channelwise

# In[21]:


from math import sqrt

def square_rooted(x): #noqa

    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

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

def calc_feat_vec(syn,real_seiz):
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
    return rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat

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


# In[22]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100,syn_100)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# ### MSE calculation

# In[23]:


def compute_similarity(real,syn):
    mse=np.mean((real-syn)**2)
    return mse

def compute_mse(syn,real_seiz):
    best_mse=[]
    for synthetic_sample in syn:
        best_match = None
        best_similarity = 1000
        for real_sample in real_seiz:
            similarity = compute_similarity(synthetic_sample,
                                            real_sample)  # Implement this function to compute the similarity between the two samples
            if similarity < best_similarity:
                best_similarity = similarity
        best_mse.append(best_similarity)
    final_mean_mse=np.mean(best_mse)
    return final_mean_mse


# In[24]:


mse_mean_train = compute_mse(real_100,syn_100)
print(mse_mean_train)


# In[25]:


# compute_mse(real_100,real_100)


# ### model - 50% overlap

# In[26]:


# synthetic seizures generated from 0 overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_trainpats_from50overlap'
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)


np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_50overlap_4cnn\acc_x.npy",x_seiz[:791,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_50overlap_4cnn\acc_y.npy",y_seiz[:791,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_50overlap_4cnn\acc_z.npy",z_seiz[:791,:,:])

labels = np.ones(791)

np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_50overlap_4cnn\labels.npy",labels)

pat_set = create_list_of_pat(seiz_numpy_path)

# real seizures of train patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_791 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
syn_791= np.concatenate([x_seiz[:791,:,:],y_seiz[:791,:,:],z_seiz[:791,:,:]],axis=2)


# In[27]:


from visualizationMetrics import visualization
visualization(real_791, syn_791, 'pca', 'Running-pca')


# In[28]:


visualization(real_791, syn_791, 'tsne', 'Mond-data-tsne')


# In[29]:


real_100=real_791[:585,:,:]
syn_100=syn_791[:585,:,:]


# #### Computer dtw

# In[30]:


# dtw_mean_train= cal_dtw(real_100,syn_100)
# print(dtw_mean_train)


# #### Computer cosine similarity

# In[31]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100,syn_100)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# #### Computer mse

# In[32]:


mse_mean_train = compute_mse(real_100,syn_100)
print(mse_mean_train)


# ### model - 70 % overlap

# In[33]:


# synthetic seizures generated from 0 overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_trainpats_from70overlap'
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)


np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_70overlap_4cnn\acc_x.npy",x_seiz[:791,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_70overlap_4cnn\acc_y.npy",y_seiz[:791,:,:])
np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_70overlap_4cnn\acc_z.npy",z_seiz[:791,:,:])

labels = np.ones(791)

np.save(r"C:\Users\seyidova\Desktop\Epilepsygan\epl_gan_70overlap_4cnn\labels.npy",labels)

pat_set = create_list_of_pat(seiz_numpy_path)

# real seizures of train patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_791 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
syn_791= np.concatenate([x_seiz[:791,:,:],y_seiz[:791,:,:],z_seiz[:791,:,:]],axis=2)


# In[34]:


from visualizationMetrics import visualization
visualization(real_791, syn_791, 'pca', 'Running-pca')


# In[35]:


visualization(real_791, syn_791, 'tsne', 'Mond-data-tsne')


# In[36]:


real_100=real_791[:585,:,:]
syn_100=syn_791[:585,:,:]


# #### Computer dtw

# In[37]:


# dtw_mean_train= cal_dtw(real_100,syn_100)
# print(dtw_mean_train)


# #### Computer cosine similarity

# In[38]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_100,syn_100)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# #### Compute mse

# In[39]:


mse_mean_train = compute_mse(real_100,syn_100)
print(mse_mean_train)


# ## 2.Compare hard coded test set(excluded) real 88 seizures & synthetic seizures from each overlap model
# ### model - 0% overlap

# In[11]:


def create_list_of_pat_test(path):
    '''
    This funtion will create the list of patients for which we have the numpy data available
    Arguments:
        path (string) :  where the numpy data is present
    Returns
        pat_list (list)  :  list of patients
    '''
    pat_list = list()  #empty list for storing patients numbers
    for file in sorted(os.listdir(path)):
        pat_name = file[0:6]   #exteacting patient number form the filename
        if pat_name in hard_test:
            if pat_name in pat_list: #Check if the name already exist in list
                continue   #if name already exist then move to the next file
            else: #else appending the name to the pat_list
                pat_list.append(pat_name) #appending to the pat_list
    return pat_list


# In[40]:


pat_set = create_list_of_pat_test(seiz_numpy_path)




# In[12]:


# synthetic seizures generated from 0 % overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_testpats_from70overlap' #syn seizures of test pats generated with this model
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)

# real seizures test patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_88 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
# syn seizures test patients
syn_88= np.concatenate([x_seiz[:88,:,:],y_seiz[:88,:,:],z_seiz[:88,:,:]],axis=2)


# In[13]:


from visualizationMetrics import visualization
visualization(real_88, syn_88, 'pca', 'Running-pca')


# In[14]:


visualization(real_88, syn_88, 'tsne', 'Mond-data-tsne')


# #### Computer dtw

# In[44]:


# dtw_mean_train= cal_dtw(real_88,syn_88)
# print(dtw_mean_train)


# #### Compute cosine similarity

# In[45]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_88,syn_88)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# #### Compute mse

# In[46]:


mse_mean_train = compute_mse(real_88,syn_88)
print(mse_mean_train)


# ###  model -  50% overlap

# In[47]:


# synthetic seizures generated from 50 % overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_testpats_from50overlap' #syn seizures of test pats generated with this model
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)

# real seizures test patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_88 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
# syn seizures test patients
syn_88= np.concatenate([x_seiz[:88,:,:],y_seiz[:88,:,:],z_seiz[:88,:,:]],axis=2)


# In[48]:


from visualizationMetrics import visualization
visualization(real_88, syn_88, 'pca', 'Running-pca')


# In[49]:


visualization(real_88, syn_88, 'tsne', 'Mond-data-tsne')


# #### Compute dtw

# In[50]:


# dtw_mean_train= cal_dtw(real_88,syn_88)
# print(dtw_mean_train)


# #### Compute cosine similarity

# In[51]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_88,syn_88)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# #### Compute mse

# In[52]:


mse_mean_train = compute_mse(real_88,syn_88)
print(mse_mean_train)


# ### model - 70% overlap

# In[53]:


# synthetic seizures generated from 70 % overlap model from train patients' non-seizures--100% ==791 seizures
x_all,y_all,z_all = [],[],[]
import os
syn_seiz_path_x = r'C:\Users\seyidova\Desktop\Epilepsygan\syn_testpats_from70overlap' #syn seizures of test pats generated with this model
for fold in os.listdir(syn_seiz_path_x):
    for file in os.listdir(os.path.join(syn_seiz_path_x,fold)):
        data_x = sio.loadmat(os.path.join(syn_seiz_path_x,fold,file))
    #         data_y = sio.loadmat(os.path.join(syn_seiz_path_y,fold,file))
    #         data_z = sio.loadmat(os.path.join(syn_seiz_path_z,fold,file))
        x_all.append(np.expand_dims(data_x['GAN_seiz'][:500,0],1))
        y_all.append(np.expand_dims(data_x['GAN_seiz'][:500,1],1))
        z_all.append(np.expand_dims(data_x['GAN_seiz'][:500,2],1))

# x_total = np.concatenate(x_all,axis=0)
# y_total = np.concatenate(y_all,axis=0)
# z_total = np.concatenate(z_all,axis=0)
x_seiz=np.stack(x_all,axis=0) # (number_of_samples,500,1)
y_seiz=np.stack(y_all,axis=0)
z_seiz=np.stack(z_all,axis=0)

# real seizures test patients
f,_,_ = load_all_numpy_data(pat_set,seiz_numpy_path)
real_88 = np.concatenate([f['acc_x'],f['acc_y'],f['acc_z']],axis=2)
# syn seizures test patients
syn_88= np.concatenate([x_seiz[:88,:,:],y_seiz[:88,:,:],z_seiz[:88,:,:]],axis=2)


# In[54]:


from visualizationMetrics import visualization
visualization(real_88, syn_88, 'pca', 'Running-pca')


# In[55]:


visualization(real_88, syn_88, 'tsne', 'Mond-data-tsne')


# #### Compute dtw

# In[56]:


# dtw_mean_train= cal_dtw(real_88,syn_88)
# print(dtw_mean_train)


# #### Compute cosine similarity

# In[57]:


rx_feat,ry_feat, rz_feat,sx_feat, sy_feat, sz_feat = calc_feat_vec(real_88,syn_88)

x_similarity=calc_cs(rx_feat, sx_feat)
y_similarity=calc_cs(ry_feat, sy_feat)
z_similarity=calc_cs(rz_feat, sz_feat)
mean_similarity=np.mean([x_similarity,y_similarity,z_similarity])

print("Avg_cosine_Similarity for acc_X  :  ",x_similarity )
print("Avg_cosine_Similarity for acc_y  :  ",y_similarity )
print("Avg_cosine_Similarity for acc_z  :  ",z_similarity )
print("Avg_cosine_Similarity for acc  :  ",mean_similarity )


# #### Compute mse

# In[58]:


mse_mean_train = compute_mse(real_88,syn_88)
print(mse_mean_train)

