#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Importing the Used Libraries
import os
import numpy as np
import pandas as pd
import random
# from scipy.io import savemat ,loadmat
import scipy.io as sio

# 1. Compare train set patients' real seizures & train set patients' synthetic seizures from each overlap model
# 2. Compare hard coded test set(excluded) real 88 seizures & synthetic seizures from each overlap model

# In[15]:


real_seiz_path = r'C:\Users\seyidova\Desktop\Epilepsy_Gan_visual\trainset'
syn_seiz_path = r'C:\Users\seyidova\Desktop\Epilepsy_Gan_visual\test_set_transformed'

# In[9]:


real_seiz_path = r'C:\Users\seyidova\Desktop\Epilepsy_Gan_visual\trainset_contin_70'
syn_seiz_path = r'C:\Users\seyidova\Desktop\Epilepsy_Gan_visual\test_set_transformed_contin_70'

# ### for a single patient

# In[31]:


l = []
for filename in os.listdir(real_seiz_path):
    if 'pat_006' in filename:
        real_seiz = sio.loadmat(os.path.join(real_seiz_path, filename))
        real_seiz = real_seiz['seiz']
        real_seiz = np.transpose(real_seiz, (1, 0))
        l.append(real_seiz)

real_all = np.stack(l, axis=0)

# In[32]:


l_test = []
for filename in os.listdir(syn_seiz_path):
    if 'pat_006' in filename:
        syn_seiz = sio.loadmat(os.path.join(syn_seiz_path, filename))
        syn_seiz = syn_seiz['GAN_seiz']
        syn_seiz = np.transpose(syn_seiz, (1, 0))
        l_test.append(syn_seiz)

syn_all = np.stack(l, axis=0)

# In[35]:


syn_all.shape

# In[18]:


from visualizationMetrics import visualization

# In[7]:


visualization(real_all, syn_all, 'pca', 'Running-pca')

# In[12]:


visualization(real_all, syn_all, 'pca', 'Running-pca')

# In[22]:


visualization(real_all[:1257, :, :], syn_all, 'pca', 'Running-pca')

# In[8]:


visualization(real_all, syn_all, 'tsne', 'Mond-data-tsne')

# In[13]:


visualization(real_all, syn_all, 'tsne', 'Mond-data-tsne')

# In[24]:


visualization(real_all[:1257, :, :], syn_all, 'tsne', 'Mond-data-tsne')

# ### for all patients

# In[16]:


l = []
for filename in os.listdir(real_seiz_path):
    if 'pat_' in filename:
        real_seiz = sio.loadmat(os.path.join(real_seiz_path, filename))
        real_seiz = real_seiz['seiz']
        #         real_seiz=np.transpose(real_seiz,(1,0))
        l.append(real_seiz)

real_all = np.stack(l, axis=0)

# In[17]:


l_test = []
for folder in os.listdir(syn_seiz_path):
    #     print(folder)
    for filename in os.listdir(os.path.join(syn_seiz_path, folder)):
        #         print(filename)
        syn_seiz = sio.loadmat(os.path.join(syn_seiz_path, folder, filename))
        #         print(syn_seiz)
        syn_seiz = syn_seiz['GAN_seiz']
        #         syn_seiz=np.transpose(syn_seiz,(1,0)) (512,3)->(3,512)
        l_test.append(syn_seiz)
#         print(l_test)

syn_all = np.stack(l_test, axis=0)  # (n,3,512)

# In[59]:


# syn_all = np.transpose(syn_all,(0,2,1))


# In[20]:


syn_all.shape, real_all.shape

# In[41]:


from math import *


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


# print(cosine_similarity(s1,r1))


# ### Cosine similarity btw all real and synthetic samples for pat_006

# In[46]:


# s1 = np.squeeze(syn_all[9,:,:])  # syn_all[0,:,:] return shape (512,1) , with np.squeeze it returns (512,)
exclude_list = []
list_cos_avg = []
for j in range(syn_all.shape[0]):
    list_cos_sim = []
    s = np.squeeze(syn_all[j, :, 2])
    for i in range(real_all.shape[0]):
        r = np.squeeze(real_all[i, :, 2])
        cos_sim = cosine_similarity(s, r)
        list_cos_sim.append(cos_sim)
    avg = np.sum(list_cos_sim) / len(list_cos_sim)
    #     if avg>0.5:
    #         print(j)
    list_cos_avg.append(avg)
#     else:
#         exclude_list.append(j)
#         print('bad syn sample',j)
final = np.sum(list_cos_avg) / len(list_cos_avg)
final

# In[61]:


from visualizationMetrics import visualization

# In[62]:


visualization(real_all, syn_all, 'pca', 'Running-pca')

# In[63]:


visualization(real_all, syn_all, 'tsne', 'Mond-data-tsne')

# In[23]:


syn_seiz_path = r'C:\Users\seyidova\Desktop\Epilepsy_Gan_visual\GAN_seizure_pat_006_GAN_test_3'

# In[24]:


syn_seiz = sio.loadmat(syn_seiz_path)

# In[25]:


syn_seiz_006 = syn_seiz['GAN_seiz']

# In[26]:


import matplotlib.pyplot as plt

# In[27]:


plt.plot(syn_seiz_006)

# In[ ]:




