#!/usr/bin/env python
# coding: utf-8

# ## Compare the similarities of the synthietic data with real data based on some evalutation matrices

# ### Step 1: Load Real running and Jumping data

# In[1]:


from dataLoader import *
from torch.utils import data


# In[2]:


train_set = Mond_data_loader(data_path=r'C:\Users\seyidova\Desktop\tts_gan\numpy_tts_seiz_numpy_0overlap',is_normalize = False,data_mode = 'Train')


# In[3]:


train_set


# In[4]:


real_data_loader = data.DataLoader(train_set, batch_size=1, num_workers=1, shuffle=True)#pytorch built in library to prepare data


# In[5]:


len(real_data_loader)


# In[6]:


import matplotlib.pyplot as plt
real_seiz = []

for i, (real_sig, label) in enumerate(real_data_loader):
    real_sig = real_sig.cpu().detach().numpy()
    sig = real_sig.reshape(real_sig.shape[1], real_sig.shape[3])
    if label[0] == 1:
        real_seiz.append(sig)

real_seiz = np.array(real_seiz)
print(real_seiz.shape)


# In[7]:


fig, axs = plt.subplots(1, 10, figsize=(35,5))
fig.suptitle('Real_Seiz', fontsize=30)
for i in range(10):
    axs[i].plot(real_seiz[i][0][:])
    axs[i].plot(real_seiz[i][1][:])
    axs[i].plot(real_seiz[i][2][:])


# In[8]:


from Synthetic_data_loader_final import *
from torch.utils import data


# ### Define the number of sample to be generated synthetically with "sample_size" argument

# In[72]:


syn_data=Synthetic_Dataset(sample_size=391,mond_model_path=r'C:\Users\seyidova\Desktop\tts_gan\checkpoint_bestcos_sim')


#
# # Loading the Synthetic Data

# In[73]:


syn_data_loader = data.DataLoader(syn_data, batch_size=1, num_workers=1, shuffle=True)#pytorch built in library to prepare data


# In[74]:


import matplotlib.pyplot as plt
syn_seiz = []

for i, (syn_sig, label) in enumerate(syn_data_loader):
    syn_sig = syn_sig.cpu().detach().numpy()
    sig = syn_sig.reshape(syn_sig.shape[1], syn_sig.shape[3])
    if label[0] == 1:
        syn_seiz.append(sig)

syn_seiz = np.array(syn_seiz)
print(syn_seiz.shape)


# In[75]:


fig, axs = plt.subplots(1, 10, figsize=(35,5))
fig.suptitle('Syn_Seiz', fontsize=30)
for i in range(10):
    j= np.random.randint(0,50)
    axs[i].plot(syn_seiz[j][0][:])
    axs[i].plot(syn_seiz[j][1][:])
    axs[i].plot(syn_seiz[j][2][:])


# ### Step 3: PCA

# In[76]:


print(real_seiz.shape)
print(syn_seiz.shape)


# In[77]:


# real_seiz = np.transpose(real_seiz, (0, 2, 1))
syn_seiz = np.transpose(syn_seiz, (0, 2, 1))
print(real_seiz.shape)
print(syn_seiz.shape)


# In[78]:


# np.expand_dims(syn_seiz[:,:,0],axis=2).shape


# In[79]:


np.ones(len(syn_seiz)).shape


# In[80]:


run = 2
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos100%/run_{run}_acc_x.npy',np.expand_dims(syn_seiz[:,:,0],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos100%/run_{run}_acc_y.npy',np.expand_dims(syn_seiz[:,:,1],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos100%/run_{run}_acc_z.npy',np.expand_dims(syn_seiz[:,:,2],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos100%/run_{run}_label.npy',np.ones(len(syn_seiz)))


# ### save the same data with different percentages

# In[81]:


def prep_syn(path):
    acc_x = np.load(os.path.join(path,'run_1_acc_x.npy'))
    acc_y = np.load(os.path.join(path,'run_1_acc_y.npy'))
    acc_z = np.load(os.path.join(path,'run_1_acc_z.npy'))
    acc_x2 = np.load(os.path.join(path,'run_2_acc_x.npy'))
    acc_y2 =np.load(os.path.join(path,'run_2_acc_y.npy'))
    acc_z2 = np.load(os.path.join(path,'run_2_acc_z.npy'))
    all_syn = np.concatenate([acc_x,acc_y,acc_z],axis=2)
    all_syn2= np.concatenate([acc_x2,acc_y2,acc_z2],axis=2)
    syn=np.concatenate([all_syn,all_syn2],axis=0)
    print(syn.shape)
    return syn
syn=prep_syn(r'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos100%')


# In[82]:


run = 1
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos10%/run_{run}_acc_x.npy',np.expand_dims(syn[:80,:,0],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos10%/run_{run}_acc_y.npy',np.expand_dims(syn[:80,:,1],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos10%/run_{run}_acc_z.npy',np.expand_dims(syn[:80,:,2],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos10%/run_{run}_label.npy',np.ones(80))


# In[83]:


run = 1
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos30%/run_{run}_acc_x.npy',np.expand_dims(syn[:238,:,0],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos30%/run_{run}_acc_y.npy',np.expand_dims(syn[:238,:,1],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos30%/run_{run}_acc_z.npy',np.expand_dims(syn[:238,:,2],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos30%/run_{run}_label.npy',np.ones(238))


# In[84]:


run = 1
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos50%/run_{run}_acc_x.npy',np.expand_dims(syn[:396,:,0],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos50%/run_{run}_acc_y.npy',np.expand_dims(syn[:396,:,1],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos50%/run_{run}_acc_z.npy',np.expand_dims(syn[:396,:,2],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos50%/run_{run}_label.npy',np.ones(396))


# In[85]:


run = 1
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos80%/run_{run}_acc_x.npy',np.expand_dims(syn[:633,:,0],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos80%/run_{run}_acc_y.npy',np.expand_dims(syn[:633,:,1],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos80%/run_{run}_acc_z.npy',np.expand_dims(syn[:633,:,2],axis=2))
np.save(f'C:/Users/seyidova/Desktop/tts_gan/tts_bestcos80%/run_{run}_label.npy',np.ones(633))


# In[27]:


from visualizationMetrics import visualization


# In[28]:


visualization(real_seiz[:400,:,:], syn_seiz[:,:,:], 'pca', 'Running-pca')


# In[29]:


visualization(real_seiz[:400,:,:], syn_seiz[:,:,:], 'tsne', 'Mond-data-tsne')


# In[30]:


# import torch
# model = torch.load(r'C:\Users\seyidova\Desktop\tts_gan\checkpoint')


# In[ ]:




