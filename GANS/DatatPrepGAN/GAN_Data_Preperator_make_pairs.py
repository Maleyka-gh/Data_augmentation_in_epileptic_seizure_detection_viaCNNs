#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Importing the Used Libraries
import os
import numpy as np
import pandas as pd
import random
from scipy.io import savemat

hard_test= ['BN_011','BN_012','BN_103','BN_107','BN_160','BN_166','BN_167']

#numpy paths
seiz_numpy_path = r'/data/Data_prep_for_gan/numpy_epl_seiz_70overlap_512_k10'
non_seiz_numpy_path = r'/data/Data_prep_for_gan/numpy_epl_non_seiz_512_k0'

#folders to which u want to save the results
save_path_train = r'/data/Corrected_code_Gan/data/trainset/'
save_path_test = r'/data/Corrected_code_Gan/test_set/'



# In[39]:


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


# In[41]:


def load_all_numpy_data(pat_set, path):
    """
    This method loads all computed numpy arrays. It creates two np arrays (features and labels) and fills them with the
    concatenated numpy arrays.
    :param pat_set: List of required patients
    :return: 3 np arrays (feats, labels, timestamps)
    """

    features = dict()   #   { 'feat' : 'pat_num' : numpy_data}
    labels = dict() # {'pat' : label_array}
    times = dict() # {'pat' : time_array}
    for file in sorted(os.listdir(path)):  #iterate through numpy_data

        if file[0:6] in pat_set:
            if "times" in file: #if the file is time file
                curr_times = np.array(pd.read_pickle(os.path.join(path, file))) #load the dataframe and convert to numpy

                times[file[0:6]] = curr_times # save in time dictionary at 'pat' as key and data as its value

                continue
            curr_np = np.load(os.path.join(path, file)) # load the numpy data
            if "label" in file: #if the file was label file
                labels[file[0:6]] = curr_np  # save the data in label dictionary with 'pat' as key and numpy_label as its value
            else: #else we have a feature file
                name = file[7:-10]     #extract the feature name form filename
                if name not in features: #if it does not already exist in features dictionary
                    features[name] = dict() #create the item 'feat name' in features dictionary  of type dict
                features[name][file[0:6]] = curr_np #save as feat data w.r.t pat

    return features, labels, times


# In[42]:


f,l,t = load_all_numpy_data(hard_test,seiz_numpy_path) #calling the function
f2,l2,t2 = load_all_numpy_data(hard_test,non_seiz_numpy_path)


# In[44]:


feat_list = ['acc_x','acc_y','acc_z']
pat_list = pat_set
print(len(pat_list))
#
# size = 512

# # pat_set=['BN_186']
# pat_less_nonseiz=["BN_084","BN_046","BN_018"]

for patient in hard_test:
    if patient!='BN_027' and patient!='BN_046':
        first_iter=True
        for feature in feat_list:
            if first_iter:
                pat_feat_seiz = f[feature][patient]  # (total_number windows , 512 , 1) -> acc_x
                pat_feat_non_seiz = f2[feature][patient]
                first_iter=False
            else:
                pat_feat_seiz = np.concatenate([pat_feat_seiz,f[feature][patient]],axis=2)  #(num,512,2) -> (num,512,3)
                pat_feat_non_seiz = np.concatenate([pat_feat_non_seiz,f2[feature][patient]],axis=2)

        # final_dict = dict()

        # idx_list=[]
        # print(patient)
        # for num_samp in range(5):
        #     for sample in range(len(pat_feat_seiz)):
        #         idx = np.random.randint(0, len(pat_feat_non_seiz) - 1)
        #         key = str(num_samp) + 's' + str(sample)
        #         final_dict[key] = dict()
        #         idx_list.append(idx)
        #
        #         # print(pat_feat_non_seiz[sample].shape)
        #         # final_dict[key]['non_seiz'] = pat_feat_non_seiz[idx]
        #         # savemat(os.path.join(save_path_test,'pat_'+ patient[3:6],'pat_' + patient[3:6] + '_GAN_test_' + str(key)) + '.mat', final_dict[key])
        #         # idx = np.random.randint(0, len(pat_feat_non_seiz) - 1)
        #         final_dict[key]['non_seiz'] = pat_feat_non_seiz[idx]
        #         final_dict[key]['seiz'] = pat_feat_seiz[sample]
        #         savemat(os.path.join(save_path_train,'pat_' + patient[3:6] + '_GAN_' + str(key)) + '.mat', final_dict[key])
        final_dict= dict()
        if patient == 'BN_160':
            os.mkdir(save_path_test + '/' + 'pat_' + patient[3:6])

            for i in range(12):#ex
                # idx = random.choice(list(set(range(len(pat_feat_non_seiz))) - set(idx_list)))# exclude idx/non seiz index that is used in train set
                key = str(i)
                final_dict[key] = dict()
                # idx_list.append(i)
                final_dict[key]['non_seiz'] = pat_feat_non_seiz[i]
                savemat(os.path.join(save_path_test,'pat_'+ patient[3:6],'pat_' + patient[3:6] + '_GAN_test_' + str(key)) + '.mat', final_dict[key])


