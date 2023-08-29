#!/usr/bin/env python
# coding: utf-8

# ## GAN:for single pat , we concatenate all 10 sec seizure windows (each has 500 values) arrays , then devide the big array into 512 pieces(to get 2^9)

# In[20]:


# Importing the Used Libraries
import os
import numpy as np
import pandas as pd
import random
from scipy.io import savemat


numpy_data_path = r'/data/WORK/numpy_learn_10s_NoOverlap'

# to store the data_for GAN
save_path = r'/data/Corrected_code_Gan/data/trainset'


# In[21]:


# Creating the list of patients from the numpy_folder

def create_list_of_pat(path):
    '''
    This funtion will create the list of patients for which we have the numpy data available
    Arguments:
        path (string) :  where the numpy data is present
    Returns
        pat_list (list)  :  list of patients
    '''
    pat_list = list()  # empty list for storing patients numbers
    for file in sorted(os.listdir(path)):
        pat_name = file[0:6]  # extracting patient number form the filename
        if pat_name in pat_list:  # Check if the name already exist in list
            continue  # if name already exist then move to the next file
        else:  # else appending the name to the pat_list
            pat_list.append(pat_name)  # appending to the pat_list
    return pat_list


# In[22]:


pat_set = create_list_of_pat(numpy_data_path)


# In[23]:


def load_all_numpy_data(pat_set, path):
    """
    This method loads all computed numpy arrays. It creates two np arrays (features and labels) and fills them with the
    concatenated numpy arrays.
    :param pat_set: List of required patients
    :return: 3 np arrays (feats, labels, timestamps)
    """

    features = dict()  # { 'feat' : 'pat_num' : numpy_data}
    labels = dict()  # {'pat' : label_array}
    times = dict()  # {'pat' : time_array}
    for file in sorted(os.listdir(path)):  # iterate through numpy_data

        if file[0:6] in pat_set:
            if "times" in file:  # if the file is time file
                curr_times = np.array(
                    pd.read_pickle(os.path.join(path, file)))  # load the dataframe and convert to numpy

                times[file[0:6]] = curr_times  # save in time dictionary at 'pat' as key and data as its value

                continue
            curr_np = np.load(os.path.join(path, file))  # load the numpy data
            if "label" in file:  # if the file was label file
                labels[file[
                       0:6]] = curr_np  # save the data in label dictionary with 'pat' as key and numpy_label as its value
            else:  # else we have a feature file
                name = file[7:-10]  # extract the feature name form filename
                if name not in features:  # if it does not already exist in features dictionary
                    features[name] = dict()  # create the item 'feat name' in features dictionary  of type dict
                features[name][file[0:6]] = curr_np  # save as feat data w.r.t pat

    return features, labels, times


# In[24]:


f, l, t = load_all_numpy_data(pat_set, numpy_data_path)  # calling the function


# In[25]:


def extract_seiz_indexes(label_dictionary):
    '''
    Extract all the seizure indexes from the labels dictionary patient by patient

    Argument:
    label_dictionary (dict) : the dictionary containing patients with their label arrays
    Returns:
    seiz_dict (dict) : the dictionary of patients with their seizure indexes list
    '''

    seiz_dict = dict()  # {'pat' : seiz_index_list }
    for patient, label in label_dictionary.items():  # iterate over the labels dictionary -- pateint by patient
        list_of_seiz_index = []  # empty list
        for i in range(len(label)):  # loop iterate over total length of label array
            if label[i] == 1:  # if the particular index value is 1 which mean seizure
                list_of_seiz_index.append(i)  # store the index value in list
        seiz_dict[
            patient] = list_of_seiz_index  # store the list inside the dictionary with key as 'pat' and value as the list of seiz_index for that patient

    return seiz_dict


# In[26]:


seiz_dict = extract_seiz_indexes(l)  # calling the above function
# seiz_dict


# In[29]:


# implementaion for using only single feature as training set for GAN

save_path2 = r'/data/Corrected_code_Gan/test_set'

for feat, dictionary in f.items():  # iterate feature by feature

    for patient, data in dictionary.items():  # iterate patient by patient
        final_dict = dict()  # empty dictionary  {'sample' :  { 'seiz':numpy_data , 'non_seiz': numpy_data}  }
        os.mkdir(os.path.join(save_path2, 'pat_' + patient[3:6]))
        sample = 0  # sample number

        seiz_win_list = []
        non_seiz_win_list = []

        for indexs in seiz_dict[patient]:  # for particular patient iterate over the seizure indexs list
            seiz_array = data[indexs]  # extracting seizure data from numpy array
            r = indexs  # variable for randomly picking the non_seiz index
            # keep generating random index within specified range until this index is not a seizure index(not an index the same as in seiz_dict[patient])
            while r in seiz_dict[patient]:  # while we have a non_seiz index in variable r generate random number
                r = random.randint(0,
                                   len(data) - 1)  # generate random number between 0 and total number of windows inside the data of particular patient

            non_seiz = data[r]  # extracting the non_seizure numpy data
            seiz_win_list.append(seiz_array)
            non_seiz_win_list.append(non_seiz)
        if len(seiz_win_list) == 0:
            continue
        else:

            concat_seiz_win = np.concatenate(seiz_win_list)
            concat_non_seiz_win = np.concatenate(non_seiz_win_list)

            sample = 0
            for i in range(0, len(concat_seiz_win), 511):
                final_dict[sample] = dict()
                seiz_window_512 = concat_seiz_win[i:512 + i, :]
                non_seiz_window_512 = concat_non_seiz_win[i:512 + i, :]

                if len(seiz_window_512) < 512:
                    continue
                else:
                    seiz_window_512 = np.reshape(seiz_window_512, (512))
                    non_seiz_window_512 = np.reshape(non_seiz_window_512, (512))

                    final_dict[sample]['non_seiz'] = non_seiz_window_512
                    savemat(os.path.join(save_path2, 'pat_' + patient[3:6],
                                         'pat_' + patient[3:6] + '_GAN_test_' + str(sample)) + '.mat',
                            final_dict[sample])
                    final_dict[sample]['seiz'] = seiz_window_512
                    savemat(os.path.join(save_path, 'pat_' + patient[3:6] + '_GAN_' + str(sample)) + '.mat',
                            final_dict[sample])
                    sample = sample + 1  # updating  sample This sample denotes the seizure window number within a patient
    break

#         final_dict[patient]=dict()
#         final_dict[patient]['seiz_array']=seiz_array


#             final_dict[sample] = dict() # creating 'sample' item
#             #creating seiz and non_seiz dictionary inside final_dict

#             final_dict[sample]['non_seiz'] = non_seiz
#             savemat(os.path.join(save_path2,feat,'pat_'+ patient[3:6],'pat_' + patient[3:6] + '_GAN_test_' + str(sample)), final_dict[sample])
#             final_dict[sample]['seiz'] = seiz_array
#             savemat(os.path.join(save_path,feat,'pat_' + patient[3:6] + '_GAN_' + str(sample)), final_dict[sample])
#             sample=sample+1 #updating  sample This sample denotes the seizure window number within a patient


# !/usr/bin/env python
# coding: utf-8

# ## GAN:for single pat , we concatenate all 10 sec seizure windows (each has 500 values) arrays , then devide the big array into 512 pieces(to get 2^9)

# In[20]:


# Importing the Used Libraries
import os
import numpy as np
import pandas as pd
import random
from scipy.io import savemat


numpy_data_path = r'/data/WORK/numpy_learn_10s_NoOverlap'

# to store the data_for GAN
save_path = r'/data/Corrected_code_Gan/data/trainset'


# In[21]:


# Creating the list of patients from the numpy_folder

def create_list_of_pat(path):
    '''
    This funtion will create the list of patients for which we have the numpy data available
    Arguments:
        path (string) :  where the numpy data is present
    Returns
        pat_list (list)  :  list of patients
    '''
    pat_list = list()  # empty list for storing patients numbers
    for file in sorted(os.listdir(path)):
        pat_name = file[0:6]  # extracting patient number form the filename
        if pat_name in pat_list:  # Check if the name already exist in list
            continue  # if name already exist then move to the next file
        else:  # else appending the name to the pat_list
            pat_list.append(pat_name)  # appending to the pat_list
    return pat_list


# In[22]:


pat_set = create_list_of_pat(numpy_data_path)


# In[23]:


def load_all_numpy_data(pat_set, path):
    """
    This method loads all computed numpy arrays. It creates two np arrays (features and labels) and fills them with the
    concatenated numpy arrays.
    :param pat_set: List of required patients
    :return: 3 np arrays (feats, labels, timestamps)
    """

    features = dict()  # { 'feat' : 'pat_num' : numpy_data}
    labels = dict()  # {'pat' : label_array}
    times = dict()  # {'pat' : time_array}
    for file in sorted(os.listdir(path)):  # iterate through numpy_data

        if file[0:6] in pat_set:
            if "times" in file:  # if the file is time file
                curr_times = np.array(
                    pd.read_pickle(os.path.join(path, file)))  # load the dataframe and convert to numpy

                times[file[0:6]] = curr_times  # save in time dictionary at 'pat' as key and data as its value

                continue
            curr_np = np.load(os.path.join(path, file))  # load the numpy data
            if "label" in file:  # if the file was label file
                labels[file[
                       0:6]] = curr_np  # save the data in label dictionary with 'pat' as key and numpy_label as its value
            else:  # else we have a feature file
                name = file[7:-10]  # extract the feature name form filename
                if name not in features:  # if it does not already exist in features dictionary
                    features[name] = dict()  # create the item 'feat name' in features dictionary  of type dict
                features[name][file[0:6]] = curr_np  # save as feat data w.r.t pat

    return features, labels, times


# In[24]:


f, l, t = load_all_numpy_data(pat_set, numpy_data_path)  # calling the function


# In[25]:


def extract_seiz_indexes(label_dictionary):
    '''
    Extract all the seizure indexes from the labels dictionary patient by patient

    Argument:
    label_dictionary (dict) : the dictionary containing patients with their label arrays
    Returns:
    seiz_dict (dict) : the dictionary of patients with their seizure indexes list
    '''

    seiz_dict = dict()  # {'pat' : seiz_index_list }
    for patient, label in label_dictionary.items():  # iterate over the labels dictionary -- pateint by patient
        list_of_seiz_index = []  # empty list
        for i in range(len(label)):  # loop iterate over total length of label array
            if label[i] == 1:  # if the particular index value is 1 which mean seizure
                list_of_seiz_index.append(i)  # store the index value in list
        seiz_dict[
            patient] = list_of_seiz_index  # store the list inside the dictionary with key as 'pat' and value as the list of seiz_index for that patient

    return seiz_dict


# In[26]:


seiz_dict = extract_seiz_indexes(l)  # calling the above function
# seiz_dict


# In[29]:


# implementaion for using only single feature as training set for GAN

save_path2 = r'/data/Corrected_code_Gan/test_set'

for feat, dictionary in f.items():  # iterate feature by feature

    for patient, data in dictionary.items():  # iterate patient by patient
        final_dict = dict()  # empty dictionary  {'sample' :  { 'seiz':numpy_data , 'non_seiz': numpy_data}  }
        os.mkdir(os.path.join(save_path2, 'pat_' + patient[3:6]))
        sample = 0  # sample number

        seiz_win_list = []
        non_seiz_win_list = []

        for indexs in seiz_dict[patient]:  # for particular patient iterate over the seizure indexs list
            seiz_array = data[indexs]  # extracting seizure data from numpy array
            r = indexs  # variable for randomly picking the non_seiz index
            # keep generating random index within specified range until this index is not a seizure index(not an index the same as in seiz_dict[patient])
            while r in seiz_dict[patient]:  # while we have a non_seiz index in variable r generate random number
                r = random.randint(0,
                                   len(data) - 1)  # generate random number between 0 and total number of windows inside the data of particular patient

            non_seiz = data[r]  # extracting the non_seizure numpy data
            seiz_win_list.append(seiz_array)
            non_seiz_win_list.append(non_seiz)
        if len(seiz_win_list) == 0:
            continue
        else:

            concat_seiz_win = np.concatenate(seiz_win_list)
            concat_non_seiz_win = np.concatenate(non_seiz_win_list)

            sample = 0
            for i in range(0, len(concat_seiz_win), 511):
                final_dict[sample] = dict()
                seiz_window_512 = concat_seiz_win[i:512 + i, :]
                non_seiz_window_512 = concat_non_seiz_win[i:512 + i, :]

                if len(seiz_window_512) < 512:
                    continue
                else:
                    seiz_window_512 = np.reshape(seiz_window_512, (512))
                    non_seiz_window_512 = np.reshape(non_seiz_window_512, (512))

                    final_dict[sample]['non_seiz'] = non_seiz_window_512
                    savemat(os.path.join(save_path2, 'pat_' + patient[3:6],
                                         'pat_' + patient[3:6] + '_GAN_test_' + str(sample)) + '.mat',
                            final_dict[sample])
                    final_dict[sample]['seiz'] = seiz_window_512
                    savemat(os.path.join(save_path, 'pat_' + patient[3:6] + '_GAN_' + str(sample)) + '.mat',
                            final_dict[sample])
                    sample = sample + 1  # updating  sample This sample denotes the seizure window number within a patient
    break

#         final_dict[patient]=dict()
#         final_dict[patient]['seiz_array']=seiz_array


#             final_dict[sample] = dict() # creating 'sample' item
#             #creating seiz and non_seiz dictionary inside final_dict

#             final_dict[sample]['non_seiz'] = non_seiz
#             savemat(os.path.join(save_path2,feat,'pat_'+ patient[3:6],'pat_' + patient[3:6] + '_GAN_test_' + str(sample)), final_dict[sample])
#             final_dict[sample]['seiz'] = seiz_array
#             savemat(os.path.join(save_path,feat,'pat_' + patient[3:6] + '_GAN_' + str(sample)), final_dict[sample])
#             sample=sample+1 #updating  sample This sample denotes the seizure window number within a patient




