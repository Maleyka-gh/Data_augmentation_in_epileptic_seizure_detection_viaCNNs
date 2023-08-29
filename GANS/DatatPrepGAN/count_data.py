import os
import numpy as np
#
# # number of pairs(seiz&non-seiz) in train set
# print(len(os.listdir(("/data/Corrected_code_Gan/data/trainset/"))))
# # in test - we have the same number of (only) non-seiz samples
# path = "/data/Corrected_code_Gan/test_set/"
# min = 110000
# max = 0
#
# for pat in os.listdir(path):
#     l = len(os.listdir(os.path.join(path, pat)))
#     if l<min:
#         min=l
#     if l>max:
#         max=l
#     print(pat,l)
#
# print(min,max)
#
# path = r'/data/Data_prep_for_gan/seiz_numpy/BN_179_label.npy'
# print(len(np.load(path)))
#
#
#
# path=r'/data/Data_prep_for_gan/GAN_16_1024/numpy_GAN_SEIZ'
#
# for file in os.listdir(path):
#     data = np.load(os.path.join(path,file))
#     s = np.sum(data)
#     a = np.isnan(s)
#     if a:
#         print('Yes')
# path1='/data/Data_prep_for_gan/numpy_EPL_GAN_Non_seiz_numpy_512'
# path2 ='/data/Data_prep_for_gan/numpy_epl_seiz_numpy_0overlap_512'
# l ,l2= [],[]
# for file in sorted(os.listdir(path2)):
#     pat = file[0:6]
#     if pat not in l:
#         l.append(pat)
# print(len(l))
# for file in sorted(os.listdir(path1)):
#     pat = file[0:6]
#     if pat not in l2:
#         l2.append(pat)
# l_n =[]
# for i in l:
#     if i not in l2:
#         l_n.append(i)
# print(l_n)
count_pat=0
count_win=0
test_set = {'BN_011': 3, 'BN_012': 1, 'BN_031': 1, 'BN_103': 2, 'BN_107': 1, 'BN_160': 4, 'BN_166': 1, 'BN_167': 9, 'BN_017': 1}

'''path='/data/Data_prep_for_gan/numpy_epl_seiz_70overlap_512_k10'
for file in os.listdir(path):#listing the filenames as strings inside the path
    if file[0:6] not in test_set.keys():
        if 'label' in file:
            count_pat+=1
            label_file=np.load(os.path.join(path,file))#os.path.join(path,file) provides file path, np.load loads the file content
            win_pat=len(label_file)
            count_win+=win_pat

print(count_win)
print(count_pat)'''

# path='/data/optimized/Numpy_Learn_10s0Overlap_acc'

# for file in os.listdir(path):#listing the filenames as strings inside the path
#     # if file[0:6] not in test_set.keys():
#     if 'label' in file:
#         label_array= np.load(os.path.join(path,file))
#         if 0 in label_array:
#             count_pat+=1
#             label_file=np.load(os.path.join(path,file))#os.path.join(path,file) provides file path, np.load loads the file content
#             win_pat=len(label_file)
#             count_win+=win_pat
#
# print(count_win)
# print(count_pat)


hard_test= ['BN_011','BN_012','BN_031','BN_103','BN_107','BN_160','BN_166','BN_167','BN_017']


# In[40]:
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

seiz_numpy_path = r'/data/Data_prep_for_gan/numpy_epl_seiz_0overlap_512_k10'

pat_set = create_list_of_pat(seiz_numpy_path)



# In[40]:

path1 ='/data/Data_prep_for_gan/numpy_epl_seiz_0overlap_512_k10'
path2 = '/data/Data_prep_for_gan/numpy_epl_non_seiz_512_k0'
for file in os.listdir(path2):#listing the filenames as strings inside the path
    if file[0:6] in hard_test:
        if 'label' in file:

            # print(file)
        # if 0 in label_array:
        #     count_pat+=1
            label_file=np.load(os.path.join(path2,file))#os.path.join(path,file) provides file path, np.load loads the file content
            print(file[0:6], len(label_file))
        #     win_pat=len(label_file)
        #     count_win+=win_pat
