import os

# path= '/data/Final_Models/23-11-2022_21-17-13K_5_fold/test_investigation_4/Seiz_As_Seiz'
#
# count=0
# for file in os.listdir(path):
#     if 'FAS' in file and 'acc' in file:
#         print(file)
#         count+=1
# print(count)
# pat_name=[]
# pat_name_acc=[]
# path = '/data/optimized/Numpy_Learn_10s0Overlap'
# for filename in os.listdir(path):
#     if filename[:6] not in pat_name and 'readme' not in filename:
#         pat_name.append(filename[:6])
# print(sorted(pat_name),len(pat_name))
# path = '/data/optimized/Numpy_Learn_10s0Overlap_acc'
# for filename in os.listdir(path):
#     if filename[:6] not in pat_name_acc:
#         pat_name_acc.append(filename[:6])
# print(sorted(pat_name_acc),len(pat_name_acc))
#
#
# for pat in pat_name:
#     if pat not in pat_name_acc:
#         print(pat)

import numpy as np

path = np.load('/data/Final_Models/29-11-2022_17-29-28K_3_fold/CNN_meisel_Multi_FINAL/2/true_label.npy')
print(sum(path))
# print(path)