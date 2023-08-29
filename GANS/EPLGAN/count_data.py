import os
import numpy as np
list1=[]
path1 = '/data/Corrected_code_Gan/test_set_transformed'
path= '/data/Corrected_code_Gan/data/trainset'
print(os.listdir(path1))
print(os.listdir(path))

# for folder in os.listdir(path):
#     if folder not in os.listdir(path1):
#         print(len(os.listdir(os.path.join(path,folder))))
#         print(folder)

#     list1.append(len(os.listdir(os.path.join(path,folder))))
# print(np.sum(list1))


# print((len(os.listdir(os.path.join(path)))))
#list='pat_006 pat_011 pat_012 pat_016 pat_017 pat_018 pat_027 pat_029 pat_036 pat_037 pat_041 pat_046 pat_047 pat_051 pat_057 pat_067 pat_070 pat_071 pat_072 pat_082 pat_083 pat_084 pat_086 pat_100 pat_103 pat_106 pat_107 pat_113 pat_123 pat_124 pat_138 pat_146 pat_149 pat_159 pat_160 pat_166 pat_167 pat_169 pat_170 pat_180 pat_186 pat_199'
#pat = 'pat_011'