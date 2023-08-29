#!/bin/bash

#list='pat_006'
##list='pat_006 pat_011 pat_012 pat_016 pat_017 pat_018 pat_027 pat_041 pat_046 pat_047 pat_051 pat_057 pat_067 pat_070 pat_071 pat_082 pat_083 pat_084 pat_086 pat_100 pat_103 pat_106 pat_107 pat_113 pat_123 pat_124 pat_138 pat_141 pat_146 pat_149 pat_159 pat_160 pat_166 pat_167 pat_169 pat_170 pat_179 pat_180 pat_186'
#list='pat_006'
#for i in $list; do
##  echo $i
#  CUDA_VISIBLE_DEVICES="0,1" python main.py --save_path gan_results/leave_out_$i/ --e2e_dataset data/TFrecords/gan_leave_out_$i.tfrecords
#done
#

##list below excludes hard coded test set patients
#list='pat_006 pat_016 pat_041 pat_047 pat_051 pat_057 pat_067 pat_070 pat_071 pat_082 pat_083 pat_086 pat_100 pat_106 pat_113 pat_123 pat_124 pat_138 pat_141 pat_146 pat_149 pat_159 pat_169 pat_170 pat_179 pat_180 pat_186'
# list below includes only hard coded test patients

list='pat_166 pat_167 pat_160 pat_107 pat_103 pat_017 pat_012 pat_011'
for i in $list; do
  CUDA_VISIBLE_DEVICES="0,1" python main.py --init_noise_std 0. --save_path gan_results_70overlap/leave_out_'pat_006'/ --weights GAN --test_set test_set_88ns_testpats/$i/ --save_transformed_path test_set_transformed/$i/
#  CUDA_VISIBLE_DEVICES="0,1" python main.py --init_noise_std 0. --save_path gan_results/leave_out_$i/ --weights GAN --test_wav test_set/$i/ --save_transformed_path test_set_transformed/$i/test_set_non_seiz_from_trainpats
done

#'BN_011': 3, 'BN_012': 1, 'BN_031': 1, 'BN_103': 2, 'BN_107': 1, 'BN_160': 4, 'BN_166': 1, 'BN_167': 9, 'BN_017': 1}
