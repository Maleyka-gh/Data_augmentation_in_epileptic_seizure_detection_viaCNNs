#!/bin/bash

#list='pat_006'
#list='pat_006 pat_011 pat_012 pat_016 pat_017 pat_018 pat_027 pat_029 pat_036 pat_037 pat_041 pat_047 pat_051 pat_057 pat_067 pat_070 pat_071 pat_072 pat_083 pat_084 pat_086 pat_100 pat_103 pat_106 pat_107 pat_113 pat_124 pat_138 pat_146 pat_149 pat_159 pat_160 pat_166 pat_167 pat_170 pat_180 pat_186'
list='pat_006'
for i in $list; do
    echo $i
    python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg --patient $i --save_path data/TFrecords/
done


