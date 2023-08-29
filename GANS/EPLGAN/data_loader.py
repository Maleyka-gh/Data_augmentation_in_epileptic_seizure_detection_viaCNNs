from __future__ import print_function
# import tensorflow as tf
from ops import *
import numpy as np


def read_and_decode(filename_queue, canvas_size):
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.compat.v1.io.parse_single_example(
        serialized_example,
        features={
            'seiz_raw': tf.compat.v1.io.FixedLenFeature([], tf.string),
            'nonseiz_raw': tf.compat.v1.io.FixedLenFeature([], tf.string),
        })
    seiz = tf.compat.v1.io.decode_raw(features['seiz_raw'], tf.float32)
    print("Size", tf.size(seiz))
    # print('VALUE',seiz)
    # print("SIZE", tf.size(seiz))
    # print('Here')
    seiz = tf.reshape(seiz, (512, 3))
    seiz.set_shape([canvas_size, None])
    print('here>')
    # seiz= tf.reshape(seiz,shape=(512,3))
    # print('SAHPE',seiz.shape)
    seiz = tf.cast(seiz, tf.float32)

    seiz = (2. / 65535.) * tf.cast((seiz), tf.float32)
    nonseiz = tf.compat.v1.io.decode_raw(features['nonseiz_raw'], tf.float32)
    nonseiz = tf.reshape(nonseiz, (512, 3))
    nonseiz.set_shape([canvas_size, None])
    # nonseiz = tf.reshape(nonseiz,shape=(512,3))
    nonseiz = tf.cast(nonseiz, tf.float32)
    nonseiz = (2. / 65535.) * tf.cast((nonseiz), tf.float32)

    return seiz, nonseiz
