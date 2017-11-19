# -*- coding: utf-8 -*-
# @Time    : 17-11-19 下午7:16
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : read_write_tfrecord.py
# @Software: PyCharm


import tensorflow as tf


def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(",")
    label = float(data[9])
    features = [float(i) for i in data[:9]]

    example = tf.train.Example(features=tf.train.Features(feature={
        "label":
        tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
        "features":
        tf.train.Feature(float_list=tf.train.FloatList(value=features)),
    }))
    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))
