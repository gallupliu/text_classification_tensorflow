# -*- coding: utf-8 -*-
# @Time    : 17-11-18 下午11:37
# @Author  : gallup
# @Email   : gallup-liu@hotmail.com
# @File    : model_utils.py
# @Software: PyCharm

import  tensorflow as tf


def text_cnn(inputs,filter_sizes,num_filters,embedding_size,sequence_length,dropout_keep_prob):
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), dtype=tf.float32, name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), dtype=tf.float32, name="b")
            conv = tf.nn.conv2d(
                inputs,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    return h_drop

def rnn_cell(cell_name,num_layers, num_hidden,  dropout):
    #
    # with tf.name_scope(cell_name+scope), tf.variable_scope(cell_name+scope):
    if cell_name == "gru":
        cells = [tf.contrib.rnn.GRUCell(num_hidden) for _ in range(num_layers)]
    elif cell_name == "lstm":
        cells = [tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True) for _ in
                 range(num_layers)]
    elif cell_name == "rnn":
        cell = [tf.contrib.rnn.RNNCell(num_hidden) for _ in range(num_layers)]
    # cells = tf.contrib.rnn.DropoutWrapper(cells)
    cells = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(cells), output_keep_prob=dropout)
    return cells

def single_rnn(inputs,cell_name,num_hidden,num_layers,num_steps,dropout,scope):

    inputs = [tf.squeeze(input,axis=[1]) for input in tf.split(inputs,num_steps,1)]

    cells = rnn_cell(cell_name,num_hidden,num_layers,dropout)

    with tf.name_scope("RNN_"+cell_name+scope),tf.variable_scope("RNN_"+cell_name+scope):
        outputs ,_=tf.nn.dynamic_rnn(cells,inputs,time_major=False,dtype=tf.float32)
    return outputs

def bi_rnn(inputs,cell_name,num_hidden,num_layers,num_steps,dropout,scope):

    inputs = [tf.squeeze(input, axis=[1]) for input in tf.split(inputs, num_steps, 1)]

    with tf.name_scope("fw" + cell_name + scope), tf.variable_scope("fw" + cell_name + scope):
        lstm_fw_cell_m = rnn_cell(cell_name,num_layers, num_hidden,  dropout)

    with tf.name_scope("bw" + cell_name + scope), tf.variable_scope("bw" + cell_name + scope):
        lstm_bw_cell_m = rnn_cell(cell_name,num_layers, num_hidden,  dropout)

    with tf.name_scope("bi" + cell_name + scope), tf.variable_scope("b" + cell_name + scope):
        outputs,_,_=tf.nn.static_bidirectional_rnn(lstm_fw_cell_m,lstm_bw_cell_m,inputs,dtype=tf.float32)

    return outputs




