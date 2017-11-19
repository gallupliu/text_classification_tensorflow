#coding=utf-8


import tensorflow as tf
import numpy as np
import model.model_utils

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self,w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if w2v_model is None:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="word_embeddings")
            else:
                self.W = tf.get_variable("word_embeddings",
                    initializer=w2v_model)

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        self.h_drop = model.model_utils.text_cnn(self.embedded_chars_expanded,filter_sizes,num_filters,embedding_size,sequence_length,self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
           W = tf.get_variable(
                        "W",
                        shape=[num_filters * len(filter_sizes), num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
           b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
           l2_loss += tf.nn.l2_loss(b)         
           self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
           self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


  