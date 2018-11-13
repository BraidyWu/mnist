# -*- coding: utf-8 -*-

import tensorflow as tf

num_input = 784
num_output = 10
num_hidden = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        'weight',
        shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weight = get_weight_variable(
            [num_input, num_hidden],
            regularizer
        )
        bias = tf.get_variable(
            'bias',
            [num_hidden],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + bias)
    
    with tf.variable_scope('layer2'):
        weight = get_weight_variable(
            [num_hidden, num_output],
            regularizer
        )
        bias = tf.get_variable(
            'bias',
            [num_output],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weight) + bias
    return layer2
