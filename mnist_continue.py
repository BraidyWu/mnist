# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train
from mnist_config import *


def continue_train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.num_input], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.num_output], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    saver = tf.train.Saver()
    
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(1, continue_step+1):
                xs, ys = mnist.train.next_batch(batch_size)
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
        else:
            print('No checkpoint file found')
            return

def main(argv=None):
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    continue_train(mnist)

if __name__ == '__main__':
    tf.app.run()
