# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train
from mnist_config import *


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.num_input], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.num_output], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(mnist.train.moving_average_decay)
        variable_to_restore = variable_average.variable_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_save_path)
                if ckpt and ckpt.model_save_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(eval_interval_sec)

def main(argv=None):
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
    
