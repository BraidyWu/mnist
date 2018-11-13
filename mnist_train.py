# -*- coding:utf-8 -*-

import os
import tensorflow as tf

batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 1e-4
training_step = 10000
moving_average_decay = 0.99

model_save_path = '/persisted_storage/Projects/mnist'
model_name = 'model.ckpt'

def train(mnist):
    