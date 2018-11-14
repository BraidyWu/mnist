# Configuration

# mnist_train.py
num_input = 784
num_output = 10
num_hidden = 500

# mnist_inference.py
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 1e-4
training_step = 30000
continue_step = 30000
moving_average_decay = 0.99

# mnist_train.py
model_save_path = '/persisted_storage/Projects/mnist/saver'
model_name = 'model.ckpt'

# mnist_eval.py
eval_interval_sec = 10
data_path = '/persisted_storage/datasets/MNIST'
