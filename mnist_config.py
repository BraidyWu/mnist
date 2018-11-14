# Configuration

num_input = 784
num_output = 10
num_hidden = 500

batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 1e-4
training_step = 20000
moving_average_decay = 0.99

model_save_path = '/persisted_storage/Projects/mnist/saver'
model_name = 'model.ckpt'
