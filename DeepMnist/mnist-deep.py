import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True, reshape=False)

def get_weights(shape):
    init_values = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_values)

def get_bias(shape):
    init_values = tf.constant(0.1, shape=shape)
    return tf.Variable(init_values)

def get_conv_layer(x, W):
    conv = tf.nn.conv2d(x, W, [1,1,1,1], padding="SAME")
    return conv;

def get_max_pool(x):
    max_p = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    return max_p

x = tf.placeholder(tf.float32, (None, 28, 28, 1))

# conv layer 1
w_c1 = get_weights([5,5,1,32])
b_c1 = get_bias([32])

o_conv1 = tf.nn.relu(get_conv_layer(x, w_c1) + b_c1)
o_max_p1 = get_max_pool(o_conv1)

# conv layer 2
w_c2 = get_weights([5,5,32,64])
b_c2 = get_bias([64])

o_conv2 = tf.nn.relu(get_conv_layer(o_max_p1, w_c2) + b_c2)
o_max_p2 = get_max_pool(o_conv2)

# fully connected layer
w_fc1 = get_weights([7*7*64, 1024])
b_fc1 = get_bias(1024)

o_flat = tf.reshape(o_max_p2, [-1, 7*7*64])
o_fc1 = tf.nn.relu(tf.matmul(o_flat, w_fc1) + b_fc1)

# dropout
keep_prop = tf.placeholder(tf.float32)
o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)

# output layer
w_fc_out = get_weights([1024, 10])
b_fc_out = get_bias([10])

final_out = tf.matmul(o_fc1_drop, w_fc_out) + b_fc_out