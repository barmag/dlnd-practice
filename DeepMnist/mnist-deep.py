import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)

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


