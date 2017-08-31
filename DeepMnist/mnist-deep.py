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
b_fc1 = get_bias([1024])

o_flat = tf.reshape(o_max_p2, [-1, 7*7*64])
o_fc1 = tf.nn.relu(tf.matmul(o_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)

# output layer
w_fc_out = get_weights([1024, 10])
b_fc_out = get_bias([10])

final_out = tf.matmul(o_fc1_drop, w_fc_out) + b_fc_out

validation_labels = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=validation_labels, logits=final_out)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(validation_labels, 1), tf.arg_max(final_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        # display progress
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], validation_labels:batch[1], keep_prob:1.})
            print("step {}, training accuracy: {}".format(i, train_accuracy))
        optimizer.run(feed_dict={x:batch[0], validation_labels:batch[1], keep_prob:0.5})
    test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, validation_labels:mnist.test.labels, keep_prob:1.0})
    print("test accuracy: {}".format(test_accuracy))