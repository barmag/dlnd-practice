import tensorflow as tf

v = tf.Variable(50)
init = tf.global_variables_initializer()

n_features = 120
n_labels = 5
wv = tf.truncated_normal((n_features, n_labels))
print(wv)
weights = tf.Variable(wv)
bias = tf.Variable(tf.zeros(n_labels))

# softmax
x = tf.nn.softmax([2.0, 1.0, 0.2])

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# cross entropy
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))
