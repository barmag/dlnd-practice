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

with tf.Session() as sess:
    sess.run(init)
