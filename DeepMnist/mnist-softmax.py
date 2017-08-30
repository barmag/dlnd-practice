import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot=True, reshape=True)
print(mnist.train.images.shape)

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.truncated_normal([784, 10], mean=0., stddev=0.03))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.bias_add(tf.matmul(x, w), b)
out = tf.nn.softmax(y)

training_labels = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(training_labels * tf.log(out), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

# measure accuracy
correct_predictions = tf.equal(tf.arg_max(out, 1), tf.arg_max(training_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_feature, batch_labels = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x: batch_feature, training_labels: batch_labels})

        
        validation_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, training_labels: mnist.validation.labels})
        print(validation_accuracy)

    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, training_labels: mnist.test.labels})
    print("test accuracy: {}".format(test_accuracy))