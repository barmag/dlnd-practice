import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# True, perform training, else load model from file
training = False

mnist = input_data.read_data_sets(".", one_hot=True)
# hyper-parameters
learning_rate = 0.001
training_epoch = 20
batch_size = 128
display_step = 1

n_inputs = 784
n_classes = 10

n_hidden_layer = 256    # layer number of features

weights = [tf.Variable(tf.random_normal([n_inputs, n_hidden_layer])),
                       tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))]

biases = [tf.random_normal([n_hidden_layer]), tf.random_normal([n_classes])]

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
x = tf.reshape(x, [-1, n_inputs])

# hidden layer ReLU activation
layer_1 = tf.add(tf.matmul(x, weights[0]), biases[0])
layer_1 = tf.nn.relu(layer_1)

# output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights[1]), biases[1])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = "./model.ckpt"
saver = tf.train.Saver()
if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # training cycle
        for epoch in range(training_epoch):
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # run training
                sess.run(optimizer, feed_dict={x: np.reshape(batch_x, [-1, n_inputs]), y: batch_y})

            # display progress
            if (epoch+1) % 5 == 0:
                valid_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

                print("Epoch {:<3} - validation accuracy: {}".format(epoch+1, valid_accuracy))
    
        saver.save(sess, save_file)
        print("model saved")
else:
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

        print("test accuracy {}".format(test_accuracy))