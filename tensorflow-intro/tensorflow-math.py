
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
tx = tf.constant(10)
ty = tf.constant(2)
tone = tf.constant(1)

z = tf.subtract(tf.cast(tf.divide(tx, ty), tf.int32), tone)
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
