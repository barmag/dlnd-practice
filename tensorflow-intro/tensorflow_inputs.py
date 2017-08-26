import tensorflow as tf

# string variable input to tensorflow
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hellow World!'})
    print (output)
