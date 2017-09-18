import tensorflow as tf

import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """
    real_input = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels)
                                , name="real_inputs")
    fake_input = tf.placeholder(tf.float32, (None, z_dim), name="fake_inputs")
    lr = tf.placeholder(tf.float32, name="learning_rate")

    return real_input, fake_input, lr


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)

alpha = 0.2     # alpha for leaky relu
def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """

    istraining = True   # flag for batch normalization

    with tf.variable_scope("discriminator", reuse=reuse):
        # input 28*28*3
        # first conv layer, kernel 5, filters 64, strides 2
        # out 14*14*64
        # leaky relu activation, no pooling, no batch normalization
        x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same')
        x1 = tf.maximum(alpha*x1, x1)

        # second conv layer, kernel 5, filters 128, strides 2
        # out 7*7*128
        # leaky relu activation, no pooling, with batch normalization
        x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=istraining)
        x2 = tf.maximum(alpha*x2, x2)

        # third conv layer, kernel 5, filters 256, strides 2
        # out 4*4*256
        # leaky relu activation, no pooling, with batch normalization
        x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=istraining)
        x3 = tf.maximum(alpha*x2, x2)

        # flatten for fully connected output
        # output 1 neuron sigmoid
        flat = tf.reshape(x3, (-1, 4*4*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

    return out, logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(discriminator, tf)

def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    with tf.variable_scope('generator', reuse=not is_train):
        # fully connected layer input
        # output 4*4*512
        x1 = tf.layers.dense(z, 4*4*512)

        #reshape
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        # first transpose conv layer 5 kernel, 256 filters, strides 2
        # output 7*7*256
        # batch norm and leaky relu activation
        x2 = tf.layers.conv2d_transpose(x1, 256, 4, strides=1, padding='valid')
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x2 = tf.maximum(alpha*x2, x2)

        # second transpose conv layer 5 kernel, 128 filters, strides 2
        # output 14*14*128
        # batch norm and leaky relu
        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=is_train)
        x3 = tf.maximum(alpha*x3, x3)

        # output conv layer 5 kernel, 3 filters, strides 2
        # output 28*28*3
        # tanh out
        logits = tf.layers.conv2d_transpose(x3, out_channel_dim, 5, strides=2, padding='same')
        out = tf.tanh(logits)
    
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(generator, tf)