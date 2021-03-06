import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST', one_hot=True)


total_epoch = 20
batch_size = 100
learning_rate = 0.0002

n_hidden = 256
n_input = 28 * 28
n_noise = 128

def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))

ACTIVATION = 'relu'



# def batch_norm(input_layer):
#
#     dimension = input_layer.get_shape().as_list()[-1]
#     mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
#     beta = tf.get_variable('beta', dimension, tf.float32,
#                                initializer=tf.constant_initializer(0.0, tf.float32))
#     gamma = tf.get_variable('gamma', dimension, tf.float32,
#                                 initializer=tf.constant_initializer(1.0, tf.float32))
#     bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON = 0.001)
#
#     return bn_layer

def selu(x):
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def vanilla_gan(X,Z):

    batch_size = 100
    dim_W1 = 256
    dim_W2 = 128
    dim_W3 = 64
    dim_channel = 1


    batch_norm = partial(tf.contrib.layers.batch_norm, decay=0.9, epsilon=1e-5, scale=True, is_training=True)
    create_variables = partial(tf.get_variable, initializer=tf.contrib.layers.xavier_initializer())
    lrelu = lambda x, leak=0.2: tf.maximum(x, leak * x)

    def generator(Z, reuse=False):
        with tf.variable_scope('generator'):


            gen_W1 = create_variables(name='gen_W1',shape = [128,256])
            gen_W2 = create_variables(name='gen_W2', shape= [256, dim_W2 * 7 * 7])
            gen_W3 = create_variables(name='gen_W3',shape = [5, 5, dim_W3, dim_W2])
            gen_W4 = create_variables(name='gen_W4', shape= [5, 5, dim_channel, dim_W3])



            h1 = tf.nn.relu(batch_norm(tf.matmul(Z, gen_W1)))
            h2 = tf.nn.relu(batch_norm(tf.matmul(h1, gen_W2)))

            h2 = tf.reshape(h2, [-1, 7, 7, dim_W2])

            output_shape_l3 = [batch_size, 14, 14, dim_W3]
            h3 = tf.nn.conv2d_transpose(h2, gen_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
            h3 = tf.nn.relu(batch_norm(h3))

            output_shape_l4 = [tf.shape(Z)[0], 28, 28, dim_channel]
            h4 = tf.nn.conv2d_transpose(h3, gen_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])


            return h4



    def discriminator(image, reuse=None):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            discrim_W1 = create_variables(name='discrim_W1', shape=[5, 5, dim_channel, dim_W3])
            discrim_W2 = create_variables(name='discrim_W2', shape=[5, 5, dim_W3 , dim_W2])
            discrim_W3 = create_variables(name='discrim_W3', shape=[5, 5, dim_W2 , dim_W1])
            discrim_W4 = create_variables(name='discrim_W4', shape=[4096, 256])
            discrim_W5 = create_variables(name='discrim_W5', shape=[256, 1])



            # block = lambda input, w: tf.nn.conv2d(
            #     batch_norm(tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')))
            if ACTIVATION != 'selu':

                block = lambda input, w: selu(
                    tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME'))

                h0 = block(image, discrim_W1)
                h1 = block(h0, discrim_W2)
                h2 = block(h1, discrim_W3)

                h2 = tf.reshape(h2, [batch_size, -1])

                h3 = selu((tf.matmul(h2, discrim_W4)))
                h4 = tf.matmul(h3, discrim_W5)

            else:

                block = lambda input, w: lrelu(
                    batch_norm(tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')))

                h0 = block(image,discrim_W1)
                h1 = block(h0, discrim_W2)
                h2 = block(h1, discrim_W3)


                h2 = tf.reshape(h2,[batch_size,-1])

                h3 = lrelu((tf.matmul(h2, discrim_W4)))
                h4 = tf.matmul(h3, discrim_W5)

            return h4



    G = generator(Z)
    D_real = discriminator(X,reuse=False)
    D_fake = discriminator(G,reuse=True)

    # D_loss_real = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    # D_loss_fake = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    #
    # loss_D = D_loss_real + D_loss_fake
    #
    # loss_G = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))


    loss_D = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
    loss_G = -tf.reduce_mean(D_fake)


    # alpha = tf.random_uniform(
    #     shape=[batch_size, 1],
    #     minval=0.,
    #     maxval=1.
    # )
    #
    # differences = G - X
    # interpolates = tf.reshape(X,[-1,28*28]) + (alpha * tf.reshape(differences,[-1,28*28]))
    # interpolates = tf.reshape(interpolates,[-1,28,28,1])
    # gradients = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    # loss_D += 10 * gradient_penalty




    return G, loss_D, loss_G

def main():

    with tf.device('/cpu:0'):

        with tf.device('/gpu:3'):

            X = tf.placeholder(tf.float32, [None, 28,28,1])
            Z = tf.placeholder(tf.float32, [None, n_noise])
            G, loss_D, loss_G = vanilla_gan(X, Z)

            D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            # train_D = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(loss_D,
            #                                                                       var_list=D_var_list)
            # train_G = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(loss_G,
            #                                                                       var_list=G_var_list)

            train_D = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
            train_G = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    total_batch = int(mnist.train.num_examples/batch_size)


    for epoch in range(total_epoch):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)


            batch_xs = np.reshape(batch_xs,[batch_size,28,28,1])
            noise = get_noise(batch_size)

            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
            _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})


        print 'Epoch:', '%04d' % (epoch + 1), \
              'D loss: {:.4}'.format(loss_val_D), \
              'G loss: {:.4}'.format(loss_val_G)


        sample_size = 100
        noise = get_noise(sample_size)

        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(9, 9, figsize=(9, 9))

        for i in range(9):
            for j in range(9):
                ax[i][j].set_axis_off()
                ax[i][j].imshow(np.reshape(samples[i*9+j], (28, 28)),cmap ='gray')

        plt.savefig('logs/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    main()

