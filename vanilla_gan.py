import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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



def vanilla_gan(X,Z):


    def generator(noise, reuse=False):
        with tf.variable_scope('generator'):
            G1 = tf.contrib.layers.fully_connected(noise, 256)
            G2 = tf.contrib.layers.fully_connected(G1, n_input)

        return G2

    def discriminator(inputs, reuse=None):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            D1 = tf.contrib.layers.fully_connected(inputs, 256)
            D2 = tf.contrib.layers.fully_connected(D1, 256)
            D3 = tf.contrib.layers.fully_connected(D2, 1, activation_fn=None)

        return D3


    G = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G,reuse=True)

    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

    loss_D = D_loss_real + D_loss_fake

    loss_G = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

    return G, loss_D, loss_G

def main():

    with tf.device('/cpu:0'):
        with tf.device('/gpu:2'):

            X = tf.placeholder(tf.float32, [None, n_input])
            Z = tf.placeholder(tf.float32, [None, n_noise])

            G, loss_D, loss_G = vanilla_gan(X, Z)

            D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
            train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)


    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    total_batch = int(mnist.train.num_examples/batch_size)


    for epoch in range(total_epoch):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            noise = get_noise(batch_size)

            _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
            _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

        print 'Epoch:', '%04d' % (epoch + 1), \
              'D loss: {:.4}'.format(loss_val_D), \
              'G loss: {:.4}'.format(loss_val_G)


        sample_size = 10
        noise = get_noise(sample_size)

        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('logs/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    main()