from __future__ import division, print_function, absolute_import
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
mnist = input_data.read_data_sets("MNIST", one_hot=True)

batch_norm = partial(tf.contrib.layers.batch_norm, decay=0.9, epsilon=1e-5, scale=True, is_training=True)



def autoencoder_share_weight():

    with tf.device('/cpu:0'):
      with tf.device('/gpu:2'):

        X = tf.placeholder(tf.float32, shape = [ None, 28*28 ])

        with tf.variable_scope('layer1',reuse=False):

            w = tf.get_variable(name='w',shape=[784,512],initializer=initializers.xavier_initializer())
            b = tf.get_variable(name='b', shape=[512], initializer=initializers.xavier_initializer())

            x_h = tf.matmul(X,w)+b
            x_h = tf.nn.relu(batch_norm(x_h))

        with tf.variable_scope('layer2',reuse=False):

            w = tf.get_variable(name='w',shape=[512,256],initializer=initializers.xavier_initializer())
            b = tf.get_variable(name='b', shape=[256], initializer=initializers.xavier_initializer())

            x_h = tf.matmul(x_h,w)+b
            x_h = tf.nn.relu(batch_norm(x_h))



        with tf.variable_scope('layer2',reuse=True):
            w = tf.get_variable(name='w', shape=[512, 256])
            b = tf.get_variable(name='b', shape=[256])

            x_r = tf.matmul(tf.add(x_h,b), tf.transpose(w))
            x_r = tf.nn.relu(x_r)

        with tf.variable_scope('layer1', reuse=True):
            w = tf.get_variable(name='w', shape=[784,512])
            b = tf.get_variable(name='b', shape=[512])

            x_r = tf.matmul(tf.add(x_r, b), tf.transpose(w))
            x_r = tf.nn.relu(x_r)

        loss = tf.nn.l2_loss(X-x_r)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        n_steps = 100000
        batch_size = 100


        for i in range(n_steps):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _,loss_train = sess.run([optimizer,loss],feed_dict={X:batch_x})

            print(i,loss_train)


            if i == 20000:

                samples = sess.run(x_r,feed_dict={X:batch_x})
                fig, ax = plt.subplots(9, 9, figsize=(9, 9))

                for ii in range(9):
                    for jj in range(9):
                        ax[ii][jj].set_axis_off()
                        ax[ii][jj].imshow(np.reshape(samples[ii*9+jj], (28, 28)))

                plt.savefig('logs/{}.png'.format(str(i)), bbox_inches='tight')
                plt.close(fig)





def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def autoencoder():


    def fully_connected(in_c, out_c, name):

        with tf.variable_scope(name):
            w = tf.get_variable(name='w', shape=[in_c, out_c], initializer=initializers.xavier_initializer())
            b = tf.get_variable(name='b', shape=[out_c], initializer=initializers.xavier_initializer())

        def wx_b(x, bn):
            x = tf.matmul(x, w) + b
            if bn == True:
                return selu(batch_norm(x))
            else:
                return selu(x)

        return wx_b

    def singleton(cls):
        instances = {}

        def wrapper(*args,**kwargs):
            if cls not in instances:
                instances[cls] = cls(*args,**kwargs)
            return instances[cls]
        return wrapper

    @singleton
    class autoencoder(object):

        def __init__(self):

            self.x = tf.placeholder(tf.float32, shape=[None, 28*28])

            self.encoder1 = fully_connected(784,512,name='encoder1')
            self.encoder2 = fully_connected(512, 256,name='encoder2')
            self.decoder2 = fully_connected(256, 512,name='decoder2')
            self.decoder1 = fully_connected(512, 784,name='decoder1')

        def forward(self):

            with tf.device('/cpu:0'):
                with tf.device('/gpu:2'):

                    x_h = self.encoder1(self.x, bn=True)
                    x_h = self.encoder2(x_h, bn=True)
                    x_r = self.decoder2(x_h, bn=False)
                    x_r = self.decoder1(x_r, bn=False)

            return x_r


    net = autoencoder()

    reconstruct_x = net.forward()
    loss = tf.nn.l2_loss(net.x-net.forward())
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        n_steps = 100000
        batch_size = 100


        for i in range(n_steps):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _,loss_train = sess.run([optimizer,loss],feed_dict={net.x:batch_x})

            print(i,loss_train)


            if i == 200:

                samples = sess.run(reconstruct_x,feed_dict={net.x:batch_x})
                fig, ax = plt.subplots(9, 9, figsize=(9, 9))

                for ii in range(9):
                    for jj in range(9):
                        ax[ii][jj].set_axis_off()
                        ax[ii][jj].imshow(np.reshape(samples[ii*9+jj], (28, 28)))

                plt.savefig('logs/{}.png'.format(str(i)), bbox_inches='tight')
                plt.close(fig)


if __name__ == '__main__':
    autoencoder()
    #autoencoder_share_weight()