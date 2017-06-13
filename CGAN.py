import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import initializers
mnist = input_data.read_data_sets("./MNIST", one_hot=True)

total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10
LAMBDA = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])

batch_norm = partial(tf.contrib.layers.batch_norm, decay=0.9, epsilon=1e-5, scale=True, is_training=True)
lrelu = lambda x, leak=0.2: tf.maximum(x, leak * x)


def fully_connected(in_c, out_c, name):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w', shape=[in_c, out_c], initializer=initializers.xavier_initializer())
        b = tf.get_variable(name='b', shape=[out_c], initializer=initializers.xavier_initializer())

    def wx_b(x,actication=True):
        x = tf.matmul(x, w) + b
        if actication == True:
            return tf.nn.relu((x))
        else:
            return x

    return wx_b
def selu(x):
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def generator(noise, labels):
    with tf.variable_scope('generator'):


        inputs = tf.concat([noise, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=None)
        hidden = lrelu(batch_norm(hidden))


        hidden = tf.concat([hidden, labels], 1)
        hidden = tf.layers.dense(hidden, n_hidden,
                                  activation=None)
        hidden = lrelu(batch_norm(hidden))

        output = tf.layers.dense(hidden, n_input,
                                 activation=tf.nn.sigmoid)

    return output

def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()


        inputs = tf.concat([inputs, labels], 1)


        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        hidden = tf.concat([hidden, labels], 1)
        hidden = tf.layers.dense(hidden, n_hidden,
                                 activation=tf.nn.relu)
        hidden = tf.concat([hidden, labels], 1)
        hidden = tf.layers.dense(hidden, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1,
                                 activation=None)



    return output


def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])


G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)



# loss_D = tf.reduce_mean(D_gene) - tf.reduce_mean(D_real)
loss_G = -tf.reduce_mean(D_gene)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))

loss_D = D_loss_real + D_loss_fake

# loss_G = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

alpha = tf.random_uniform(
    shape=[batch_size, 1],
    minval=0.,
    maxval=1.
)

# differences = G - X
# interpolates = X + (alpha * differences)
# gradients = tf.gradients(discriminator(interpolates,Y,reuse=True), [interpolates])[0]
# slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
# gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
# loss_D += LAMBDA * gradient_penalty


vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator')
#
# train_D = tf.train.AdamOptimizer(1e-4,beta1=0.5,beta2=0.9).minimize(loss_D,
#                                             var_list=vars_D)
# train_G = tf.train.AdamOptimizer(1e-4,beta1=0.5,beta2=0.9).minimize(loss_G,
#                                             var_list=vars_G)
train_D = tf.train.AdamOptimizer(1e-3).minimize(loss_D,
                                            var_list=vars_D)
train_G = tf.train.AdamOptimizer(1e-3).minimize(loss_G,
                                            var_list=vars_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Y: batch_ys, Z: noise})

    print 'Epoch:', '%04d' % (epoch + 1), \
        'D loss: {:.4}'.format(loss_val_D), \
        'G loss: {:.4}'.format(loss_val_G)

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G,
                           feed_dict={Y: mnist.test.labels[:sample_size],
                                      Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)),cmap ='gray')
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)),cmap ='gray')

        plt.savefig('logs/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
