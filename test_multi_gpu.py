import tensorflow as tf


tf.app.flags.DEFINE_integer("NUM_GPUS", 4, "How many GPUs to use")
FLAGS = tf.app.flags.FLAGS

c = []
for i in range(FLAGS.NUM_GPUS):
    with tf.device('/gpu:%d' % i):
        a = tf.constant([1.,2.,3.,4.,5.,6.], shape = [2,3])
        b = tf.constant([1.,2.,3.,4.,5.,6.], shape = [3,2])
        c.append(tf.matmul(a,b))
        

with tf.device('/cpu:0'):

    sum = tf.add_n(c)/4

config = tf.ConfigProto(log_device_placement = True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    print(sess.run(sum))
    print(sess.run(c))

