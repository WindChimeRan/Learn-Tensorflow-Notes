import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


# length of RNN
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 2
OUTPUT_SIZE = 1

#hidden_size
CELL_SIZE = 100
LSTM_LAYERS = 5
LR = 0.006


BATCH_START = 0

def get_batch():

    """
    :return returned (seq, res and xs: shape (batch, step, input))
    """

    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)

    seq1 = np.sin(xs).reshape(BATCH_SIZE,TIME_STEPS,1)
    seq2 = np.cos(xs).reshape(BATCH_SIZE,TIME_STEPS,1)
    seq = np.concatenate([seq1,seq2],axis=2)

    res = 0.5*np.sin(2*xs)+0.1*np.cos(xs)*np.sin(xs)+0.7*np.cos(9*xs)*np.cos(9*xs)

    BATCH_START += TIME_STEPS

    return (seq, res[:, :, np.newaxis], xs)

def l2(labels, logits):
    return tf.nn.l2_loss(labels - logits)

def compute_cost(pred,ys):

    losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [tf.reshape(pred, [-1], name='reshape_pred')],
        [tf.reshape(ys, [-1], name='reshape_target')],
        [tf.ones([BATCH_SIZE * TIME_STEPS], dtype=tf.float32)],
        average_across_timesteps=True,
        softmax_loss_function=l2,
        name='losses'
    )



    with tf.name_scope('average_cost'):
        cost = tf.div(
            tf.reduce_sum(losses, name='losses_sum'),
            BATCH_SIZE,
            name='average_cost')
    return  cost

def _weight_variable(shape, name='weights'):
    initializer = tf.random_normal_initializer(mean=0., stddev=1., )
    return tf.get_variable(shape=shape, initializer=initializer, name=name)


def _bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)




def main(argv=None):

    with tf.device('/cpu:0'):
        with tf.device('/gpu:3'):

            with tf.name_scope('inputs'):
                xs = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE], name='xs')
                ys = tf.placeholder(tf.float32, [None, TIME_STEPS, OUTPUT_SIZE], name='ys')

            with tf.variable_scope('in_hidden'):

                l_in_x = tf.reshape(xs, [-1, INPUT_SIZE], name='2_2D')  # (batch*n_step, in_size)
                Ws_in = _weight_variable([INPUT_SIZE, CELL_SIZE])
                bs_in = _bias_variable([CELL_SIZE, ])
                # l_in_y = (batch * n_steps, cell_size)

                with tf.name_scope('Wx_plus_b'):
                    l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
                    # reshape l_in_y ==> (batch, n_steps, cell_size)
                    l_in_y = tf.reshape(l_in_y, [-1, TIME_STEPS, CELL_SIZE], name='2_3D')

            with tf.variable_scope('LSTM_cell'):

                dropout = lambda cell,output_keep_prob:tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob = 0.8)
                single_lstm = lambda :tf.contrib.rnn.BasicLSTMCell(CELL_SIZE, forget_bias=1.0, state_is_tuple=True)
                single_gru = lambda: tf.contrib.rnn.GRUCell(CELL_SIZE)

                stacked_lstm = [dropout(single_lstm(),output_keep_prob=1) for _ in range(LSTM_LAYERS)]
                lstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm,state_is_tuple = True)

                # lstm_cell = single_gru()
                # lstm_cell = single_lstm()

                with tf.name_scope('initial_state'):
                    cell_init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

                '''
                dynamic_rnn?

                cell_outputs = []
                state = _initial_state
                with tf.variable_scope("RNN"):
                for time_step in range(num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output_time_step, state) = cell(inputs[:, time_step, :], state)
                    cell_outputs.append(cell_output_time_step)
                final_state = cell_output[-1]
                '''

                cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                    lstm_cell, l_in_y, initial_state=cell_init_state, time_major=False)

            with tf.variable_scope('out_hidden'):

                l_out_x = tf.reshape(cell_outputs, [-1, CELL_SIZE], name='2_2D')
                Ws_out = _weight_variable([CELL_SIZE, OUTPUT_SIZE])
                bs_out = _bias_variable([OUTPUT_SIZE, ])
                with tf.name_scope('Wx_plus_b'):
                    pred = tf.matmul(l_out_x, Ws_out) + bs_out

                # classification head : softmax(wx+b) . cross_entropy
                # regression head : l2

            with tf.name_scope('cost'):
                cost = compute_cost(pred, ys)


            with tf.name_scope('train'):
                train_op = tf.train.AdamOptimizer(LR).minimize(cost)

    tf.summary.scalar('cost', cost)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    plt.ion()
    plt.show()
    for step in range(2000):

        seq, res, xxs = get_batch()


        if step == 0:
            feed_dict = {
                    xs: seq,
                    ys: res,
                    # create initial state
            }
        else:
            feed_dict = {
                xs: seq,
                ys: res,
                cell_init_state: state    # use last state as the initial state for this run
            }



        start_time = time.time()
        _, loss, state, pred_y = sess.run(
            [train_op, cost, cell_final_state, pred],
            feed_dict=feed_dict)
        duration = time.time() - start_time

        # plotting
        plt.plot(xxs[0, :], res[0].flatten(), 'r', xxs[0, :], pred_y.flatten()[:TIME_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if step % 5 == 0:
            print('step {:d} \t loss = {:.8f}, ({:.3f} sec/step)'.format(step, loss, duration))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, step)

if __name__ == '__main__':

    tf.app.run()
    # a,b,c = get_batch()
    # print(a.shape)