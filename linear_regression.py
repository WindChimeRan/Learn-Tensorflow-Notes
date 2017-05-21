import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Training Data
train_size = 2000
train_X = np.linspace(0,50,train_size)
test_Y = train_X+2*np.sin(1.5*train_X)
train_Y = train_X+2*np.sin(1.5*train_X) + random.gauss(0,0.2)

learning_rate = 0.01
training_epochs = 200


def tf_lr(train_X,train_Y,learning_rate,training_epochs):

    n_samples = train_X.shape[0]

    X = tf.placeholder("float",[n_samples])
    Y = tf.placeholder("float",[n_samples])


    W = tf.get_variable(name="weight",initializer=tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))  #name = bias

    pre_y = X*W+b

    loss = tf.reduce_mean(tf.nn.l2_loss(pre_y-Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)
    for epoch in range(training_epochs):
        _,training_loss = sess.run([optimizer,loss],feed_dict={X:train_X,Y:train_Y})
        mes = "epoch\t"+str(epoch)+"\t"+str(training_loss)
        print(mes)

    w = sess.run(W)
    b = sess.run(b)

    return w,b


def plotAns(pre_y,test_y):
    plt.plot(test_y, 'ro', label='Original data')
    plt.plot(pre_y, label='Fitted line')
    plt.show()

if __name__ == '__main__':

    w,b = regressor = tf_lr(train_X,train_Y,learning_rate,training_epochs)
    print(w,b)
    plotAns(w*train_X+b,test_Y)
