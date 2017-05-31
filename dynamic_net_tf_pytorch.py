import tensorflow as tf
import torch
from torch.autograd import Variable
import random
def dynamic_pytorch():

    class DynamicNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we construct three nn.Linear instances that we will use
            in the forward pass.
            """
            super(DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear(D_in, H)
            self.middle_linear = torch.nn.Linear(H, H)
            self.output_linear = torch.nn.Linear(H, D_out)

        def forward(self, x):
            """
            For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
            and reuse the middle_linear Module that many times to compute hidden layer
            representations.

            Since each forward pass builds a dynamic computation graph, we can use normal
            Python control-flow operators like loops or conditional statements when
            defining the forward pass of the model.

            Here we also see that it is perfectly safe to reuse the same Module many
            times when defining a computational graph. This is a big improvement from Lua
            Torch, where each Module could be used only once.
            """
            h_relu = self.input_linear(x).clamp(min=0)
            for _ in range(random.randint(0, 3)):
                h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)
            return y_pred

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    # Construct our model by instantiating the class defined above
    model = DynamicNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = (torch.nn.MSELoss(size_average=False))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def dynamic_tf():
    N, D_in, H, D_out = 64, 1000, 100, 10

    x = tf.get_variable(name = "x",shape=[N,D_in],initializer=tf.random_normal_initializer(),trainable=False)
    y = tf.get_variable(name = "y",shape=[N,D_out],initializer=tf.random_normal_initializer(),trainable=False)

    input_linear = tf.contrib.layers.fully_connected(x, D_in)
    middle_linear = tf.contrib.layers.fully_connected(input_linear, H)

    limit = tf.cast(tf.random_uniform([],minval=0,maxval=3),tf.int32)

    ii = tf.constant(0)

    def cond(i,l,middle_linear):
        return i<l

    def body(i,l,middle_linear):
        middle_linear = tf.contrib.layers.fully_connected(middle_linear, H)
        i = i + 1
        return  i, l, middle_linear

    _,_,middle_linear = tf.while_loop(cond,body,[ii,limit,middle_linear])


    output_linear = tf.contrib.layers.fully_connected(middle_linear,D_out,activation_fn=None)

    # loss = tf.reduce_mean(tf.squared_difference(output_linear,y))
    loss = tf.reduce_mean(tf.nn.l2_loss(output_linear-y))
    train_op = tf.train.MomentumOptimizer(learning_rate=1e-4,momentum=0.9).minimize(loss)


    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


    for i in range(500):
        _,loss_p, lim = sess.run([train_op,loss,limit])
        print(i,loss_p,lim)

    sess.close()

if __name__ == '__main__':

    #dynamic_tf()
    dynamic_pytorch()