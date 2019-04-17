import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

N = 100
D = 3
A_v = tf.placeholder(tf.float32, name='A') # partially defined to be flexible
x_v = tf.placeholder(tf.float32, name='x')

b_v = tf.matmul(A_v, x_v)

with tf.Sessions() as sess:
    out = sess.run(b_v,
        feed_dict = {
            A_v: np.random.randn(5,5),
            x_v: np.random.randn(5,1)
        }
    )
    print(out)
    print(type(out))

w = tf.Variable(20.0, 'w')
J = w*w + w + 1

eta = 0.3
train_op = tf.train.GradientDescentOptimizer(eta).minimize(J)

J_vals = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for t in range(50):
        sess.run(train_op)
        J_vals.append(sess.run(J, feed_dict={w:w.eval()}))

    print(w.eval())
