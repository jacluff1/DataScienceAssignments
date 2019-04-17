import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

X_v = T.matrix('X')
Y_v = T.matrix('Y')

W1 = theano.shared(W1_0, 'W1')
b1 = theano.shared(b1_0, 'b1')
W2 = theano.shared(W2_0, 'W2')
b2 = theano.shared(b2_0, 'b2')

Z1_v = T.nnet.relu(X_v.dot(W1) + b1)
P_v = T.nnet.softmax(Z1_v.dot(W2) + b2)

y_hat_v = T.argmax(P_v)
