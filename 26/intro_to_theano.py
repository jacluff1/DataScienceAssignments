import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Theano tensor objects
x = T.scalar(name='x')
v = T.vector(name='v')
A = T.matrix(name='A')
X = T.tensor3(name='X')

# make a symbolic theano function
b = A.dot(v)

# make a function that takes values and feeds them into the symbolic function 'b'
matmul = theano.function(
    inputs = [A,v],
    outputs = b
)

# try using the function and plut in actual numbers
A_val = np.random.randn(3,4)
v_val = np.random.randn(4)
b_val = matmul(A_val, v_val)

w = theano.shared(20.0, "w")
J = w**2 + w + 1
w_update = w - 0.3*T.grad(J,w)

train_op = theano.function(
    inputs = [],
    updates = [(w, w_update)],
    ouputs = J
)

J_val = []

for epoch in range(100):
    J_val.append(train_op())

plt.plot(J_val)

print(w.get_value())
