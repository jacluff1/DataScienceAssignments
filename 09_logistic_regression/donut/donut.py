# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

# import data
data = pd.read_csv("donut.csv", names=['x1','x2','c'], delim_whitespace=True)

N = data.shape[0]


"""
make X. equation for an ellipse is (x_1-mu_1)^2 / a^2 + (x_2-mu_2)^2 / b^2 = 1.
To engineer the features, expand the terms to get feautures (x_1^2) (x_1 mu_1) (mu_1)^2 (x_2^2) (x_2 mu_2) (mu_2)^2. the mu factors are absorbed in the weights

OR

we can center the data around 0 and just use features: bias + x_1^2 x_2^2
"""

# center the data around 0
x1 = data.x1.values - data.x1.values.sum()/N
x2 = data.x2.values - data.x2.values.sum()/N
# X = np.column_stack((np.ones((N,1)), x1**2, x2**2))
X = np.column_stack((np.ones((N,1)), x1**2, x2**2, x1*x2))
y = data.c.astype(int)

def sigmoid(h):
    return 1/(1 + np.exp(-h))

def cross_entropy(y, p_hat):
    return -np.sum(y*np.log(p_hat) + (1 - y)*np.log(1 - p_hat))

def accuracy(y, p_hat):
    return np.mean(y == np.round(p_hat))

w = np.random.randn(X.shape[1])
# w = np.random.randn(4)
eta = 1e-5
epochs = int(1e5)
J = [0]*epochs

for epoch in range(epochs):
    p_hat = sigmoid(X.dot(w))
    J[epoch] = cross_entropy(y, p_hat)
    w -= eta*X.T.dot(p_hat - y)

# plot objective junction
fig = plt.figure()
plt.plot(J)
fig.savefig("J.pdf")
plt.close(fig)

xrange = 8
xm = np.linspace(-xrange,xrange,100)
xm,ym = np.meshgrid(xm,xm)
Z = w[0] + w[1]*xm**2 + w[2]*ym**2 + w[3]*xm*ym

# make new figure
fig,ax = plt.subplots()
fig.suptitle("'Donut' Logistic Regression Classification Problem")
ax.scatter(x1, x2, c=y, label='data', alpha=.5)
ax.contour(xm,ym,Z, [0], colors='r', linewidths=2)
ax.set_aspect(1)
ax.set_title("Centered Around Mean")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend()

# save the figure
fig.savefig("donut.pdf")
plt.close(fig)

print("Accuracy: {:0.4f}".format(accuracy(y,p_hat)))
