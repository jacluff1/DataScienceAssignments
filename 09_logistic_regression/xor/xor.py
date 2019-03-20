# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

# import data
data = pd.read_csv("xor.csv")

# get data as np.arrays
N = data.shape[0]
x1 = data.x1.values
x2 = data.x2.values
y = data.y.astype(int)

# construct design matrix
X = np.column_stack((np.ones((N,1)), x1, x2, x1*x2))

def sigmoid(h):
    return 1/(1 + np.exp(-h))

def cross_entropy(y, p_hat):
    return -np.sum(y*np.log(p_hat) + (1 - y)*np.log(1 - p_hat))

def accuracy(y, p_hat):
    return np.mean(y == np.round(p_hat))

w = np.random.randn(X.shape[1])
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

xm = np.linspace(-6,6,2000)
ym = -(w[0] + w[1]*xm) / (w[2] + w[3]*xm)

# make new figure
fig,ax = plt.subplots()
# fig.suptitle("Centered Around Mean")
ax.scatter(x1, x2, c=y, label='data', alpha=.5)
ax.plot(xm,ym, c='r', linewidth=2, label='Decision Line')
ax.set_aspect(1)
ax.set_title("'Xor' Logistic Regression Classification Problem")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_ylim([-6,6])
ax.legend()

# save the figure
fig.savefig("Xor.pdf")
plt.close(fig)

print("Accuracy: {:0.4f}".format(accuracy(y,p_hat)))
