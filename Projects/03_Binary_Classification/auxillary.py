# import dependencies
import numpy as np
import pandas as pd
import os
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#===============================================================================
# basic functions
#===============================================================================

def sigmoid(h):
    return 1/(1 + np.exp(-h))

def cross_entropy(y,p_hat):
    return -np.sum(y*np.log(p_hat) + (1 - y)*np.log(1 - p_hat))

def accuracy(y,p_hat):
    return np.mean(y == np.round(p_hat))

#===============================================================================
# solve
#===============================================================================

def solve(PHI,y,lambda2):
    if lambda2 == 0:
        return np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(Y))
    else:
        P = PHI.shape[1]
        I = np.eye(P)
        I[0,0] = 0
        return np.linalg.solve(PHI.T.dot(PHI) + lambda2*I, PHI.T.dot(y))

def gradient_descent(PHI,y,lambda1,lambda2, eta=1e-3, epochs=1e5, save_curve=False):
    N = PHI.shape[0]
    # make sure epochs is an integer
    epochs = int(epochs)
    # start with random w
    w = np.random.randn(PHI.shape[1])
    # instantiate J for cross entropy + regularization
    J = [0]*epochs
    # start gradient descent
    for epoch in range(epochs):
        # pdb.set_trace()
        p_hat = sigmoid(PHI.dot(w))
        J[epoch] = ( cross_entropy(y, p_hat) + ( lambda1*np.sum(np.abs(w)) + lambda2*np.sum(w**2) ) / 2 ) / N
        w -= eta * ( PHI.T.dot(p_hat - y) + lambda1*np.sign(w) + lambda2*w )/N

    # plot objective junction
    if save_curve:
        # make sure a directory exists to store J-curves
        if not os.path.isdir("J"): os.mkdir("J")
        filename = f"J/l1{lambda1}_l2{lambda2}_eta{eta}_epochs{epochs}.pdf"
        # make figure
        fig = plt.figure()
        fig.suptitle(f"$\\lambda_1$: {lambda1}, $\\lambda_2$: {lambda2}, $\\eta$: {eta}, epochs: {epochs}", fontsize=20)
        plt.plot(J)
        fig.savefig(filename)
        print(f"saved {filename}")
        plt.close(fig)

    # output
    return w
