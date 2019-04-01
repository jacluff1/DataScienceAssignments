# import dependencies
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#===============================================================================
# function definitions
#===============================================================================

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def d_sigmoid(h):
    eh = np.exp(-h)
    return eh / (1 + eh)**2

#===============================================================================
# plot
#===============================================================================

def plot_it():
    H = np.linspace(-10,10,1000)
    Y = sigmoid(H)
    dY = d_sigmoid(H)

    fig,ax = plt.subplots()
    ax.plot(H,Y, label="sigmoid(h)")
    ax.plot(H,dY, label="d/dh sigmoid(h)")
    ax.vlines(0,0,1, color='r', label="max d/dh of sigmoid(h)")
    ax.set_xlabel("h")
    ax.set_xlim(-10,10)
    ax.set_ylim(0,1)
    plt.legend(loc='best')
    fig.suptitle("A look at sigmoid(h) and d/dh sigmoid(h)")
    fig.savefig("f_df.pdf")
    plt.close(fig)
