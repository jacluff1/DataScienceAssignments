# import dependencies
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#===============================================================================
# set up dataas
#===============================================================================

D = 2
K = 3
N = int(K*1.5e4)

X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0,X1,X2))

y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))

#===============================================================================
# functions
#===============================================================================

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

def shuffle(*args):
    idx = np.random.permutation(len(args[0]))
    return [X[idx] for X in args]

def ReLU(H):
    return H*(H>0)

def softmax(H):
    eH = np.exp(H)
    return eH / eH.sum(axis=1, keepdims=True)

def feed_forward(X,W,b):
    Z = {}
    Z[1] = ReLU(np.matmul(X,W[1]) + b[1])
    Z[2] = softmax(np.matmul(Z[1],W[2]) + b[2])
    return Z

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def accuracy(y,P_hat):
    return np.mean(y == P_hat.argmax(axis=1))

#===============================================================================
# setup
#===============================================================================

M = 6
Y = one_hot_encode(y)

W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

mu = 0.9
gamma = 0.999
epsilon = 1e-8
eta = 1e-3
epochs = 200

#===============================================================================
# back_propagation
#===============================================================================

def Adam():

    dZ,dH,dW,db = {},{},{},{}
    mW,mb,vW,vb = {1:0, 2:0},{1:0, 2:0},{1:0, 2:0},{1:0, 2:0}

    J = np.zeros(epochs)
    W = {1:W1.copy(), 2:W2.copy()}
    b = {1:b1.copy(), 2:b2.copy()}

    for epoch in range(1,epochs):

        Z = feed_forward(X,W,b)
        J[epoch] = cross_entropy(Y,Z[2])

        # vanilla
        dH[2] = Z[2] - Y
        dW[2] = np.matmul(Z[1].T,dH[2])
        db[2] = dH[2].sum(axis=0)
        # moment
        mW[2] = mu*mW[2] + (1-mu)*dW[2]
        mb[2] = mu*mb[2] + (1-mu)*db[2]
        # variance
        vW[2] = mu*vW[2] + (1-gamma)*dW[2]**2
        vb[2] = mu*vb[2] + (1-gamma)*db[2]**2
        # update
        W[2] -= eta/np.sqrt(vW[2]/(1-gamma**epoch) + epsilon) * mW[2]/(1-mu**epoch)
        b[2] -= eta/np.sqrt(vb[2]/(1-gamma**epoch) + epsilon) * mb[2]/(1-mu**epoch)

        # vanilla
        dZ[1] = np.matmul(dH[2],W[2].T)
        dH[1] = dZ[1] * (Z[1] > 0)
        dW[1] = np.matmul(X.T,dH[1])
        db[1] = dH[1].sum(axis=0)
        # moment
        mW[1] = mu*mW[1] + (1-mu)*dW[1]
        mb[1] = mu*mb[1] + (1-mu)*db[1]
        # variance
        vW[1] = mu*vW[1] + (1-gamma)*dW[1]**2
        vb[1] = mu*vb[1] + (1-gamma)*db[1]**2
        # update
        W[1] -= eta/np.sqrt(vW[1]/(1-gamma**epoch) + epsilon) * mW[1]/(1-mu**epoch)
        b[1] -= eta/np.sqrt(vb[1]/(1-gamma**epoch) + epsilon) * mb[1]/(1-mu**epoch)

    #output
    return J[1:]

def RMSProp_with_momentum():

    dZ,dH,dW,db = {},{},{},{}
    vW,vb,GW,Gb = {1:0, 2:0},{1:0, 2:0},{1:0, 2:0},{1:0, 2:0}

    J = np.zeros(epochs)
    W = {1:W1.copy(), 2:W2.copy()}
    b = {1:b1.copy(), 2:b2.copy()}

    for epoch in range(epochs):

        Z = feed_forward(X,W,b)
        J[epoch] = cross_entropy(Y,Z[2])

        # vanilla
        dH[2] = Z[2] - Y
        dW[2] = np.matmul(Z[1].T,dH[2])
        db[2] = dH[2].sum(axis=0)
        # RMSProp
        GW[2] = gamma*GW[2] + (1-gamma)*dW[2]**2
        Gb[2] = gamma*Gb[2] + (1-gamma)*db[2]**2
        vW[2] = mu*vW[2] - (eta/np.sqrt(GW[2] + epsilon))*dW[2]
        vb[2] = mu*vb[2] - (eta/np.sqrt(Gb[2] + epsilon))*db[2]
        # update
        # W[2] -= (eta/np.sqrt(GW[2] + epsilon)) * dW[2]
        # b[2] -= (eta/np.sqrt(Gb[2] + epsilon)) * db[2]
        W[2] += vW[2]
        b[2] += vb[2]

        # vanilla
        dZ[1] = np.matmul(dH[2],W[2].T)
        dH[1] = dZ[1] * (Z[1] > 0)
        dW[1] = np.matmul(X.T,dH[1])
        db[1] = dH[1].sum(axis=0)
        #RMSprop
        GW[1] = gamma*GW[1] + (1-gamma)*dW[1]**2
        Gb[1] = gamma*Gb[1] + (1-gamma)*db[1]**2
        vW[1] = mu*vW[1] - (eta/np.sqrt(GW[1] + epsilon))*dW[1]
        vb[1] = mu*vb[1] - (eta/np.sqrt(Gb[1] + epsilon))*db[1]
        # update
        # W[1] -= (eta/np.sqrt(GW[1] + epsilon))*dW[1]
        # b[1] -= (eta/np.sqrt(Gb[1] + epsilon))*db[1]
        W[1] += vW[1]
        b[1] += vb[1]

    # output
    return J

#===============================================================================
# plot
#===============================================================================

def plot():

    J_adam = Adam()
    J_RMS = RMSProp_with_momentum()

    fig,ax = plt.subplots()
    fig.suptitle("Comparing RMSProp + Momentum with Adam")
    ax.plot(J_adam, label="Adam")
    ax.plot(J_RMS, label="RMSProp with Momentum")
    ax.set_xlabel("epochs")
    ax.set_ylabel("J")
    ax.legend()

    fig.savefig("ComparingRMSPropWithAdam.pdf")
    plt.close(fig)
