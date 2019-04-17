#===============================================================================
# import libraries
#===============================================================================

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

#===============================================================================
# set up
#===============================================================================

D = 2
K = 3
N = int(K*1e3)

X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0,X1,X2))

y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))

#===============================================================================
# basic functions
#===============================================================================

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))
    Y = np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]] = 1
    return Y

def shuffle(*args):
    idx = np.random.RandomState(seed=0).permutation(len(args[0]))
    return [X[idx] for X in args]

def cross_entropy(self,Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def assign_random_weights_and_biases(X,Y,M,**kwargs):

    # dimentions
    N,D = X.shape
    L = len(M)

    # kwargs
    seed = kwargs['seed'] if 'seed' in kwargs else 0
    np.random.seed(seed)

    # assign empty dictionaries for weights and biases
    W,b = {},{}

    # add weights and biases for input layer
    W[1] = np.random.randn(D,M[1])
    b[1] = np.random.randn(M[1])

    # add weights and biases for hidden layers:
    for l in range(2,L):
        W[l] = np.random.randn(M[l-1],M[l])
        b[l] = np.random.randn(M[l])

    # add weights and biases for ouptu layer
    W[L] = np.random.randn(M[L-1],K)
    b[L] = np.random.randn(K)

    # add random Gamma (same shape as Z)
    if kwargs['Gamma']:
        Gamma = {}
        Gamma[1] = np.random.randn(N,D)
        for l in range(2,L+1):
            Gamma[l] = np.random.randn(N,M[l-1])
        return W,b,Gamma
    else:
        return W,b

def L2(W):
    L2 = 0
    for Wl in W.values():
        L2 += (Wl**2).sum()
    return L2/2

def L1(W):
    L1 = 0
    for Wl in W.values():
        L1 += np.abs(Wl).sum()
    return L1

#===============================================================================
# activation functions
#===============================================================================

class softmax:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        eH = np.exp(H)
        assert np.all(np.isfinite(eH)), "NOOOOO! try decreasing learning rate."
        return eH / eH.sum(axis=1, keepdims=True)

    @ staticmethod
    def df(Z,**kwargs):
        return Z * (1 - Z)

class tanh:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        # eH = np.exp(H)
        # eHn = np.exp(-H)
        # return (eH - eHn) / (eH + eHn)
        return np.tanh(H)

    @staticmethod
    def df(Z,**kwargs):
        return 1 - Z**2

class ReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        return H*(H > 0)

    @staticmethod
    def df(Z,**kwargs):
        return 1*(Z > 0)

class LReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        return H*(H >= 0) + H*alpha*(H < 0)

    @staticmethod
    def df(Z,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        return 1*(Z >= 0) + alpha*(Z < 0)

#===============================================================================
# feed forward
#===============================================================================

def feed_forward_train(X,W,b,af,M,**kwargs):

    # set up & dimentions
    Z = {}
    N,D = X.shape
    L = len(W)

    # kwargs
    noise_injection = kwargs['noise_injection'] if 'noise_injection' in kwargs else True
    covariance = kwargs['covariance'] if 'covariance' in kwargs else 1e-3
    p_keep = kwargs['p_keep'] if 'p_keep' in kwargs else {key:1 for key in W}

    # input layer
    mask = 1*(p_keep[1] >= np.random.rand(N))
    NOISE = np.random.randn(*X.shape) * covariance
    # pdb.set_trace()
    Z[1] = af[1].f( np.matmul((X+NOISE)*X[mask],W[1]) + b[1] )

    # hidden layers (if any)
    for l in range(2,L+1):
        mask = 1*(p_keep[l] >= np.random.rand(N))
        NOISE = np.random.randn(N,M[l]) * covariance
        Z[l] = af[l].f( np.matmul((Z[l-1]+NOISE)*Z[l-1][mask],W[l]) + b[l] )

    return Z

def feed_forward_predict(X,W,b,af,M,**kwargs):

    # setup
    Z = {}
    N,D = X.shape
    L = len(W)

    # kwargs
    p_keep = kwargs['p_keep'] if 'p_keep' in kwargs else {key:1 for key in W}

    # input layer
    Z[1] = af[1].f( np.matmul((1/p_keep[1])*X,W[1] ) + b[1] )

    # hidden layers
    for l in range(1,L):
        Z[l] = af[l].f( np.matmul((1/p_keep[l])*Z[l-1],W[l] ) + b[l] )

    # output layer
    Z[L] = af[L].f( np.matmul((1/p_keep[L])*Z[L-1],W[L] ) + b[L] )

    return Z[L]

#===============================================================================
# back propagation
#===============================================================================

def RMSProp_momentum_noise_dropout(X,Y,af,M,**kwargs):

    # dimentions
    N,D = X.shape
    K = Y.shape[0]
    L = len(M)

    # kwargs
    lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
    lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
    mu = kwargs['mu'] if 'mu' in kwargs else 0.9
    gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.999
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 100
    # variables from kwargs
    alpha = 1-gamma
    batches = N//batch_size
    epochs = int(epochs)

    # set up dZ,H,dH,dW,db
    dZ,dH,dW = {},{},{}
    H,dHbar,dg,db = {},{},{},{}
    vW,vb,GW,Gb = {},{},{},{}
    for l in range(1,L+1):
        vW[l] = 0
        vb[l] = 0
        GW[l] = 0
        Gb[l] = 0

    # get Weights and bias
    kwargs['Gamma'] = True
    W,b,Gamma = assign_random_weights_and_biases(X,Y,M,**kwargs)

    # set up cost function, mu, and covariance
    J = np.zeros(epochs*batches)
    mu_train = np.zeros((epochs,batches))
    covariance_train = np.zeros_like(mu)

    #=======================================================================
    # RMSProp with Nesterov momentum - gradient descent
    #=======================================================================

    for epoch in tqdm(range(epochs)):
        X,Y = shuffle(X,Y)
        for batch in tqdm(range(batches)):

            # get the batch data
            X_b = X[(batch*batch_size):(batch+1)*batch_size]
            Y_b = Y[(batch*batch_size):(batch+1)*batch_size]
            N_batch = X_b.shape[0]
            mu_batch = X_b.sum(axis=0) / N_batch
            covariance_batch = ((X_b - mu_batch)**2).sum(axis=0) / N_batch

            # feed forward
            Z = feed_forward_train(X_b,W,b,af,M,**kwargs)

            # start with output layer
            # vanilla
            H[L] = Z[L-1].dot(W[L])
            dHbar[L] = Z[L] - Y_b
            dg[L] = (dHbar[L] * (H[L]-mu_batch)/np.sqrt(covariance_batch + epsilon)).sum(axis=0)
            db[L] = dHbar[L].sum(axis=0)
            dH[L] = dHbar[L] * Gamma[L]/np.sqrt(covariance_batch + epsilon)
            dW[L] = (np.matmul( Z[L-1].T , dH[L] ) + lambda1*np.sign(W[L]) + lambda2*W[L] ) / N
            # RMSProp with Nestorov momentum
            GW[L] = gamma*GW[L] + (1-gamma)*dW[L]**2
            Gb[L] = gamma*Gb[L] + (1-gamma)*db[L]**2
            vW[L] = mu*vW[L] - (eta/np.sqrt(GW[L] + epsilon))*dW[L]
            vb[L] = mu*vb[L] - (eta/np.sqrt(Gb[L] + epsilon))*db[L]
            # update
            W[L] += vW[L]
            b[L] += vb[L]

            # now work back through each layer till input layer
            for l in np.arange(2,L)[::-1]:
                # vanilla
                dZ[l] = np.matmul( dH[l+1] , W[l+1].T )
                H[l] = Z[l-1].dot(W[l])
                dHbar[l] = dZ[l] * af[l].df(Z[l])
                dg[l] = (dHbar[l] * (H[l]-mu_batch)/np.sqrt(covariance_batch + epsilon)).sum(axis=0)
                db[l] = dHbar[l].sum(axis=0)
                dH[l] = dHbar[l] * Gamma[l]/np.sqrt(covariance_batch + epsilon)
                dW[l] = (np.matmul( Z[l-1].T , dH[l] ) + lambda1*np.sign(W[l]) + lambda2*W[l]) / N
                db[l] = dH[l].sum(axis=0) / N
                # RMSProp with Nesterov momentum
                GW[l] = gamma*GW[l] + (1-gamma)*dW[l]**2
                Gb[l] = gamma*Gb[l] + (1-gamma)*db[l]**2
                vW[l] = mu*vW[l] - (eta/np.sqrt(GW[l] + epsilon))*dW[l]
                vb[l] = mu*vb[l] - (eta/np.sqrt(Gb[l] + epsilon))*db[l]
                # update
                W[l] += vW[l]
                b[l] += vb[l]

            # end with input layer
            # vanilla
            dZ[1] = np.matmul( dH[2] , W[2].T )
            H[1] = X_b.dot(W[1])
            dHbar[1] = dZ[1] * af[1].df(Z[1])
            dg[1] = (dHbar[1] * (H[1]-mu_batch)/np.sqrt(covariance_batch + epsilon)).sum(axis=0)
            db[1] = dHbar[1].sum(axis=0)
            dH[1] = dHbar[1] * Gamma[1]/np.sqrt(covariance_batch + epsilon)
            dW[1] = (np.matmul( X_b.T , dH[1] ) + lambda1*np.sign(W[1]) + lambda2*W[1]) / N
            db[1] = dH[1].sum(axis=0) / N
            # RMSProp with Nesterov momentum
            GW[1] = gamma*GW[1] + (1-gamma)*dW[1]**2
            Gb[1] = gamma*Gb[1] + (1-gamma)*db[1]**2
            vW[1] = mu*vW[1] - (eta/np.sqrt(GW[1] + epsilon))*dW[1]
            vb[1] = mu*vb[1] - (eta/np.sqrt(Gb[1] + epsilon))*db[1]
            # update
            W[1] += vW[1]
            b[1] += vb[1]

            # get training batch objective function
            index = batch + (epoch*batches)
            Z = feed_forward_train(X,W,b,af,M,**kwargs)
            J[index] = (cross_entropy(Y,Z[L-1]) + lambda1*L1(W) + lambda2*L2(W)) / N
            mu_train[epoch,batch] = mu_batch
            covariance_train[epoch,batch] = covariance_batch

    # collect results
    results = {
        'J'                 :   J,
        'lambda1'           : lambda1,
        'lambda2'           : lambda2,
        'mu'                :   mu,
        'gamma'             : gamma,
        'epsilon'           : epsilon,
        'eta'               : eta,
        'epochs'            : epochs,
        'batch_size'        : batch_size,
        'W'                 :   W,
        'b'                 :   b,
        'mu_batch'          :   mu_train,
        'covariance_batch'  :   covariance_train
        }

    # output
    return results

#===============================================================================
# set up a default model with options
#===============================================================================

def run(**kwargs):

    # kwargs
    af = kwargs['af'] if 'af' in kwargs else {1:ReLU(), 2:ReLU(), 3:softmax()}
    M = kwargs['M'] if 'M' in kwargs else {1:20, 2:10}
    if not 'p_keep' in kwargs: kwargs['p_keep'] = {1:3/4, 2:4/5, 3:1}

    # train
    Y = one_hot_encode(y)
    results = RMSProp_momentum_noise_dropout(X,Y,af,M,**kwargs)

    # plot figure
    fig,ax=plt.subplots()
    fig.suptitle("RMSProp with Nesterov Momentum, Noise Injection, and Drop-out")
    ax.plot(results['J'], color='b')
    ax.set_xlabel("iteration")
    ax.set_ylabel("J")

    # save figure
    fig.savefig("J_Noise_dropout.pdf")
    plt.close(fig)
