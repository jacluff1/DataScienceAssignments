# import dependencies
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import from my DataScience library @ https://github.com/jacluff1/DataScience
import DataScience.ActivationFunctions as AF
import DataScience.ANN as ANN

D = 2
K = 3
N = int(K*1.5e4)

X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0,X1,X2))

y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))


def shuffle(*args):
    idx = np.random.permutation(len(args[0]))
    return [X[idx] for X in args]

def back_propagation(train,validate,af,M,l1,l2,**kwargs):
    """
    explanation:
        Does batch gradient descent using the AdaGrad adaptive learning rate
    input:
        train:      dict - 'PHI' & 'Y'
        validate:   dict - 'PHI' & 'Y'
        af:         dict - activation function class instances for ALL layers
        M:          dict - numers of nodes for all hidden layers
        l1:         float - lambda1 for LASSO regression
        l2:         float - lambda2 for Ridge regression
        kwargs:
            save_plot:      bool - saves plot if True
            eta0:           float - initial 'basic' learning rate
            epsilon:        float - very small number so that AdaGrad won't
                            have 0 in denominator
            epochs:         int - number of epochs for training
            batch_size:     int - how many observations to feed at a time
            G0:             float/int - what to initialize the AdaGrad constant to
    output:
        dict - weights 'W' and bias 'b'
    """

    # input dimentions
    if train['Y'].shape[0] == train['Y'].size:
        pdb.set_trace()
        K = len(set(train['Y']))
    else:
        K = train['Y'].shape[1]
        N,D = train['PHI'].shape

    # kwargs
    save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
    eta0 = kwargs['eta0'] if 'eta0' in kwargs else 1e-3
    epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    epochs = int(epochs)
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else train['PHI'].shape[0]
    G0 = kwargs['G0'] if 'G0' in kwargs else 1

    # set random W and b of appropriate shapes
    W = {}
    b = {}
    W[1] = np.random.randn(D,M[1])
    b[1] = np.random.randn(M[1])
    for l in range(2,L):
        W[l] = np.random.randn(M[l-1],M[l])
        b[l] = np.random.randn(M[l])
    W[L] = np.random.randn(M[L-1],K)
    b[L] = np.random.randn(K)

    # set up back propagation
    batches = N//batch_size
    J_train = np.zeros(epochs*batches)
    J_validate = np.zeros_like(J_train)
    GW,Gb = {},{}
    for l in af:
        GW[l] = 1
        Gb[l] = 1

    for epoch in range(epochs):

        X,Y = shuffle(train['PHI'],train['Y'])

        for batch in range(batches):

            X_b = X[(batch*batch_size):(batch+1)*batch_size]
            Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

            # feed forward
            Z = feed_forward(X_b,W,b,af)

            # set up dZ,H,dH,dW, and db
            dZ,H,dH,dW,db,G = {},{},{},{},{},{}

            # start with output layer
            dH[L] = Z[L] - Y_b
            dW[L] = np.matmul(Z[L-1].T,dH[L])
            db[L] = dH[L].sum(axis=0)
            GW[L] += dW[L]**2
            Gb[L] += db[L]**2
            W[L] -= eta0/np.sqrt(GW[L]+epsilon)*dW[L] / batch_size
            b[L] -= eta0/np.sqrt(Gb[L]+epsilon)*db[L]

            # now work back through each layer till input layer
            for l in np.arange(2,L)[::-1]:
                dZ[l] = np.matmul(dH[l+1],W[l+1].T)
                dH[l] = dZ[l] * af[l].df(Z[l])
                dW[l] = np.matmul(Z[l-1].T,dH[l])
                db[l] = dH[l].sum(axis=0)
                GW[l] += dW[l]**2
                Gb[l] += db[l]**2
                W[l] -= eta0/np.sqrt(GW[l]+epsilon)*dW[l] / batch_size
                b[l] -= eta0/np.sqrt(Gb[l]+epsilon)*db[l]

            # end with input layer
            dZ[1] = np.matmul(dH[2],W[2].T)
            dH[1] = dZ[1] * af[1].df(Z[1])
            dW[1] = np.matmul(X_b.T,dH[1])
            db[1] = dH[1].sum(axis=0)
            GW[1] += dW[1]**2
            Gb[1] += db[1]**2
            W[1] -= eta0/np.sqrt(GW[1]+epsilon)*dW[1] / batch_size
            b[1] -= eta0/np.sqrt(Gb[1]+epsilon)*db[1]

            # feed forward for whole train and validation sets
            Z_train = feed_forward(train['PHI'],W,b,af)
            Z_validate = feed_forward(validate['PHI'],W,b,af)

            # update train and validation cost functions
            index = batch + (epoch*batches)
            J_train[index] = ANN.cross_entropy(train['Y'],Z_train[L]) / N
            J_validate[index] = ANN.cross_entropy(validate['Y'],Z_validate[L]) / validate['Y'].shape[0]

    # save figure
    if save_plot:
        fig,ax = plt.subplots()
        fig.suptitle(f"$\\eta$: {eta}, epochs: {epochs}, batch size: {batch_size}")
        ax.plot(J_train, label="J: Training")
        ax.plot(J_validate, label="J: Validation")
        ax.set_xlabel("batch + (epochs x baches)")
        ax.set_ylabel("J")
        ax.legend(loc='best')
        if not os.path.isdir("J"): os.mkdir("J")
        savename = f"J/J_eta_{eta}_epochs_{epochs}.pdf"
        fig.savefig(savename)
        plt.close(fig)
        print(f"saved {savename}")

    # collect results
    results = {
        'W'     : W, # weights
        'b'     : b # bias
        # 'P_hat' : Z[L] # output predictions
        }

    # output results
    return results
