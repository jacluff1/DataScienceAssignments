# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pdb
# import from my DataScience library
import DataScience.ActivatingFunctions as AF
import DataScience.ANN as ANN

#===============================================================================
# get data
#===============================================================================

def pickle_data_sets():

    #===========================================================================
    # load up data
    #===========================================================================

    # load data
    data = pd.read_csv("usps_digit_recognizer.csv")

    # extract the label column
    y = data['label'].values
    data.drop(columns=['label'], inplace=True)
    X = data.values
    N = X.shape[0]
    K = 10

    #===========================================================================
    # design matrix & target matrix
    #===========================================================================

    # construct design matrix
    PHI = X/255

    # one-hot Y
    Y = np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]] = 1

    #===========================================================================
    # shuffle data
    #===========================================================================

    # creat shuffle mask
    np.random.seed(0)
    shuffle = np.arange(N)
    np.random.shuffle(shuffle)

    # shuffle PHI and Y
    PHI,Y = PHI[shuffle],Y[shuffle]

    # get the numbers of observations for the data sets
    N_train = int(.6*N)
    N_validate = int(.2*N)
    N_test = N - N_validate - N_train

    # get the cross validation design matrices
    PHI_train = PHI[ :N_train ]
    PHI_validate = PHI[N_train : N_train+N_validate ]
    PHI_test = PHI[ N_train+N_validate: ]

    # get the cross validation target arrays
    Y_train = Y[ :N_train ]
    Y_validate = Y[ N_train : N_train+N_validate ]
    Y_test = Y[ N_train+N_validate: ]

    # get panda.Series of sets
    train = pd.Series( dict(PHI=PHI_train, Y=Y_train) )
    validate = pd.Series( dict(PHI=PHI_validate, Y=Y_validate) )
    test = pd.Series( dict(PHI=PHI_test, Y=Y_test) )

    # send Series to pickle
    print("\npickling train, validation and test sets...")
    train.to_pickle("./train.pkl")
    validate.to_pickle("./validate.pkl")
    test.to_pickle("./test.pkl")

def read_pickle(set):
    filename = f"./{set}.pkl"
    print(f"\nreading {set} pickle...")
    return pd.read_pickle(filename)

#===============================================================================
# set up model
#===============================================================================

# set up activation functions
af_ReLU_5 = {
    1: AF.ReLU(),
    2: AF.ReLU(),
    3: AF.ReLU(),
    4: AF.ReLU(),
    5: AF.ReLU(),
    6: AF.softmax()
    }

# set up number of nodes for each layer
M_5 = {
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 8
    }

# number of layers
L = len(af)

#===============================================================================
# basic functions
#===============================================================================

def feed_forward(X,W,b,af):

    # collect Z
    Z = {}
    Z[1] = af[1].f( np.matmul(X,W[1]) + b[1] )
    for l in range(2,L+1):
        Z[l] = af[l].f( np.matmul(Z[l-1],W[l]) + b[l] )

    return Z

def back_propagation(X,Y,af,M,l1,l2,**kwargs):

    # kwargs
    save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    epochs = int(epochs)

    # input dimentions
    if Y.shape[0] == Y.size:
        K = np.unique(Y).shape[0]
    else:
        K = Y.shape[1]
    N,D = X.shape

    # set random W and b of appropriate shapes
    W = {}
    b = {}
    W[1] = np.random.randn(D,M[1])
    b[1] = np.random.randn(M[1])
    for l in M:
        W[l] = np.random.randn(M[l-1],M[l])
        b[l] = np.random.randn(M[l])
    W[L] = np.random.randn(M[l-1],K)
    b[L] = np.random.randn(K)

    # set up back propagation
    J = np.zeros(epochs)

    for epoch in range(epochs):

        # feed forward
        Z = feed_forward(X,W,b)

        # cross entropy
        J[epoch] = ANN.cross_entropy(Y,Z[L])

        # set up dZ, H,dH, W,dW, and b,db
        dZ = {}
        H,dH = {},{}
        W,dW = {},{}
        b,db = {},{}

        # start with output layer
        dH[L] = Z[L] - Y
        dW[L] = np.matmul(Z[L-1].T,dH[L])
        db[L] = dH[L].sum(axis=0)
        W[L] -= eta*dW[L]
        b[L] -= eta*db[L]

        # now work back through each layer till input layer
        for l in np.arange(2,L)[::-1]:
            dZ[l] = np.matmul(dH[l+1],dW[l+1].T)
            dH[l] = af[l].df(dZ[l],Z[l])
            dW[l] = np.matmul(Z[l-1].T,dH[l])
            db[l] = dH[l].sum(axis=0)
            W[l] -= eta*dW[l]
            b[l] -= eta*db[l]

        # end with input layer
        dZ[1] = np.matmul(dH[2],W[2].T)
        dH[1] = af[l].df(dZ[1],Z[1])
        dW[1] = np.matmul(X.T,dH[1])
        db[1] = dH[1].sum(axis=0)
        W[1] -= eta*dW[1]
        b[1] -= eta*db[1]

    # save figure
    if save_plot:
        fig,ax = plt.figure()
        fig.suptitle(f"$\\eta$: {eta}, epochs: {epochs}")
        ax.plot(np.arange(epochs),J)
        ax.set_xlabel(epoch)
        ax.set_ylabel(J)
        if not os.path.isdir("J"): os.mkdir("J")
        fig.savefig(f"J/J_eta_{eta}_epochs_{epochs}.pdf")
        plt.close(fig)

    # collect results
    results = {
        'W'     : W, # weights
        'b'     : b, # bias
        'P_hat' : Z[L] # output predictions
        }

    # output results
    return results

#===============================================================================
# cross validation
#===============================================================================

def train_validate_test(train,validate,test,af,M,l1,l2,**kwargs):

    # kwargs
    print_acc = kwargs['print_acc'] if 'print_acc' in kwargs else False
    pickle = kwargs['pickle'] if 'pickle' in kwargs else True
    output = kwargs['output'] if 'output' in kwargs else True

    print(f"\nperforming train-validate-test for lambda: {l1},{l2}")

    # training results
    tr = back_propagation(train['PHI'],train['Y'],af,M,l1,l2,**kwargs)
    P_hat_train = tr['P_hat']
    acc_train = ANN.accuracy(train['Y'],P_hat_train)
    W = tr['W']
    b = tr['b']

    # validate results
    Z_validate = feed_forward(validate['PHI'],W,b,af)
    P_hat_validate = Z_validate[len(af)]
    acc_validate = ANN.accuracy(validate['Y'],P_hat_validate)

    # test results
    Z_test = feed_forward(test['PHI'],W,b,af)
    P_hat_test = Z_test[len(af)]
    acc_validate = ANN.accuracy(test['Y'],P_hat_test)

    # collect results
    results = {
        'W'         : W,
        'b'         : b,
        'P_hat'     : P_hat_test,
        'accuracy'  : {
            'train'     :acc_train,
            'validate'  :acc_validate,
            'test'      :acc_validate
            }
        }

    # print results
    if print_acc:
        print("\nAccuracy:\ntrain: {:0.4f}\nvalidate: {:0.4f}\ntest: {:0.4f}".format(acc_train, acc_validate, acc_test))

    # save results
    if pickle: pd.Series(results).to_pickle('./train_results_{:0.4f}'.format(acc))

    # output
    if output: return results

#===============================================================================
# figures
#===============================================================================

def plot_handwritten_pictures(nrows=4, ncols=6, cmap='inferno'):

    # get total number of subplots
    N = int(nrows*ncols)

    # load data
    data = pd.read_csv("usps_digit_recognizer.csv")
    data = data[:N]

    # make figure
    fig,ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle("Hand-Written Numbers", fontsize=25)

    # loop over entire number of subplots
    for n in range(N):

        # get the image data
        X = data.iloc[n].values[1:].reshape([28,28])
        # num = data.iloc[n].values[0]

        # find the axis indicies
        i,j = divmod(n,ncols)
        ax[i,j].imshow(X, cmap=cmap)
        # ax[i,j].set_title(num)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

    # clean up figure
    plt.tight_layout()
    plt.subplots_adjust(top=.85)

    # save figure
    filename = "Number_Examples.pdf"
    fig.savefig(filename)
    print(f"\nsaved {filename}")
    plt.close(fig)

#===============================================================================
# run
#===============================================================================

def run(L1,L2,af,M,**kwargs):

    # kwargs
    overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else False

    # check that pickles exist
    if any([ not os.path.isfile('train.pkl') , overwrite ]) : pickle_data_sets()

    # load data
    train = read_pickle('train')
    validate = read_pickle('validate')
    test = read_pickle('test')

    # run cross validation
    results = train_validate_test(train,validate,test,L1,L2,af,M,**kwargs)

    # flatten out Y and Y_hat
    Y_hat = results['Y_hat']
    Y_hat = ANN.collapse_Y(Y_hat)
    Y = ANN.collapse_Y(test['Y'])

    # update results
    results['Y'] = Y
    results['Y_hat'] = Y_hat
    results['CM'] = confusion_matrix(Y,Y_hat)

    # send results to pickle
    pd.Series(results).to_pickle("./results_{:0.4}.pkl".format(results['validate']))

    # output results
    return results
