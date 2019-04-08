# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pdb
# import from my DataScience library
import DataScience.ActivationFunctions as AF
import DataScience.ANN as ANN

#===============================================================================
# get data
#===============================================================================

def pickle_data_sets():

    #===========================================================================
    # load kaggle data
    #===========================================================================

    data = pd.read_csv("usps_digit_recognizer.csv")

    # #===========================================================================
    # # load up data https://pjreddie.com/projects/mnist-in-csv/
    # #===========================================================================
    #
    # # load training and test data
    # names = ['label'] + [f"pixel_{x}" for x in range(784)]
    # train = pd.read_csv("mnist_train.csv", names=names)
    # test = pd.read_csv("mnist_test.csv", names=names)
    #
    # # merge sets
    # data = train.append(test, ignore_index=True)

    #===========================================================================
    # design matrix & target matrix
    #===========================================================================

    # extract the label column
    y = data['label'].values
    data.drop(columns=['label'], inplace=True)
    X = data.values
    N,D = X.shape
    K = 10

    # one-hot Y
    Y = np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]] = 1

    # construct design matrix
    PHI = X/255
    # PHI = np.column_stack((np.ones((N,1)), X/255))

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
    print("\npickle-ing train, validation and test sets...")
    train.to_pickle("train.pkl")
    validate.to_pickle("validate.pkl")
    test.to_pickle("test.pkl")

def read_pickle(set):
    filename = f"{set}.pkl"
    return pd.read_pickle(filename)

#===============================================================================
# set up model
#===============================================================================

# set up activation functions
af = {
    1 : AF.tanh(),
    2 : AF.tanh(),
    3 : AF.softmax()
    }

# set up number of nodes for each layer
M = {
    1 : 500,
    2 : 300
    }

# number of layers
L = len(af)

#===============================================================================
# basic functions
#===============================================================================

def shuffle(*args):
    idx = np.random.permutation(len(args[0]))
    return [X[idx] for X in args]

def feed_forward(X,W,b,af):

    # collect Z
    Z = {}
    Z[1] = af[1].f( np.matmul(X,W[1]) + b[1] )
    for l in range(2,L+1):
        Z[l] = af[l].f( np.matmul(Z[l-1],W[l]) + b[l] )

    return Z

def back_propagation(train,validate,af,M,l1,l2,**kwargs):

    # input dimentions
    N,D = train['PHI'].shape
    if train['Y'].shape[0] == train['Y'].size:
        K = len(set(train['Y']))
    else:
        K = train['Y'].shape[1]

    # kwargs
    save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    epochs = int(epochs)
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else N

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

    for epoch in range(epochs):

        X,Y = shuffle(train['PHI'],train['Y'])

        for batch in range(batches):

            # start timer
            t0 = datetime.now()

            # get the batch data
            X_b = X[(batch*batch_size):(batch+1)*batch_size]
            Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

            # feed forward
            Z = feed_forward(X_b,W,b,af)

            # set up dZ,H,dH,dW, and db
            dZ,H,dH,dW,db = {},{},{},{},{}

            # start with output layer
            dH[L] = Z[L] - Y_b
            dW[L] = np.matmul(Z[L-1].T,dH[L])
            db[L] = dH[L].sum(axis=0)
            W[L] -= eta*dW[L] / N
            b[L] -= eta*db[L]

            # now work back through each layer till input layer
            for l in np.arange(2,L)[::-1]:
                dZ[l] = np.matmul(dH[l+1],W[l+1].T)
                dH[l] = dZ[l] * af[l].df(Z[l])
                dW[l] = np.matmul(Z[l-1].T,dH[l])
                db[l] = dH[l].sum(axis=0)
                W[l] -= eta*dW[l] / N
                b[l] -= eta*db[l]

            # end with input layer
            dZ[1] = np.matmul(dH[2],W[2].T)
            dH[1] = dZ[1] * af[1].df(Z[1])
            dW[1] = np.matmul(X_b.T,dH[1])
            db[1] = dH[1].sum(axis=0)
            W[1] -= eta*dW[1] / N
            b[1] -= eta*db[1]

            # feed forward for whole train and validation sets
            Z_train = feed_forward(train['PHI'],W,b,af)
            Z_validate = feed_forward(validate['PHI'],W,b,af)

            # update train and validation cost functions
            index = batch + (epoch*batches)
            J_train[index] = ANN.cross_entropy(train['Y'],Z_train[L]) / N
            J_validate[index] = ANN.cross_entropy(validate['Y'],Z_validate[L]) / validate['Y'].shape[0]

            #===================================================================
            # approximate time left till training is done
            #===================================================================

            # find batch time
            tf = (datetime.now() - t0).seconds

            # find number of sub-epochs left
            epochs_left = J_train.shape[0] - index - 1

            # time left till training done (minutes)
            time_left = tf * epochs_left / 60

            print("Approximately {:0.2f} minutes left.".format(time_left))

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
        }

    # output results
    return results

#===============================================================================
# cross validation
#===============================================================================

def train_validate_test(train,validate,test,af,M,l1,l2,**kwargs):

    # print message
    print(f"\nperforming train-validate-test for lambda: {l1},{l2}")

    # training results
    bp = back_propagation(train,validate,af,M,l1,l2,**kwargs)
    W = bp['W']
    b = bp['b']
    Z_train = feed_forward(train['PHI'],W,b,af)
    P_hat_train = Z_train[len(af)]
    acc_train = ANN.accuracy(train['Y'],P_hat_train)

    # validate results
    Z_validate = feed_forward(validate['PHI'],W,b,af)
    P_hat_validate = Z_validate[len(af)]
    acc_validate = ANN.accuracy(validate['Y'],P_hat_validate)

    # test results
    Z_test = feed_forward(test['PHI'],W,b,af)
    P_hat_test = Z_test[len(af)]
    acc_test = ANN.accuracy(test['Y'],P_hat_test)

    # collect results
    results = {
        'W'         : W,
        'b'         : b,
        'P_hat'     : P_hat_test,
        'accuracy'  : {
            'train'     :acc_train,
            'validate'  :acc_validate,
            'test'      :acc_test
            }
        }

    return results

#===============================================================================
# figures
#===============================================================================

def plot_handwritten_pictures(nrows=4, ncols=6, cmap='inferno'):

    # get total number of subplots
    N = int(nrows*ncols)

    # load data
    # data = pd.read_csv("usps_digit_recognizer.csv")
    data = pd.read_csv("mnist_train.csv")
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

def run(af,M,L1,L2,**kwargs):

    # kwargs
    overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else False

    #===========================================================================
    # set up data
    #===========================================================================

    # check that pickles exist
    if any([ not os.path.isfile('train.pkl') , overwrite ]) : pickle_data_sets()

    # load data
    train = read_pickle('train')
    validate = read_pickle('validate')
    test = read_pickle('test')

    #===========================================================================
    # run cross validation
    #===========================================================================

    # run cross validation
    results = train_validate_test(train,validate,test,af,M,L1,L2,**kwargs)

    # update results
    try:
        results['CM'] = ANN.confusion_matrix(test['Y'],results['P_hat'])
    except:
        print("something went wrong with the confusion matrix")
    results['M'] = M
    results['af'] = af
    results['L'] = L
    results['lambda1'] = L1
    results['lambda2'] = L2
    results['eta'] = kwargs['eta'] if 'eta' in kwargs else 1e-3
    results['epochs'] = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    results['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 30

    #===========================================================================
    # send results to pickle
    #===========================================================================

    # validation accuracy from current run
    acc = results['accuracy']['validate']

    # find a list of previously pickled best results
    DIR = np.array(os.listdir())
    filter = ['results_' in x for x in DIR]
    DIR = DIR[filter]

    # if there are past results, save current results only if they are better than any previous results
    if len(DIR) > 0:
        best = np.array([ x[ x.find("_")+1 : x.find(".pkl") ] for x in DIR ], dtype=np.float32).max()
        if acc > best:
            print("\nfound new best result!")
            pd.Series(results).to_pickle("results_{:0.4}.pkl".format(acc))
        else:
            print("\nno such luck...")
    # if there are no results, just save the current results
    else:
        print("\nfound new best result!")
        pd.Series(results).to_pickle("results_{:0.4}.pkl".format(acc))

    print("Accuracy from this round: {:0.4f}".format(acc))

    # output results
    return results
