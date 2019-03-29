# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pdb
# import my auxillary module
import auxillary as aux

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
    # engineer features -- sea auxillary module
    #===========================================================================

    # engineer some MF features
    E = np.column_stack((
        # how much ink does the image use?
        X.sum(axis=1),
        # find a measure of vertical symmetry
        np.array([ aux.vertical_symmetry(X[i,:]) for i in range(N) ]),
        # find a measure of horizontal symmetry
        np.array([ aux.horizonal_symmetry(X[i,:]) for i in range(N) ]),
        # find the number of empty pixels
        np.array([ aux.how_many_empty_pixels(X[i,:]) for i in range(N) ]),
        # find the sum of quadrant 1
        np.array([ aux.average_quadrant1(X[i,:]) for i in range(N) ]),
        # find the sum of quadrant 2
        np.array([ aux.average_quadrant2(X[i,:]) for i in range(N) ]),
        # find the sum of quadrant 3
        np.array([ aux.average_quadrant3(X[i,:]) for i in range(N) ]),
        # find the sum of qudrant 4
        np.array([ aux.average_quadrant4(X[i,:]) for i in range(N) ])
        ))
    # pdb.set_trace()

    # normalize the engineered features
    minimum = E.min(axis=0)
    maximum = E.max(axis=0)
    E = (E - minimum) / (maximum - minimum)

    #===========================================================================
    # design matrix & target matrix
    #===========================================================================

    # construct design matrix
    PHI = np.column_stack(( np.ones((N,1)), X/255, E ))

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
# logistic regression
#===============================================================================

def softmax(H):
    eH = np.exp(H)
    # return np.nan_to_num(eH / eH.sum(axis=1, keepdims=True), copy=False)
    return eH / eH.sum(axis=1, keepdims=True)

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def accuracy(Y,P_hat):
    return np.mean(Y.argmax(axis=1) == P_hat.argmax(axis=1))

#===============================================================================
# solve
#===============================================================================

def solve(PHI,Y,**kwargs):

    # kargs
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
    lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
    save_curve = kwargs['save_curve'] if 'save_curve' in kwargs else False

    # get dimentions
    N,P = PHI.shape
    K = Y.shape[1]

    # start with random W
    W = np.random.randn(P,K)

    # set up gradient descent
    epochs = int(epochs)
    J = np.zeros(epochs)

    # run gradient descent
    for epoch in range(epochs):
        P_hat = softmax(PHI.dot(W))
        J[epoch] = (cross_entropy(Y,P_hat) + lambda1*np.sum(np.abs(W)) + lambda2*np.sum(W**2)/2)/N
        W -= eta*(PHI.T.dot(P_hat-Y) + lambda1*np.sign(W) + lambda2*W)/N

    # plot
    if save_curve:
        filename = f"J/J_eta{eta}_epochs{epochs}_lambda1{lambda1}_lambda2{lambda2}.pdf"
        plt.figure()
        plt.plot(J)
        plt.savefig(filename)
        print(f"\nsaved {filename}")
        plt.close()

    # output
    return W

#===============================================================================
# cross validation
#===============================================================================

def train_validate_test(train,validate,test,L1,L2,**kwargs):

    # instantiate some variables to keep track of best results
    bestAc1 = 0
    bestAc2 = 0
    bestAc3 = 0
    bestl1 = 0
    bestl2 = 0
    bestEta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    bestEpochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    bestW = None
    bestThreshold = None

    # cast a net and find best outcome
    for l1 in L1:
        for l2 in L2:
            # print iteration to track progress
            print(f"\nworking on {l1},{l2}...")
            # update kwargs
            kwargs['lambda1'] = l1
            kwargs['lambda2'] = l2
            # train
            W = solve(train['PHI'],train['Y'],**kwargs)
            P_hat1 = softmax(train['PHI'].dot(W))
            ac1 = accuracy(train['Y'],P_hat1)

            # validate
            P_hat2 = softmax(validate['PHI'].dot(W))
            ac2 = accuracy(validate['Y'],P_hat2)

            # print iteration results
            print(f"train accuracy: {ac1}, validate accuracy {ac2}")

            # update best values
            if ac2 > bestAc2:
                # run the test set
                P_hat3 = softmax(test['PHI'].dot(W))
                ac3 = accuracy(test['PHI'],P_hat3)
                # update the best values
                bestAc1 = ac1
                bestAc2 = ac2
                bestAc3 = ac3
                bestl1 = l1
                bestl2 = l2
                bestW = W
                print(f"best value (so far)!")

    return dict(train_ac=bestAc1, validate_ac=bestAc2, test_ac=bestAc3, lambda1=bestl1, lambda2=bestl2, eta=bestEta, epochs=bestEpochs, W=W)

#===============================================================================
# check model
#===============================================================================

def get_test_Y_hat(test,tvt_results):
    # extract required arrays
    W = tvt_results['W']
    PHI = test['PHI']
    # get P_hat
    P_hat = softmax(PHI.dot(W))
    # get Y_hat
    return np.round(P_hat)

def confusion_matrix(Y,Y_hat):
    # Y = pd.Series(Y.argmax(axis=1), name='actual')
    # Y_hat = pd.Series(Y_hat.argmax(axis=1), name='predicted')
    # return pd.crosstab(Y,Y_hat)
    return np.matmul(Y.T,Y_hat)

#===============================================================================
# figures
#===============================================================================

def plot_handwritten_pictures(nrows=4, ncols=6, cmap='viridis'):

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







    return

#===============================================================================
# run
#===============================================================================

def run(L1,L2,overwrite=False,**kwargs):

    # check that pickles exist
    if any([ not os.path.isfile('train.pkl') , overwrite ]) : pickle_data_sets()

    # load data
    train = read_pickle('train')
    validate = read_pickle('validate')
    test = read_pickle('test')

    # run cross validation
    tvt = train_validate_test(train,validate,test,L1,L2,**kwargs)

    Y_hat = get_test_Y_hat(test,tvt)

    CM = confusion_matrix(test['Y'],Y_hat)

    tvt.update( dict(Y_hat=Y_hat, CM=CM) )

    # send results to pandas.Series
    results = pd.Series( tvt )

    # send results to pickle
    results.to_pickle("./results_{:0.4}.pkl".format(tvt['validate_ac']))

    # output results
    return results
