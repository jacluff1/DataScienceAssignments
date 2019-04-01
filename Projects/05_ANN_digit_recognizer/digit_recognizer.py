# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import pdb
# import from my DataScience library
import DataScience.ActivationFunctions as AF
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
# cross validation
#===============================================================================

def train_validate_test(train,validate,test,L1,L2,afs,M,**kwargs):

    # instantiate some variables to keep track of best results
    bestAc = dict(train=0, validate=0, test=0)
    bestResults = None

    # cast a net and find best outcome
    for l1 in L1:
        for l2 in L2:
            for af in afs.values():
                for m in M.values():

                    # print iteration to track progress
                    print(f"\nworking on {l1},{l2}...")

                    # update kwargs
                    kwargs['lambda1'] = l1
                    kwargs['lambda2'] = l2

                    # instantiate ArtificalNeuralNet instance
                    ann = ANN.ArtificalNeuralNet(test['PHI'],test['Y'],af,m)

                    # solve and run cross validation test for this instance
                    results = ann.train_validate_test(train,validate,test,**kwargs)

                    # print iteration results
                    print(f"train accuracy: {results['train']}, validate accuracy {results['validate']}")

                    # update best values
                    if results['validate'] > bestAc['validate']:
                        bestAc['validate'] = results['validate']
                        bestResults = results
                        print(f"best value (so far)!")

    # collect results
    results = pd.Series( bestResults )

    # send results to pickle
    pd.Series(results).to_pickle(f"./results_{np.round(bestAc['validate'],4)}.pkl")

    # ouput
    return results

#===============================================================================
# check model
#===============================================================================

def confusion_matrix(Y,Y_hat):
    return np.matmul(Y.T,Y_hat)

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

def run(L1,L2,overwrite=False,**kwargs):

    # check that pickles exist
    if any([ not os.path.isfile('train.pkl') , overwrite ]) : pickle_data_sets()

    # load data
    train = read_pickle('train')
    validate = read_pickle('validate')
    test = read_pickle('test')
    # pdb.set_trace()

    # get dictionaries for activating functions and M
    afs = {
        1: {1:AF.ReLU(), 2:AF.ReLU(), 3:AF.ReLU(), 4:AF.ReLU(), 5:AF.ReLU(), 6:AF.ReLU(), 7:AF.ReLU(), 8:AF.ReLU(), 9:AF.ReLU(), 10:AF.softmax()}
        }
    Ms = {
        1: {1:4, 2:4, 3:4, 4:4, 5:4, 6:4, 7:4, 8:4, 9:4}
        }

    # run cross validation
    results = train_validate_test(train,validate,test,L1,L2,afs,Ms,**kwargs)

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
