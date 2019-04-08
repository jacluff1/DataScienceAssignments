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
    Y = ANN.one_hot_encode(y)

    # construct design matrix
    PHI = X/255
    # PHI = np.column_stack((np.ones((N,1)), X/255))

    #===========================================================================
    # shuffle data
    #===========================================================================

    # shuffle PHI and Y
    PHI,Y = ANN.shuffle(PHI,Y)

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
# basic functions
#===============================================================================

M1 = 100
# M2 = 300
# M3 = 8
# M4 = 8
# M5 = 8

def ReLU(H):
    return H*(H > 0)

def tanh(H,**kwargs):
    eH = np.exp(H)
    eHn = np.exp(-H)
    return (eH - eHn) / (eH + eHn)

def softmax(H):
    eH = np.exp(H)
    assert np.all(np.isfinite(eH)), "NOOOOO! try decreasing learning rate."
    return eH/eH.sum(axis=1, keepdims=True)

def feed_forward(X,W1,b1,W2,b2):

    Z1 = tanh(np.matmul(X,W1) + b1)
    # Z2 = tanh(np.matmul(Z1,W2) + b2)
    P_hat = softmax(np.matmul(Z1,W2) + b2)

    return Z1, P_hat

# def feed_forward(X,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6):
#
#     Z1 = ReLU(np.matmul(X,W1) + b1)
#     Z2 = ReLU(np.matmul(Z1,W2) + b2)
#     Z3 = ReLU(np.matmul(Z2,W3) + b3)
#     Z4 = ReLU(np.matmul(Z3,W4) + b4)
#     Z5 = ReLU(np.matmul(Z4,W5) + b5)
#     P_hat = softmax(np.matmul(Z5,W6) + b6)
#
#     return Z1, Z2, Z3, Z4, Z5, P_hat

def back_propagation(train,**kwargs):

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
    W1 = np.random.randn(D,M1)
    b1 = np.random.randn(M1)
    # W2 = np.random.randn(M1,M2)
    # b2 = np.random.randn(M2)
    # W3 = np.random.randn(M2,M3)
    # b3 = np.random.randn(M3)
    # W4 = np.random.randn(M3,M4)
    # b4 = np.random.randn(M4)
    # W5 = np.random.randn(M4,M5)
    # b5 = np.random.randn(M5)
    # W6 = np.random.randn(M5,K)
    # b6 = np.random.randn(K)
    W2 = np.random.randn(M1,K)
    b2 = np.random.randn(K)

    # set up back propagation
    batches = N//batch_size
    J_train = np.zeros(epochs*batches)

    for epoch in range(epochs):

        X,Y = ANN.shuffle(train['PHI'],train['Y'])

        for batch in range(batches):

            # start timer
            t0 = datetime.now()

            # get the batch data
            X_b = X[(batch*batch_size):(batch+1)*batch_size]
            Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

            # feed forward
            Z1, P_hat = feed_forward(X_b,W1,b1,W2,b2)
            # Z1, Z2, Z3, Z4, Z5, P_hat = feed_forward(X_b,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6)

            dH2 = P_hat-Y_b
            dW2 = np.matmul(Z1.T,dH2)
            db2 = dH2.sum(axis=0)
            W2 -= eta*dW2
            b2 -= eta*db2

            # dZ2 = np.matmul(dH3,W3.T)
            # dH2 = dZ2*(1 - Z2**2)
            # dW2 = np.matmul(Z1.T,dH2)
            # db2 = dH2.sum(axis=0)
            # W2 -= eta*dW2
            # b2 -= eta*db2

            dZ1 = np.matmul(dH2,W2.T)
            dH1 = dZ1*(1 - Z1**2)
            dW1 = np.matmul(X_b.T,dH1)
            db1 = dH1.sum(axis=0)
            W1 -= eta*dW1
            b1 -= eta*db1

            # dH6 = P_hat-Y_b
            # dW6 = np.matmul(Z5.T,dH6)
            # db6 = dH6.sum(axis=0)
            # W6 -= eta*dW6
            # b6 -= eta*db6
            #
            # dZ5 = np.matmul(dH6,W6.T)
            # dH5 = dZ5*(Z5 > 0)
            # dW5 = np.matmul(Z4.T,dH5)
            # db5 = dH5.sum(axis=0)
            # W5 -= eta*dW5
            # b5 -= eta*db5
            #
            # dZ4 = np.matmul(dH5,W5.T)
            # dH4 = dZ4*(Z4 > 0)
            # dW4 = np.matmul(Z3.T,dH4)
            # db4 = dH4.sum(axis=0)
            # W4 -= eta*dW4
            # b4 -= eta*db4
            #
            # dZ3 = np.matmul(dH4,W4.T)
            # dH3 = dZ3*(Z3 > 0)
            # dW3 = np.matmul(Z2.T,dH3)
            # db3 = dH3.sum(axis=0)
            # W3 -= eta*dW3
            # b3 -= eta*db3
            #
            # dZ2 = np.matmul(dH3,W3.T)
            # dH2 = dZ2*(Z2 > 0)
            # dW2 = np.matmul(Z1.T,dH2)
            # db2 = dH2.sum(axis=0)
            # W2 -= eta*dW2
            # b2 -= eta*db2
            #
            # dZ1 = np.matmul(dH2,W2.T)
            # dH1 = dZ1*(Z1 > 0)
            # dW1 = np.matmul(X_b.T,dH1)
            # db1 = dH1.sum(axis=0)
            # W1 -= eta*dW1
            # b1 -= eta*db1

            # feed forward for whole train and validation sets
            Z1, P_hat = feed_forward(train['PHI'],W1,b1,W2,b2)
            # Z1, Z2, Z3, Z4, Z5, P_hat = feed_forward(train['PHI'],W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6)

            # update train and validation cost functions
            index = batch + (epoch*batches)
            J_train[index] = ANN.cross_entropy(train['Y'],P_hat) / N

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
        ax.set_xlabel("batch + (epochs x baches)")
        ax.set_ylabel("J")
        ax.legend(loc='best')
        if not os.path.isdir("J"): os.mkdir("J")
        savename = f"J/J_eta_{eta}_epochs_{epochs}.pdf"
        fig.savefig(savename)
        plt.close(fig)
        print(f"saved {savename}")

    # # collect results
    # results = {
    #     'W'     : (W1,W2,W3,W4,W5,W6), # weights
    #     'b'     : (b1,b2,b3,b4,b5,b6) # bias
    #     }

    print("Accuracy: {:0.4f}".format(ANN.accuracy(train['Y'],P_hat)))

    # # output results
    # return results

# #===============================================================================
# # cross validation
# #===============================================================================
#
# def train_validate_test(train,validate,test,af,M,l1,l2,**kwargs):
#
#     # print message
#     print(f"\nperforming train-validate-test for lambda: {l1},{l2}")
#
#     # training results
#     bp = back_propagation(train,validate,af,M,l1,l2,**kwargs)
#     W = bp['W']
#     b = bp['b']
#     Z_train = feed_forward(train['PHI'],W,b,af)
#     P_hat_train = Z_train[len(af)]
#     acc_train = ANN.accuracy(train['Y'],P_hat_train)
#
#     # validate results
#     Z_validate = feed_forward(validate['PHI'],W,b,af)
#     P_hat_validate = Z_validate[len(af)]
#     acc_validate = ANN.accuracy(validate['Y'],P_hat_validate)
#
#     # test results
#     Z_test = feed_forward(test['PHI'],W,b,af)
#     P_hat_test = Z_test[len(af)]
#     acc_test = ANN.accuracy(test['Y'],P_hat_test)
#
#     # collect results
#     results = {
#         'W'         : W,
#         'b'         : b,
#         'P_hat'     : P_hat_test,
#         'accuracy'  : {
#             'train'     :acc_train,
#             'validate'  :acc_validate,
#             'test'      :acc_test
#             }
#         }
#
#     return results
#
# #===============================================================================
# # figures
# #===============================================================================
#
# def plot_handwritten_pictures(nrows=4, ncols=6, cmap='inferno'):
#
#     # get total number of subplots
#     N = int(nrows*ncols)
#
#     # load data
#     # data = pd.read_csv("usps_digit_recognizer.csv")
#     data = pd.read_csv("mnist_train.csv")
#     data = data[:N]
#
#     # make figure
#     fig,ax = plt.subplots(nrows=nrows, ncols=ncols)
#     fig.suptitle("Hand-Written Numbers", fontsize=25)
#
#     # loop over entire number of subplots
#     for n in range(N):
#
#         # get the image data
#         X = data.iloc[n].values[1:].reshape([28,28])
#         # num = data.iloc[n].values[0]
#
#         # find the axis indicies
#         i,j = divmod(n,ncols)
#         ax[i,j].imshow(X, cmap=cmap)
#         # ax[i,j].set_title(num)
#         ax[i,j].set_xticks([])
#         ax[i,j].set_yticks([])
#
#     # clean up figure
#     plt.tight_layout()
#     plt.subplots_adjust(top=.85)
#
#     # save figure
#     filename = "Number_Examples.pdf"
#     fig.savefig(filename)
#     print(f"\nsaved {filename}")
#     plt.close(fig)
#
# #===============================================================================
# # run
# #===============================================================================
#
# def run(af,M,L1,L2,**kwargs):
#
#     # kwargs
#     overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else False
#
#     #===========================================================================
#     # set up data
#     #===========================================================================
#
#     # check that pickles exist
#     if any([ not os.path.isfile('train.pkl') , overwrite ]) : pickle_data_sets()
#
#     # load data
#     train = read_pickle('train')
#     validate = read_pickle('validate')
#     test = read_pickle('test')
#
#     #===========================================================================
#     # run cross validation
#     #===========================================================================
#
#     # run cross validation
#     results = train_validate_test(train,validate,test,af,M,L1,L2,**kwargs)
#
#     # update results
#     try:
#         results['CM'] = ANN.confusion_matrix(test['Y'],results['P_hat'])
#     except:
#         print("something went wrong with the confusion matrix")
#     results['M'] = M
#     results['af'] = af
#     results['L'] = L
#     results['lambda1'] = L1
#     results['lambda2'] = L2
#     results['eta'] = kwargs['eta'] if 'eta' in kwargs else 1e-3
#     results['epochs'] = kwargs['epochs'] if 'epochs' in kwargs else 1e3
#     results['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 30
#
#     #===========================================================================
#     # send results to pickle
#     #===========================================================================
#
#     # validation accuracy from current run
#     acc = results['accuracy']['validate']
#
#     # find a list of previously pickled best results
#     DIR = np.array(os.listdir())
#     filter = ['results_' in x for x in DIR]
#     DIR = DIR[filter]
#
#     # if there are past results, save current results only if they are better than any previous results
#     if len(DIR) > 0:
#         best = np.array([ x[ x.find("_")+1 : x.find(".pkl") ] for x in DIR ], dtype=np.float32).max()
#         if acc > best:
#             print("\nfound new best result!")
#             pd.Series(results).to_pickle("results_{:0.4}.pkl".format(acc))
#         else:
#             print("\nno such luck...")
#     # if there are no results, just save the current results
#     else:
#         print("\nfound new best result!")
#         pd.Series(results).to_pickle("results_{:0.4}.pkl".format(acc))
#
#     print("Accuracy from this round: {:0.4f}".format(acc))
#
#     # output results
#     return results
