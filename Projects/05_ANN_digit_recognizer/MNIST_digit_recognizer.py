# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import from my DataScience library
import DataScience.ActivationFunctions as AF
import DataScience.ANN as ANN

#===============================================================================
# get data
#===============================================================================

def get_PHI_Y():

    #===========================================================================
    # load kaggle data
    #===========================================================================

    # data = pd.read_csv("usps_digit_recognizer.csv")

    #===========================================================================
    # load up data https://pjreddie.com/projects/mnist-in-csv/
    #===========================================================================

    # load training and test data
    names = ['label'] + [f"pixel_{x}" for x in range(784)]
    train = pd.read_csv("mnist_train.csv", names=names)
    test = pd.read_csv("mnist_test.csv", names=names)

    # merge sets
    data = train.append(test, ignore_index=True)

    #===========================================================================
    # design matrix & target matrix
    #===========================================================================

    # extract the label column
    y = data['label'].to_numpy(dtype=np.int32)
    data.drop(columns=['label'], inplace=True)

    # # one-hot Y
    # Y = ANN.one_hot_encode(y)

    # construct design matrix
    PHI = data.to_numpy(np.float32) / 255

    # output
    return PHI,y

def read_pickle(set):
    filename = f"{set}.pkl"
    return pd.read_pickle(filename)

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

def run_my_class(*args,**kwargs):

    #===========================================================================
    # set up model parameters
    #===========================================================================

    # activation functions
    af = {
        1   :   AF.tanh(),
        2   :   AF.tanh(),
        4   :   AF.softmax()
        }

    # numbers of nodes in hidden layers
    M = {
        1   :   300,
        2   :   200
        }

    #===========================================================================
    # instantiate ANN class
    #===========================================================================

    ann = ANN.ArtificalNeuralNet(af,M)

    #===========================================================================
    # run cross validation
    #===========================================================================

    PHI,y = get_PHI_Y()
    Y = ann._one_hot_encode(y)

    results = ann.train_validate_test(PHI,Y,**kwargs)

def run_sklearn():

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(300,200), random_state=1, activation='tanh')

    PHI,y = get_PHI_Y()

    ann = ANN.ArtificalNeuralNet({1:None, 2:None},{1:None})
    train,validate,test = ann._cv_tvt_data_sets(PHI,y, normalize=False, assign=False, output=True)

    # collect accuracy
    accuracy = {}

    # train
    clf.fit(train['PHI'],train['Y'])
    y_hat_train = clf.predict(train['PHI'])
    accuracy['train'] = np.mean(train['Y'] == y_hat_train)

    # validate
    y_hat_validate = clf.predict(validate['PHI'])
    accuracy['validate'] = np.mean(validate['Y'] == y_hat_validate)

    # test
    y_hat_test = clf.predict(test['PHI'])
    accuracy['test'] = np.mean(test['Y'] == y_hat_test)
    Y = ann._one_hot_encode(test['Y'])
    Y_hat = ann._one_hot_encode(y_hat_test)
    CM = np.matmul(Y.T,Y_hat)

    # collect results
    results = {
        'accuracy'  :   accuracy,
        'train'     :   train,
        'validate'  :   validate,
        'test'      :   test,
        'y_hat'     :   y_hat_test,
        'CM'        :   CM,
        'sklearn'   :   clf
        }

    pd.Series(results).to_pickle("sklearn_assist_{:0.4f}.pkl".format(accuracy['validate']))

    return results
