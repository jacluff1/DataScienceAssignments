# import dependencies
import numpy as np
import pandas as pd
import os
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import file with auxillary functions
import auxillary as aux
from classification import schema

#===============================================================================
""" get the data """
#===============================================================================

# load the data
data = pd.read_csv("mushrooms.csv")

# get the necessary dimentions -- P will be fine because we will drop target from the data and will add a bias column
N,P = data.shape

# checked that N_edible ~ N_poisonous
# convert target to binary (edible --> 1; poisonous --> 0)
data['class'] = (data['class'] == 'e').astype(np.int32)
# extract the target from the data
Y = data['class'].values
data.drop(columns=['class'], inplace=True)

# do a preliminary check on the features, plot Y vs sorted features, sort by feature values
def explore_features():
    # check if a directory exists to hold exploritory figures
    if not os.path.isdir("exploritory"): os.mkdir("exploritory")
    # go through each feature in the data to make a quick plot
    for feature in data:
        # pull out the feature as an array
        x = data[feature].values
        # find an array of indicies that sort the feature
        i_sort = np.argsort(x)
        # sort the feature and the target
        x,y = x[i_sort],Y[i_sort]
        # make a quick plot
        fig,ax = plt.subplots()
        fig.suptitle(feature, fontsize=20)
        ax.scatter(x,y, alpha=.5)
        ax.set_xlabel('feature')
        ax.set_ylabel('class')
        # save and close figure
        fig.savefig(f"exploritory/{feature}.pdf")
        plt.close(fig)

#===============================================================================
""" get design matrix """
#===============================================================================

# engineer features
def one_hot_encode():
    # get the initial column names for the data
    column_names = data.keys()
    # it seems like all the data columns will be expanded with one hot encoding
    for feature in data:
        # get unique entries in column
        unique_values = data[feature].unique()
        # get the relevent dictionary from schema to convert the names
        schema_key = feature.replace("-","_")
        # grab the dictionary of feature names
        feature1 = schema[schema_key]
        # for each unique value, add a column to data with binary input
        for uv in unique_values:
            # uv maps to a value in feature1 dictionary, we need to get the key in feature1 that maps to the uv value
            for name,identifier in feature1.items():
                if identifier == uv:
                    description = name
            # make a new column in data named after the feature and description
            data[f"{feature}-{description}"] = (data[feature] == uv).astype(np.int32)
    # now, drop all the original columns
    data.drop(columns=column_names, inplace=True)
# run one_hot_encode to update data
one_hot_encode()

# make the design matrix
PHI = np.column_stack( (np.ones((N,1)),data.values) )

# Normalize Data -- wont need to do this step since all data columns are binary

#===============================================================================
""" cross validation """
#===============================================================================

# Use K-folds for low N; Use train-validate-test for large N
# going to use train-validate-test, because N is order 10^3
def get_cross_validation_sets():
    # get a mask that randomly shuffles the observations
    mask = np.arange(N)
    np.random.shuffle(mask)
    # shuffle the observations
    PHI0,Y0 = PHI[mask],Y[mask]

    # get the training set
    N1 = int(N*.6)
    PHI1,Y1 = PHI0[:N1],Y0[:N1]
    train = dict(PHI=PHI1, Y=Y1, N=N1)

    # get the validation set
    N2 = int(N*.2)
    PHI2,Y2 = PHI0[N1:N1+N2],Y0[N1:N1+N2]
    validate = dict(PHI=PHI2, Y=Y2, N=N2)

    # get the testing set
    N3 = N - N1 - N2
    PHI3,Y3 = PHI0[N1+N2:],Y0[N1+N2:]
    test = dict(PHI=PHI3, Y=Y3, N=N3)

    # output
    return train,validate,test
train,validate,test = get_cross_validation_sets()

# run through gradient descent a few times on the full data to get a sense of feasible eta and epoch values
# aux.gradient_descent(PHI,Y,0,0, save_curve=True)
# found good values for eta and epochs: 1e-3, 1e5

# run through train_validate_test multiple times to hone in on best values
def train_validate_test(L1,L2,**kwargs):

    # instantiate some variables to keep track of best results
    bestAc1 = 0
    bestAc2 = 0
    bestAc3 = 0
    bestl1 = 0
    bestl2 = 0
    bestEta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    bestEpochs = kwargs['epochs'] if 'epochs' in kwargs else 1e5
    bestW = None

    # cast a net and find best outcome
    for l1 in L1:
        for l2 in L2:
            # print iteration to track progress
            print(f"\nworking on {l1},{l2}...")
            # train
            W = aux.gradient_descent(train['PHI'],train['Y'],l1,l2,**kwargs)
            P_hat1 = aux.sigmoid(train['PHI'].dot(W))
            ac1 = aux.accuracy(train['Y'],P_hat1)

            # validate
            P_hat2 = aux.sigmoid(validate['PHI'].dot(W))
            ac2 = aux.accuracy(validate['Y'],P_hat2)

            # print iteration results
            print(f"train accuracy: {ac1}, validate accuracy {ac2}")

            # update best values
            if ac2 > bestAc2:
                # run the test set
                P_hat3 = aux.sigmoid(test['PHI'].dot(W))
                ac3 = aux.accuracy(test['PHI'],P_hat3)
                # update the best values
                bestAc1 = ac1
                bestAc2 = ac2
                bestAc3 = ac3
                bestl1 = l1
                bestl2 = l2
                bestW = W
                print(f"best value (so far)!")

    return dict(train_ac=bestAc1, validate_ac=bestAc2, test_ac=bestAc3, lambda1=bestl1, lambda2=bestl2, eta=bestEta, epochs=bestEpochs, W=W)
# trial 1
# L = np.array([np.e**x for x in range(10)])
# tvt1 = train_validate_test(L,L)
# best l1,l2 --> 7.3890560989306495,2.718281828459045
# validation accuracy --> 0.9919950738916257

# trial 2
# tvt2 = train_validate_test(np.linspace(5,10,10), np.linspace(0,5,10))
# best l1,l2 --> 5.0, 0.0
# validation accuracy --> 0.9907635467980296

# trial 3
# tvt3 = train_validate_test(np.array([0,7.3890560989306495]), np.array([0,2.718281828459045]))
# best l1,l2 --> 7.3890560989306495, 0.0
# validation accuracy --> 0.9821428571428571

# trial 4
# tvt4 = train_validate_test(np.linspace(0,7.3890560989306495,3), np.linspace(0,2.718281828459045,3))
# best l2,l2 --> 7.3890560989306495, 2.718281828459045
# validation accuracy --> 0.9870689655172413

# final run
tvt_results = train_validate_test(np.arange(6.4,8,.1),np.arange(2.5,3,.1))

# test run to make sure code is generally working
# just get the requirements for the best parameters
# tvt_results = train_validate_test([7.3890560989306495], [2.718281828459045],epochs=1e2)

# get Y_hat from test data
def get_test_Y_hat(tvt_results):
    # extract required arrays
    W = tvt_results['W']
    Y = test['Y']
    PHI = test['PHI']
    # get P_hat
    P_hat = aux.sigmoid(PHI.dot(W))
    # get Y_hat
    Y_hat = np.round(P_hat)
    return Y_hat
Y_hat = get_test_Y_hat(tvt_results)

#===============================================================================
""" checking the model """
#===============================================================================

# get the percent error -- can't really do this since the target is binary

# plot Y vs Y_hat -- actually, this wouldn't really be useful in a classifiation problem...
def plot_Y_vs_Y_hat(tvt_results):
    # make sure save directory exists and make filename
    if not os.path.isdir("check"): os.mkdir("check")
    filename = "Y_vs_Yhat.pdf"
    # make figure
    fig,ax = plt.subplots()
    fig.suptitle("Comparing Predicted Classification with Actual Classification", fontsize=20)
    ax.scatter(Y_hat,Y, color='r')
    ax.set_aspect(1)
    ax.set_xlabel("$\\hat{Y}", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    fig.savefig(filename)
    print(f"\nsaved {filename}")
    plt.close(fig)

# check for statistical significance of weights -- this is probably only for a linear regression model; doesn't seem applicable to classifiation
def p_test(tvt_results,alpha=0.05):
    return

# get confusion matrix
def confusion_matrix(tvt_results):
    Y = test['Y']
    # True results
    T = (Y == Y_hat)
    # False results
    F = (Y != Y_hat)
    # set up matrix
    CM = np.zeros([2,2])
    # fill in the Matrix
    CM[0,0] = (Y[T] == 0).sum()
    CM[1,1] = (Y[T] == 1).sum()
    CM[0,1] = (Y[F] == 1).sum()
    CM[1,0] = (Y[F] == 0).sum()
    # output
    return CM

# precision
def precision(CM):
    # TP / (TP + FP)
    return CM[1,1] / (CM[1,1] + CM[0,1])

# recall
def recall(CM):
    # TP / (TP + FN)
    return CM[1,1] / (CM[1,1] + CM[1,0])

# F-score
def F_score(precission,recall):
    return 2*precission*recall / (precission+recall)

# ROC_AUC

#===============================================================================
""" business application """
#===============================================================================

# what is the probability that a mushroom marked as edible is actually poisonous? p( Y(0) | Yhat(1) )
def explore1():
    #====================================================================
    # p(Y(0))
    #====================================================================

    # get the probability of it actually being poisonous
    p1 = (test['Y'][test['Y'] == 0]).shape[0]

    #====================================================================
    # p(Yhat(1))
    #====================================================================

    # get the probability of the model predicting it edible
    p2 = Y_hat.sum()

    #====================================================================
    # p( Yhat(1) | Y(0) )
    #====================================================================

    # get a mask that filters for only Y == 0 and apply it to Yhat = 1
    mask1 = test['Y'] == 0
    p3 = Y_hat[mask1].sum() / Y_hat[mask1].shape[0]

    return p3*p1/p2

# Which shrooms are the most dangerous to eat and which ones are the safest? (Shrooms with which features?) -- this may be worth looking into, I'll do it if I have time
def explore2():
    return

# if the model says its edible, which features have the highest chance of being misclassified and poisoning you? -- this is nice to have, not sure it should go in the presentation though
def explore3():

    #====================================================================
    # see which features show up the most in FP --> N_FP
    #====================================================================

    Y = test['Y'].copy()
    # get a mask of length N3 where Y and Y_hat are not equal
    mask1 = (Y != Y_hat)
    # get a mask of length N < N3 where the first filter is applied and also must be False Negatives
    mask2 = (Y[mask1] == 0)
    # get a filtered design matrix that only has false negatives and drops the bias column
    FP = test['PHI'][mask1][mask2][:,1:]
    # sum up the features column wise to find which features show up most as false negatives
    N_FP = FP.sum(axis=0)

    #====================================================================
    # see which features show up most that are poisonous --> N_p
    #====================================================================

    # get a mask to select only the poisonous data
    mask3 = (Y == 0)
    # apply the mask to PHI and drop bias column
    p = test['PHI'][mask3][:,1:]
    # add up the totals for each feature
    N_p = p.sum(axis=0)
    # get a mask to filter out 0 from N_p
    mask4 = N_p != 0
    # filter N_p and N_FP
    N_p = N_p[mask4]
    N_FP = N_FP[mask4]

    #====================================================================
    # put the two results together
    #====================================================================

    # get the ratio N_FP : N_p
    ratio = N_FP / N_p
    # find a mask to sort ratio and the names by so largest ratios are first
    mask5 = np.argsort(ratio)[::-1]
    # apply mask and take first 5
    ratio = ratio[mask5][:20]
    names = data.keys()[mask5][:20]
    # get a mask to only select ratios > .5
    mask6 = ratio >= .5
    # filter ratio and names
    ratio = ratio[mask6]
    names = names[mask6]

    #====================================================================
    # plot the results in barh figure
    #====================================================================

    # set up a barh plot
    fig,ax = plt.subplots()
    fig.suptitle("Most Deadly Features")
    index = np.arange(names.shape[0])
    ax.barh(index, ratio*100)
    ax.vlines(100,index[0]-.5,index[-1]+1, color='r')
    for i in np.arange(10,100,20):
        ax.vlines(i,index[0]-.5,index[-1]+1, color='k', alpha=.5)
    ax.set_yticks(index)
    ax.set_yticklabels(names)
    ax.set_ylim(index[0]-.5,index[-1]+1)
    ax.set_xlim(0,101)
    ax.set_xlabel("% Deadly")
    plt.subplots_adjust(left=.4)
    # save fig
    filename = "deadlyFeatures.pdf"
    fig.savefig(filename)
    print(f"\nsaved {filename}")
    plt.close(fig)

#========================================================================
# get all reportable results
#========================================================================

def run():
    # get confusion matrix
    CM = confusion_matrix(tvt_results)
    # get precision
    P = precision(CM)
    # get recall
    R = recall(CM)
    # get F-score
    F = F_score(P,R)
    # get probability of eating poisonous mushroom
    p_poison = explore1()
    # get the most deadly features
    explore3()
    return dict(CM=CM, P=P, R=R, F=F, p_poison=p_poison)
