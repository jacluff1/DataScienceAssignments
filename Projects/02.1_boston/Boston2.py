# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

#===============================================================================
# Functions
#===============================================================================

def r_squared(Y,Yhat):
    return np.sum( (Y-Yhat)**2 ) / np.sum( (Y-Y.mean())**2 )

def OLS(Y,Yhat):
    return (Y-Yhat).dot(Y-Yhat)

def solve_closed_form(PHI,Y,lambda2):
    if lambda2 > 0:
        return np.linalg.solve(PHI.T.dot(PHI) + lambda2*np.eye(P), PHI.T.dot(Y) )
    else:
        return np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(Y))

def solve_gradient_descent(PHI,Y,**kwargs):

    # kwargs
    lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
    lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-8
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else int(1e3)
    showPlot = kwargs['showPlot'] if 'showPlot' in kwargs else False

    # make sure that the input values have appropriate shapes, no way to check the value for P
    N = Y.shape[0]
    P = PHI.shape[1]
    assert PHI.shape[0] == N, "PHI must have shape: N x (P+1)"

    # start with random W
    try:
        K = Y.shape[1]
        W = np.random.randn(P*K).reshape([P,K])
    except:
        W = np.random.randn(P)

    # make empty list for OLS
    J = [0]*epochs

    # try and solve for w
    for i in range(epochs):
        Yhat = PHI.dot(W)
        J[i] = ( OLS(Y,Yhat) + lambda1*np.sum( np.abs(W) ) + lambda2*np.sum(W**2) ) / (2*N)
        W -= eta*( PHI.T.dot(Yhat-Y) + lambda1*np.sign(W) + lambda2*W ) / N
        print(W)

    return W

def train_validate(PHI1,Y1,PHI2,Y2,**kwargs):
    # kwargs
    closed = kwargs['closed'] if 'closed' in kwargs else False
    lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
    lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0

    # train
    if closed:
        W = solve_closed_form(PHI1,Y1,**kwargs)
    else:
        W = solve_gradient_descent(PHI1,Y1,**kwargs)
        Yhat1 = PHI1.dot(W)
        R21 = r_squared(Y1,Yhat1)

    # validate
    Yhat2 = PHI2.dot(W)
    R22 = r_squared(Y2,Yhat2)

    return R22

#===============================================================================
# Get Data
#===============================================================================

# import data using ID column as index
data = pd.read_csv('train.csv', index_col="ID")

# get number of observations and features (we are going to take out the target and add a ones column, so P will be the dimention of PHI)
N,P = data.shape

# pull out the target from the data
target = "medv"
Y = data[target].values
data.drop(columns=[target], inplace=True)

# make an object array of strings containing all the feature names for the PHI matrix
feature_names = np.empty(data.shape[1]+1, dtype='O')
feature_names[0] = "bias"
feature_names[1:] = data.keys()

#===============================================================================
# Data/Design Matrix
#===============================================================================

# make PHI matrix
PHI = np.column_stack( (np.ones((N,1)),data.values) )

# # normalize PHI
# for j in range(PHI.shape[1]):
#     xmin,xmax = PHI[:,j].min(), PHI[:,j].max()
#     PHI[:,j] = (PHI[:,j] - xmin) / (xmax - xmin)

#===============================================================================
# # Setup for Train - Validate - Test
#===============================================================================

# get dictionaries to hold results for training - validation - and testing
Yhat = {}
R2 = {}

# get mask to shuffle for train-validate-test
mask = np.arange(PHI.shape[0])
np.random.shuffle(mask)
PHI0,Y0 = PHI[mask],Y[mask]

# get number of observations for each set
N1 = int(N*.60)
N2 = int(N*.20)
N3 = N - N1 - N2

# split randomized observations into training (1), validation (2), and testing (3)
PHI1 = PHI[:N1]
PHI2 = PHI[N1:N1+N2]
PHI3 = PHI[N1+N2:]
Y1 = Y[:N1]
Y2 = Y[N1:N1+N2]
Y3 = Y[N1+N2:]

#===============================================================================
# Train - Validate
#===============================================================================

def run_train_validate():
    Lambda = np.arange(0,1000,100)
    maxR2 = 0
    bestl1 = 0
    bestl2 = 0
    for l1 in Lambda:
        for l2 in Lambda:
            R22 = train_validate(PHI1,Y1,PHI2,Y2, lambda1=l1, lambda2=l2)
            if R22 > maxR2:
                print(l1,l2)
                maxR2 = R22
                bestl1 = l1
                bestl2 = l2
    return dict(R2=maxR2, lambda1=bestl1, lambda2=bestl2)
