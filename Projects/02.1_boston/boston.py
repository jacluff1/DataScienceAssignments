# import dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
# import from my library
from DataScience.LinearRegression import LinearRegression

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

# make PHI matrix
PHI = np.column_stack( (np.ones((N,1)),data.values) )

# normalize PHI
for j in np.arange(1,PHI.shape[1]):
    xmin,xmax = PHI[:,j].min(), PHI[:,j].max()
    PHI[:,j] = (PHI[:,j] - xmin) / (xmax - xmin)

# make an instance of LinearRegression
lr = LinearRegression()

# get the train, validate, and test data
train, validate, test = lr.get_train_validation_test_sets(PHI,Y)

# train - validate

# tv_results = lr.grid_search_train_validate(train,validate, save_curve=True, eta=1e-4, epochs=1e7, L1=np.arange(0,50,10), L2=np.arange(0,50,10))

# L1: (0,50,10)
# L2: (0,50,10)
# l1,l2 --> 0,10

# tv_results = lr.grid_search_train_validate(train,validate, save_curve=True, eta=1e-4, epochs=1e7, L1=[0], L2=np.arange(0,20,1))

# L2: (0,20,1)
# l2 --> 4

tv_results = lr.grid_search_train_validate(train,validate, save_curve=True, eta=1e-4, epochs=1e7, L1=np.arange(0,2,1), L2=np.arange(0,10,1))

# L1: (0,10,1)
# L2: (0,10,1)
#l1,l2 --> ?,?
