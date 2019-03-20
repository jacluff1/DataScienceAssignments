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
Y = data[target].values.reshape([N,1])
data.drop(columns=[target], inplace=True)

# make an object array of strings containing all the feature names for the PHI matrix
feature_names = np.empty(data.shape[1]+1, dtype='O')
feature_names[0] = "bias"
feature_names[1:] = data.keys()

# make PHI matrix
PHI = np.column_stack( (np.ones((N,1)),data.values) )

# make an instance of LinearRegression using PHI and Y
lr = LinearRegression(PHI,Y,feature_names)
