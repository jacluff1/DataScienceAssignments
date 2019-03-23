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

# rename column names
rename = dict(
    crim = "per capita crime rate by town",
    zn = "proportion of residential land zoned for lots over 25,000 sq.ft",
    indus = "proportion of non-retail business acres per town",
    chas = "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
    nox = "nitrogen oxides concentration (parts per 10 million)",
    rm = "average number of rooms per dwelling",
    age = "proportion of owner-occupied units built prior to 1940",
    dis = "weighted mean of distances to five Boston employment centres",
    rad = "index of accessibility to radial highways",
    tax = "full-value property-tax rate per $10,000",
    ptratio = "pupil-teacher ratio by town",
    black = "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
    lstat = "lower status of the population (percent)",
    medv = "median value of owner-occupied homes in $1000s"
)
# capitalize
for key in rename:
    rename[key] = rename[key].capitalize()

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

# # make an instance of LinearRegression
# lr = LinearRegression(PHI,Y,feature_names)
#
# # add the train, validate, and test data (normalized by default)
# lr.cv_add_sets()

# use method cv_grid_search(**kwargs) until best hyperparameters found
# lr.cv_grid_search(L1=np.arange(0,10,1), L2=np.arange(0,10,1), epochs=1e5)

# add parameters
# lr.cv_results(lambda1=8.5, lambda2=2.5, epochs=1e5, eta=.001)
