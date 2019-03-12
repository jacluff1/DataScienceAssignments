# import dependencies
import numpy as np
import pandas as pd

def R2(Y,y_hat):
    """
    input:
        Y: np.array - Y data array
        y_hat: np.array - Y model array
    output:

    explanation:
    Returns the R^2 value, or the wellness of fit, for the linear regression model.
    """
    return 1 - np.sum( (Y - y_hat)**2 ) / np.sum( (Y - Y.mean())**2 )

def load_data(**kwargs):
    """
    input:
        kwargs dictionary:
            filename: str - the csv file that will load - default = "fuel_economy.csv"
    output: pandas.DataFrame

    explanation:
    Loads csv file into a pandas.DataFrame
    """
    # kwargs
    filename = "fuel_economy.csv"
    if 'filename' in kwargs: filename = kwargs['filename']

    return pd.read_csv(filename)

def transform_data(data):
    """
    explanation:
    input:
        data - pd.DataFrame - raw data loaded from csv file
    output:
    """
    # get the number of observations
    N = data.shape[0]

    # convert all columns with discrete values into binary values for the one-hot encoding.
    data1 = pd.get_dummies(data, columns=["manufacturer", "model", "cyl", "trans", "drv", "fl", "class"])

    # pull out the target
    y = data1.hwy.values.reshape([N,1])

    # drop the unused columns
    data1 = data1.drop(columns=['cty', 'hwy'])

    # set up the Data Matrix
    X = np.column_stack( (np.ones((N,1)),data1.values) )

    return X,y

def fit_data(X,y):
    """
    explanation:
    input: data: pd.DataFrame - cleaned DataFrame
    output: np.array - solved w
    """
    return np.linalg.solve(X.T.dot(X), X.T.dot(y))

def find_target(w,X):
    """
    """
    return X.dot(w)

def run(**kwargs):
    """
    input:
    output: None
    explanation:
    """
    data = load_data(**kwargs)
    X,y = transform_data(data)
    w = fit_data(X,y)
    y_hat = find_target(w,X)
    r_squared = R2(y,y_hat)
    return dict(w=w, X=X, y_hat=y_hat, R2=r_squared)
