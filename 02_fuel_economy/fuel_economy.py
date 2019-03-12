# dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

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

def transform_fit_data_and_plot(data,**kwargs):
    """
    input:
        data: pd.DataFrame - cleaned DataFrame
        kwargs dictionary:
            show: bool - controls whether plot is visible or not - default = True
            save: bool - controls wheter plot is saved or not - default = False
            savename: str - what the filename of the plot will be if saved - default = "fuel_economy_plot.pdf"
    output: dic - important results

    explanation:
    Takes Data with two features, creates data matrix, and solves for the three fit parameters for the multiple linear regression model. Comes with graphing options
    """
    # kwargs
    show = True
    save = False
    savename = "fuel_economy_plot.pdf"
    if 'show' in kwargs: show = kwargs['show']
    if 'save' in kwargs: save = kwargs['save']
    if 'savename' in kwargs: savename = kwargs['savename']
    # create data matrix
    N = data.shape[0]
    X = np.ones((N,3))
    X[:,1] = data.cyl.values
    X[:,2] = data.displ.values
    Y = data.hwy.values
    # find fit parameters
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    # find min,maxes of data
    xmin,xmax = X[:,1].min(),X[:,1].max()
    ymin,ymax = X[:,2].min(),X[:,2].max()
    # make best fit line
    N_hat = 100
    X_hat = np.ones((N_hat,3))
    X_hat[:,1] = np.linspace(xmin,xmax,N_hat)
    X_hat[:,2] = np.linspace(ymin,ymax,N_hat)
    y_hat = X_hat.dot(w)
    y_hat1 = X.dot(w)
    r2 = R2(Y,y_hat1)
    results = dict(w0=w[0], w1=w[1], w2=w[2], R2=r2)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,1],X[:,2],Y,color="blue",alpha=.5,label="data")
    ax.plot(X_hat[:,1],X_hat[:,2],y_hat,color="red",linewidth=2,label="fit")
    # ax.plot(X,y_hat,color="red",linewidth=2,label="fit")
    ax.set_title("Fuel Economy",fontsize=20)
    ax.set_xlabel("N$_{cylinders}$",fontsize=10)
    ax.set_ylabel("Displacement Volume",fontsize=10)
    ax.set_zlabel("MPG (highway)", fontsize=10)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect(.5)
    ax.legend(loc='best')
    ax.annotate("R$^2$ = %s" % round(r2,3), xy=(xmin+1,y_hat[0]+13), color='red', fontsize=10)
    if show: plt.show()
    if save:
        fig.savefig(savename)
        print("saved figure to %s." % savename)
        plt.close()
    return results

def run(**kwargs):
    """
    input:
        kwargs dictionary:
            print_results - bool - controls whether results are printed in terminal
    output: None

    explanation:
    Runs the model with selected params
    """
    # kwargs
    print_results = True
    if 'print_results' in kwargs: print_results = kwargs['print_results']
    # load data
    data = load_data(**kwargs)
    # run model with graphing options
    results = transform_fit_data_and_plot(data,**kwargs)
    # print results
    if print_results:
        for key in results:
            print("\n%s = %s" % (key,results[key]))
