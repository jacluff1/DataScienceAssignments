## dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(**kwargs):
    """
    input:
        kwargs dictionary:
            filename: str - the csv file that will load - default = "moore.csv"
    output: pandas.DataFrame

    explanation:
    Loads csv file into a pandas.DataFrame
    """
    # kwargs
    filename = "moore.csv"
    if 'filename' in kwargs: filename = kwargs['filename']

    return pd.read_csv(filename)

def clean_data(data):
    """
    input: pandas.DataFrame
    output: pandas.DataFrame

    explanation:
    removes unwanted elements from strings in the entire  DataFrame, converts columns used for linear regression into int data type, and sorts the data by year. Returns cleaned DataFrame.
    """
    data = data.replace(regex=[" ", ",", "~", "cca", "nm", r"mm.", r"\[...", r"\[.."], value="")
    data.year = data.year.astype(int)
    data.N = data.N.astype(int)
    return data.sort_values("year")

def denominator(X):
    """
    input:
        X: np.array - array of X data
    output: float

    explanation:
    Returns the denominator used in simple linear regression
    """
    return np.average(X**2) - np.average(X)**2

def intercept(X,Y,den):
    """
    input:
        X: np.array - X data array
        Y: np.array - Y data array
        den: float - denominator for simple linear regression model.
    output: float

    explanation:
    Returns the w_0, or the y-intercept, for the simple linear regression
    """
    num =  np.average(Y) * np.average(X**2) - np.average(X) * np.average(X*Y)
    return num/den

def slope(X,Y,den):
    """
    input:
        X: np.array - X data array
        Y: np.array - Y data array
        den: float - denominator for simple linear regression model.
    output: float

    explanation:
    Returns the w_1, or the slope, for the simple linear regression model.
    """
    num = np.average(X*Y) - np.average(X) * np.average(Y)
    return num/den

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
            savename: str - what the filename of the plot will be if saved - default = "moore_plot.pdf"
    output: dic - important results

    explanation:
    moore's law should show an exponential growth over time in the number of transistors. If we take the natural log of the data, it should fit to a line.
    """
    # kwargs
    show = True
    save = False
    savename = "moore_plot.pdf"
    if 'show' in kwargs: show = kwargs['show']
    if 'save' in kwargs: save = kwargs['save']
    if 'savename' in kwargs: savename = kwargs['savename']
    # transform
    X = data.year.values
    Y = np.log(data.N.values)
    # find fit
    den = denominator(X)
    m = slope(X,Y,den)
    b = intercept(X,Y,den)
    y_hat = m*X + b
    r2 = R2(Y,y_hat)
    # collect results
    results = dict(w1=m, w0=b, R2=r2)
    # plot
    xmin,xmax = X.min(),X.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X,Y,color="blue",alpha=.5,label="data")
    ax.plot(X,y_hat,color="red",linewidth=2,label="fit")
    ax.set_title("Moore's Law",fontsize=25)
    ax.set_xlabel("year",fontsize=20)
    ax.set_ylabel("ln(N)",fontsize=20)
    ax.set_xlim(xmin,xmax)
    ax.legend(loc='best')
    ax.annotate("R$^2$ = %s" % round(r2,3), xy=(xmin+1,y_hat[0]+13), color='red', fontsize=10)
    if show: plt.show()
    if save:
        fig.savefig(savename)
        print("\nSaving figure as %s." % savename)
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
    # clean data
    data = clean_data(data)
    # run model with graphing options
    results = transform_fit_data_and_plot(data,**kwargs)
    # print results
    if print_results:
        for key in results:
            print("\n%s = %s" % (key,results[key]))
