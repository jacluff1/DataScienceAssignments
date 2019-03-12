# import external dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

# import from my library @ https://github.com/jacluff1/DJAK
# having some issue with self and/or positional arguments so I inserted them directly into the module.
# from DJAK.DataScience.LinearRegression import LinearRegression

def get_data(preview=False):
    """
    explanation:
        load the data from the csv file into a Pandas.DataFrame with column names 'x' and 'y'.
    input:
        kwargs:
            preview: bool - if true, will do a quick plot of the data to check what it looks like. Default = False.
    output:
        pd.DataFrame - the data from the 'gaussianrbf.csv' file
    """
    data = pd.read_csv("gaussianrbf.csv", names=['x','y'])
    if preview:
        data.plot(kind='scatter', x='x', y='y')
        plt.show()
    return data

def OLS(y,y_hat):
    """
    explanation:
        The ordinary least square in vector dot product form.
    input:
        y: np.array - N x 1 array of the target data
        y_hat: np.array - N x 1 array of target model
    output:
        float: ordinary least square dot product
    """
    # make sure the shape of y and y_hat are the same
    assert y.shape == y_hat.shape, "in OLS, y and y_hat must be same shape!"
    assert y.shape == (y.shape[0],1), "y and y_hat should be shape: N x 1"
    return (y-y_hat).T.dot(y - y_hat)[0,0]

def Gradient_Descent(PHI,y,**kwargs):
    """
    explanation:
        Performs the gradient descent method for finding a numerical solution for w.
    input:
        PHI: np.array - N x (P+1) - the design matrix.
        y: np.array - N x 1 - the target data
        kwargs:
            eta: float - the 'step factor' - default = 1e-4
            epochs: int - the number of iterations to find OLS - default = int(2e3)
            show: bool - will show the plot if True - default = True
            save: bool - will save the figure if True - default = False
            savename: str - what the filename will be if saved - default = "gradientDescentConvergence.pdf"
            title: str - the title of the plot - default = Ordinary Least Squares Convergence for Gradient Descent"
    output:
        w: np.array - (D+1) x 1 - the solution values
    """
    # make sure that the input values have appropriate shapes, no way to check the value for P
    N = y.shape[0]
    y = y.reshape([N,1])
    assert PHI.shape[0] == N, "PHI must have shape: N x (P+1)"

    # kwargs
    eta = 1e-4
    epochs = int(2e3)
    show = True
    save = False
    savename = "gradientDescentConvergence.pdf"
    title = "Ordinary Least Squares Convergence for Gradient Descent"
    if 'eta' in kwargs: eta = kwargs['eta']
    if 'epochs' in kwargs: epochs = kwargs['epochs']
    if 'show' in kwargs: show = kwargs['show']
    if 'save' in kwargs: save = kwargs['save']
    if 'savename' in kwargs: savename = kwargs['savename']
    if 'title' in kwargs: title = kwargs['title']

    # make empty list for OLS
    J = []
    # start with random w
    w = np.random.randn(PHI.shape[1]).reshape([PHI.shape[1],1])
    # try and solve for w
    for epoch in range(epochs):
        y_hat = PHI.dot(w)
        J.append(OLS(y,y_hat))
        try:
            w -= eta*PHI.T.dot(y_hat-y)
        except:
            print(w.shape)
            pdb.set_trace()

    # preview J to make sure it converges
    plt.plot(np.arange(epochs),J, color='b')
    plt.suptitle(title)
    plt.xlabel("epoch")
    plt.ylabel("J")
    plt.annotate("J = {:04f}".format(J[-1]), xy=(epochs*.8, J[0]*.2), color='b', fontsize=10)
    if J[-1] < 20: print("Parameters may need to be adjusted.")
    if show: plt.show()
    if save:
        plt.savefig(savename)
        print("\nSaved to %s" % savename)
        plt.close()
    return w

def Gaussian_RBF(x,y,**kwargs):
    """
    explanation:
        Fits the data to the target using a linear combination of constructed Gaussian functions. Constructs the model using "P" values for mu, which are equally spaced across the domain of the data. At this point, there is only one number used for the variance and is calclated using the spatial span of the data divided by the number of basis functions.
    input:
        x: np.array - N x ? - feature data
        y: np.array - N x 1 - target data
        kwargs:
            P: int - the order of the model, the number of gaussian functions used to model data - default = 5
            method: str - the method used for finding the solutions - default = 'Closed Form'. Other possible methods ready to impliment include: "Gradient Descent"
    output: dict -
        w: np.array - (D+1) x 1 - the solution values
        y_hat: np.array - N x 1 - the model target
        mu: np.array - P x , - the means used for the gaussian functions
        var: np.array - P x , - the variance used for the gaussian functions
        method: str - title friendly string that describes the model and order of the model
        solve_method: str - description of what solution method was used to solve the model
    """
    # check shapes of input
    N = x.shape[0]
    # assert x.shape[0] == y.shape[0], "x & y must have the same N value!"
    # assert y.shape == (N,1), "y must be shape: N x 1!"

    # kwargs
    P = 5
    method = "Closed Form"
    if 'P' in kwargs: P = kwargs['P']
    if 'method' in kwargs: method = kwargs['method']

    # construct 1 x N array of 'mu'
    xmin,xmax = x.min(),x.max()
    mu = np.linspace(xmin,xmax,P)

    # construct 1 x N array of variance
    var = np.ones(P) * (xmax-xmin)/P

    # basis function
    def phi_j(x_i,mu,var):
        """
        explanation:
            gaussian radial basis function
        input:
            x: float OR np.array - data feature(s) of x_i
            mu: float - mean of the distribution: mu_j
            var: float - variance of the distribution: sigma_j^2
        output: float
        """
        # L2 norm squared of the difference between x and mu
        diff_vec = x_i - mu
        try:
            L2 = diff_vec.T.dot(diff_vec)[0,0]
        except:
            L2 = diff_vec**2
        return np.exp(-L2 / (2*var))

    # make design matrix
    PHI = np.ones((N,P+1))
    for j in np.arange(1,P+1):
        PHI[:,j] = phi_j(x, mu[j-1], var[j-1])

    # solve for w
    if method == "Gradient Descent":
        w = Gradient_Descent(PHI,y,**kwargs)
    else:
        # pdb.set_trace()
        w = np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(y))

    # find y_hat
    y_hat = PHI.dot(w)

    return dict(w=w, y_hat=y_hat, mu=mu, var=var, method="Gaussian Radial Basis Function: P = %s, %s" % (P,method), solve_method=method)

def Polynomial_Regression(x,y,**kwargs):
    """
    explanation:
        Fits the data to the target using a linear polynomial regression model. I do hope to include an algorithm for a design matrix that allows for multiple feature data, but I haven't figured out how to systimatically include the cross terms yet. So only 1 target is possible using this method for the time being.
    input:
        x: np.array - N x 1 - feature data
        y: np.array - N x 1 - target data
        kwargs:
            P: int - the order of the polynomial - default = 5
            method: str - the method used for finding the solutions - default = 'Closed Form'. Other possible methods ready to impliment include: "Gradient Descent"
    output: dict -
        w: np.array - (D+1) x 1 - the solution values
        y_hat: np.array - N x 1 - the model target
        method: str - title friendly string that describes the model and order of the model
        solve_method: str - description of what solution method was used to solve the model
    """
    # check shapes of input
    N = x.shape[0]
    assert x.size == N, "This version of polynomial regression only handles 1 feature!"
    assert x.size == y.size, "x and y must be the same size!"
    x = x.reshape([N,1])
    y = y.reshape([N,1])

    # kwargs
    P = 5
    method = "Closed Form"
    if 'P' in kwargs: P = kwargs['P']
    if 'method' in kwargs: method = kwargs['method']

    # make design matrix
    # PHI = np.ones((N,P+1))
    # for p in np.arange(1,P):
    #     try:
    #         PHI[:,p+1] = x**p
    #     except:
    #         pdb.set_trace()
    PHI = np.ones((N,1))
    for p in np.arange(1,P+1):
        PHI = np.column_stack((PHI,x**p))

    # solve for w
    if method == "Gradient Descent":
        w = Gradient_Descent(PHI,y,**kwargs)
    else:
        # pdb.set_trace()
        w = np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(y))

    # find y_hat
    y_hat = PHI.dot(w)

    return dict(w=w, y_hat=y_hat, method="Polynomial Regression: P = %s, %s" % (P,method), solve_method=method)

def Plot(data,**kwargs):
    """
    explanation:
        fits the given data using different models and solution methods makes the plot to compare them.
    input:
        data: pd.DataFrame - the data that you wish to analyze
        kwargs:
            show: bool - will show the plot if True - default = True
            save: bool - will save the figure if True - default = False
            savename: str - what the filename will be if saved - default = "LRviaBasisVectors.pdf"
            title: str - the title of the plot - default = "Linear Regression through Combination of Basis Vectors"
            figsize: tuple - the size of the figure - default = (15,10)
            linewidth: int - the width of the model lines on the plot - default = 3
            fontsize: int - the size of the font in the title, axis labels, etc... - default = 20
    output:

    """
    # kwargs
    show = True
    save = False
    savename = "LRviaBasisVectors.pdf"
    title = "Linear Regression through Linear Combination of Basis Vectors"
    figsize = (15,10)
    linewidth = 3
    fontsize = 20
    if 'show' in kwargs: show = kwargs['show']
    if 'save' in kwargs: save = kwargs['save']
    if 'savename' in kwargs: savename = kwargs['savename']
    if 'title' in kwargs: title = kwargs['title']
    if 'figsize' in kwargs: figsize = kwargs['figsize']
    if 'linewidth' in kwargs: linewidth = kwargs['linewidth']
    if 'fontsize' in kwargs: fontsize = kwargs['fontsize']

    # collect data
    N = data.shape[0]
    # x = data.x.values.reshape([N,1])
    # y = data.y.values.reshape([N,1])
    x = data.x.values
    y = data.y.values

    # fit gaussian radial basis functions model
    grbf1 = Gaussian_RBF(x,y, P=20, **kwargs)
    grbf2 = Gaussian_RBF(x,y, P=20, method="Gradient Descent", eta=1e-3, epoch=int(1e4), **kwargs)
    pr1 = Polynomial_Regression(x,y, P=40, **kwargs)
    # did not find converging solution
    # pr2 = Polynomial_Regression(x,y, P=40, method="Gradient Descent", **kwargs)

    # construct figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title,fontsize=fontsize)
    ax = fig.add_subplot(111)
    ax.scatter(x,y, color='b', label='data', alpha=.5)
    ax.plot(x,grbf1['y_hat'], label=grbf1['method'], linewidth=linewidth, color='r')
    ax.plot(x,grbf2['y_hat'], label=grbf2['method'], linewidth=linewidth, color='orange')
    ax.plot(x,pr1['y_hat'], label=pr1['method'], linewidth=linewidth, color='g')
    ax.set_xlabel("x", fontsize=fontsize)
    ax.set_ylabel("Target", fontsize=fontsize)
    # did not find conferging solution
    # ax.plot(x,pr2['y_hat'], label=pr2['method'], linewidth=linewidth)
    ax.legend(loc='best')
    ax.set_xlim([x.min(),x.max()])

    if show: plt.show()
    if save:
        plt.savefig(savename)
        print("\nSaved to %s" % savename)
        plt.close()

# grbf = LinearRegression.Gaussian_RBF(data.x.values, data.y.values)
