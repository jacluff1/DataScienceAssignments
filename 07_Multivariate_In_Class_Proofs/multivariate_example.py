# import dependencies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 101
P = 3
K = 4

x1 = np.linspace(0,1,N).reshape([N,1])
PHI = np.column_stack((np.ones((N,1)), x1, x1**2, np.exp(x1)))
W_true = np.array([5.231, 13.134, 1.139, 8.234, 6.234, 8.213, 3.194, 2.213, 7.354, 9.133, 14.135, 1.493, 6.223, 4.256, 8.432, 23.213]).reshape([P+1,K])
Y = PHI.dot(W_true) + np.random.randn(N,K) * 2

W = np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(Y))
Y_hat = PHI.dot(W_true)

def r_squared(y,y_hat):
    R2 = 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )
    # print("R-squared w/ two vars: {}".format(R2))
    return R2

R2 = []
for k in range(K):
    R2.append(r_squared(Y[:,k],Y_hat[:,k]))

def plot():
    fig = plt.figure(figsize=(15,10))
    fig.suptitle("Multivariate Example for Simulated Data")

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(PHI[:,1], PHI[:,2], Y[:,0], color='b', alpha=.5, label="Sim Data")
    ax1.plot(PHI[:,1], PHI[:,2], Y_hat[:,0], color='r', linewidth=2, label="Model: R$^2$ = %s" % round(R2[0],3))
    ax1.set_xlabel("X$_1$")
    ax1.set_ylabel("X$_1^2$")
    ax1.set_zlabel("Y")
    ax1.set_title("Y$_1$")
    ax1.legend(loc='best')

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(PHI[:,1], PHI[:,2], Y[:,1], color='b', alpha=.5, label="Sim Data")
    ax2.plot(PHI[:,1], PHI[:,2], Y_hat[:,1], color='r', linewidth=2, label="Model: R$^2$ = %s" % round(R2[1],3))
    ax2.set_xlabel("X$_1$")
    ax2.set_ylabel("X$_1^2$")
    ax2.set_zlabel("Y")
    ax2.set_title("Y$_2$")
    ax2.legend(loc='best')

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(PHI[:,1], PHI[:,2], Y_hat[:,2], color='r', linewidth=2, label="Model: R$^2$ = %s" % round(R2[2],3))
    ax3.scatter(PHI[:,1], PHI[:,2], Y[:,2], color='b', alpha=.5, label="Sim Data")
    ax3.set_xlabel("X$_1$")
    ax3.set_ylabel("X$_1^2$")
    ax3.set_zlabel("Y")
    ax3.set_title("Y$_3$")
    ax3.legend(loc='best')

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(PHI[:,1], PHI[:,2], Y[:,3], color='b', alpha=.5, label="Sim Data")
    ax4.plot(PHI[:,1], PHI[:,2], Y_hat[:,3], color='r', linewidth=2, label="Model: R$^2$ = %s" % round(R2[3],3))
    ax4.set_xlabel("X$_1$")
    ax4.set_ylabel("X$_1^2$")
    ax4.set_zlabel("Y")
    ax4.set_title("Y$_4$")
    ax4.legend(loc='best')

    plt.legend()
    plt.show()
    fig.savefig("Multivarate Example.pdf")
