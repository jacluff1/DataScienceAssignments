import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
import numpy as np

# make a quick plot
def basic_plots(lr):

    plt.close('all')

    Wdf = lr.model_results
    W = Wdf.W[1:]
    # pdb.set_trace()
    Wpos = W[W > 0].sort_values()
    Wneg = W[W < 0].sort_values()
    ipos = np.arange(1, Wpos.shape[0]+1)
    ineg = np.arange(1, Wneg.shape[0]+1)

    fig,ax = plt.subplots()
    fig.suptitle("Factors that Negatively Affect Sale Price")

    ax.barh(ineg,Wneg.values,align='center',tick_label=Wneg.index)
    # ax.vlines(0,ineg.min(),ineg.max(), color='r', label="Average")
    # ax.set_yticks(i,feature_names)
    ax.legend()
    fig.savefig("negative.pdf")
    plt.close(fig)

    #===========================================================================

    fig,ax = plt.subplots()
    fig.suptitle("Factors that Positively Affect Sale Price")

    ax.barh(ipos,Wpos.values,align='center',tick_label=Wpos.index, label="Features")
    # ax.vlines(0,ipos.min(),ipos.max(), color='r', label="Average")
    # ax.set_yticks(i,feature_names)
    ax.legend()
    fig.savefig("positive.pdf")
    plt.close(fig)

    #===========================================================================

    # PHI = lr.PHI
    # for j in range(1,lr.P):
    #     PHI[:,j] = (PHI[:,j] - lr.xmin[j]) / (lr.xmax[j] - lr.xmin[j])
    #
    # W = Wdf.Wn
    # Y = lr.Y
    # Yhat = lr.Y_hat(PHI,W)
    #
    # i_x = np.arange(Y.shape[0])
    # isort = np.argsort(Y)
    # # Y = Y[isort]
    # # Yhat = Yhat[isort]
    #
    # fig,ax=plt.subplots()
    # fig.suptitle("Comparing Y and Yhat")
    #
    # ax.scatter(i_x,Y, color='b', label="Data")
    # ax.scatter(i_x,Yhat, color='g', label="Model")
    # ax.legend(loc='best')
    # ax.set_xlabel("Observation")
    # ax.set_ylabel("median value of owner-occupied homes in $1000s.")
    #
    # fig.savefig("YvsYhat.pdf")
    # plt.close(fig)
