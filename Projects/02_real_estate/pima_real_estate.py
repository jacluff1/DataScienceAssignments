# import dependencies
import numpy as np
import pandas as pd
import requests
import zipfile
import io
import os
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import from my library
from DataScience.LinearRegression import LinearRegression

# make instance of LinearRegression
lr = LinearRegression()

def extract_data(**kwargs):
    """
    """
    print("\nstarting data extraction from the web...\n")
    startYear = kwargs['startYear'] if 'startYear' in kwargs else 2012

    # set up directories
    sale_dir = "sale/"
    real_dir = "real/"
    if not os.path.isdir(sale_dir): os.mkdir(sale_dir)
    if not os.path.isdir(real_dir): os.mkdir(real_dir)

    # make a list of years
    currentYear = pd.datetime.now().year
    years = np.arange(startYear,currentYear+2)

    # go through each year and download the data
    for year in years:

        # collect category meta infomation
        category = [
            # make a dictionary to collect Sales data
            dict(
                url = f"http://www.asr.pima.gov/Downloads/Data/sales/{year}/SALE{year}.ZIP",
                save_dir = sale_dir,
                filename = f"Sale{year}.csv"
            ),
            # make a dictionary to collect residential real estate data
            dict(
                url = f"http://www.asr.pima.gov/Downloads/Data/realprop/{year}/noticeval/MAS{str(year)[2:]}.ZIP",
                save_dir = real_dir,
                filename = f"Mas{str(year)[2:]}.csv"
            )
        ]

        # go through each category of data
        for cat in category:

            # start requests session
            r = requests.get(cat['url'])
            # open up the zipfile and try to extract it
            try:

                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    try:
                        z.extract(cat['filename'])
                        os.rename(cat['filename'], cat['save_dir']+cat['filename'])
                    except KeyError:
                        try:
                            z.extract(cat['filename'].upper())
                            os.rename(cat['filename'].upper(), cat['save_dir']+cat['filename'])
                        except KeyError:
                            print(f"\nCan't find {cat['filename']} in zip, check if their conventions changed.\n")
                            continue
                print(f"\nsaved file -- {cat['save_dir']}{cat['filename']}\n")
            except:
                print(f"\nno zip file available for {cat['filename']}")
                continue

def append_data_by_source(**kwargs):
    """
    """
    extract = kwargs['extract'] if 'extract' in kwargs else False
    append_sale = kwargs['append_sale'] if 'append_sale' in kwargs else True
    append_real = kwargs['append_real'] if 'append_real' in kwargs else True

    # extract data if chosen
    if extract: extract_data(**kwargs)

    # collect category meta infomation
    category = [
        # make a dictionary to hold sales meta data
        dict(
            df = pd.DataFrame(),
            save_dir = "sale",
            filename = 'sale.csv',
            append = append_sale
        ),
        # make a dictionary to hold residential real estate data
        dict(
            df = pd.DataFrame(),
            save_dir = "real",
            filename = 'real.csv',
            append = append_real
        )
    ]

    # go through both sets of data and put all the csvs into a single DataFrame
    for cat in category:
        if not cat['append']: continue
        print(f"\nappending all {cat['save_dir']} data...\n")
        # get the list of all csv files in the directory
        csvs = os.listdir(cat['save_dir'])
        # go through each csv in the directory list
        for csv in csvs[::-1]:
            # load csv
            df = pd.read_csv(f"{cat['save_dir']}/{csv}", low_memory=False)
            # add csv to totals DataFrame
            cat['df'] = cat['df'].append(df, ignore_index=True)
            # print shape of the two dataFrames -- for tracking
            print(f"\nfilename: {csv}")
            print(f"shape of Single DataFrame: {df.shape}")
            print(f"shape of total DataFrame: {cat['df'].shape}\n")
        # save csv
        print(f"saving csv to {cat['filename']}\n")
        cat['df'].to_csv(cat['filename'])

def join(**kwargs):
    """
    """
    # if 'sale_csv' doesn't exist, create it
    if not os.path.isfile("sale.csv"): append_data_by_source(**kwargs)
    print("\ntransforming sale data...\n")
    # load file to DataFrame
    sale = pd.read_csv("sale.csv", index_col="Unnamed: 0", low_memory=False)

    # if 'real_csv' doesn't exist, create it
    if not os.path.isfile("sale.csv"): append_data_by_source(**kwargs)
    print("\ntransforming real data...\n")
    # load file to DataFrame
    real = pd.read_csv("real.csv", index_col="Unnamed: 0")

    # to join the two DataFrames, make sure the join by column matches
    real = real.rename(index=str, columns={'PARCEL':'Parcel'})

    # join the DataFrames
    join = sale.merge(real, how='left', on='Parcel')

    return join

def clean_and_transform(**kwargs):
    """
    """
    # grab the cleaned, joined DataFrame
    joinDF = kwargs['join_data'] if 'join_data' in kwargs else join(**kwargs)
    print(f"\nstarting shape of join: {joinDF.shape}")

    # only keep single family homes
    joinDF = joinDF[joinDF.PropertyType == 'Single Family']
    print(f"\nshape after filtering 'Single Family': {joinDF.shape}")


    # only keep single family homes
    joinDF = joinDF[joinDF.SFRCONDO == 'S']
    print(f"\nshape after filtering 'SFRCONDO': {joinDF.shape}")

    # set TAXYEAR to integers
    joinDF.TAXYEAR.fillna(method='ffill', inplace=True)
    joinDF.TAXYEAR = joinDF.TAXYEAR.astype(int)
    tax_years = joinDF.TAXYEAR.values
    # set SaleDate to pd.datetime
    joinDF.SaleDate = joinDF.SaleDate.fillna(method='ffill')
    joinDF.SaleDate = pd.to_datetime(joinDF.SaleDate, format='%Y%m')
    sale_years = pd.DatetimeIndex(joinDF.SaleDate).year
    # get rid of data where sale year and tax year don't match
    joinDF = joinDF[ sale_years == tax_years]
    print(f"\nshape after filtering out tax year != sale year: {joinDF.shape}")

    # delete columns
    drop_columns = ['PropertyType', 'ValidationDescription', 'LASTACTION', 'VALUATIONC', 'APPLICATION', 'PHONE', 'MAIN', 'SFRCONDO', 'COMPLEXID']
    joinDF.drop(columns=drop_columns, inplace=True)
    print(f"\nshape after dropping columns: {joinDF.shape}")

    # fill nans by column, using context of that column
    joinDF.IntendedUse.fillna("Unknown", inplace=True)

    # convert SaleDate to useful numerical value
    joinDF.SaleDate = pd.to_numeric(joinDF.SaleDate.values)

    # convert RecordingDate to numerical value
    joinDF.RecordingDate = pd.to_datetime(joinDF.RecordingDate)
    joinDF.RecordingDate = pd.to_numeric(joinDF.RecordingDate)

    # set inspection column to a useful datetime
    joinDF.INSPECTION = pd.to_datetime(joinDF.INSPECTION.values)
    joinDF.INSPECTION = pd.to_numeric(joinDF.INSPECTION)

    # set columns that should be one-hot encoded
    classification_columns = ['IntendedUse', 'Deed', 'Financing', 'BuyerSellerRelated', 'Solar', 'PersonalProperty', 'PartialInterest', 'CLASS', 'QUALITY', 'WALLS', 'ROOF', 'HEAT', 'COOL', 'PATIO', 'CONDITION', 'GARAGE', 'APPRAISER']

    # one hot encoding
    joinDF = lr.one_hot_encoding(joinDF,classification_columns)
    print(f"shape after one hot encoding: {joinDF.shape}")

    # handle nans?

    return joinDF

def get_PHI_and_target_and_normalizations(**kwargs):
    """
    """
    # grab the cleaned, joined DataFrame
    cleanDF = kwargs['clean_data'] if 'clean_data' in kwargs else clean_and_transform(**kwargs)
    target = kwargs['target'] if 'target' in kwargs else "SalePrice"

    # extract target from
    y = cleanDF[target].values.copy().astype(float)
    cleanDF = cleanDF.drop(columns=[target, 'Parcel'])

    # save the feature names
    features = cleanDF.keys()

    # instantiate array to hold normalization factors
    Xnorm = np.ones(cleanDF.shape)
    XunNorm = np.ones_like(Xnorm)

    # normalize later?
    for j,f in enumerate(features):
        # check the column
        uniques = cleanDF[f].unique()
        assert uniques.shape[0] > 1, f"since {f} is single valued, just drop it!"
        if uniques.shape[0] == 2:
            Xnorm[:,j] = cleanDF[f].values
            XunNorm[:,j] = cleanDF[f].values
            continue
        # normalize column
        try:
            x_j = cleanDF[f].values.astype(float)
            min, max = x_j.min(), x_j.max()
            Xnorm[:,j] = (x_j - min) / (max - min)
            XunNorm[:,j] = XunNorm[:,j]*(max-min) + min
        except ValueError:
            print(f"couldn't convert {f} to numbers!")

    # get PHI from normalized X
    PHI = np.column_stack( (np.ones( (Xnorm.shape[0],1) ), Xnorm) )

    cleanDF.to_csv("cleaned.csv")

    N = y.shape[0]

    # return design matrix
    return PHI,y.reshape([N,1])

parameters = dict(
    eta = 1e-2,
    epochs = 1e4,
    lambda1 = None,
    lambda2 = None
    )

# PHI,Y = get_PHI_and_target_and_normalizations()
# results1 = lr.plot_J(**parameters) # found good eta and epochs here
# results = lr.cv_grid_search_train_validate_test(**parameters) # found good lambda1 and lambda2 here
# optimal_W = results['W']
# optimal_parameters = results['optimal_parameters']
# R2s = results['R2']

def make_graphs(results,**kwargs):
    """
    """
    lower_cutoff = kwargs['lower_cutoff'] if 'lower_cutoff' in kwargs else 10
    upper_cutoff = kwargs['upper_cutoff'] if 'upper_cutoff' in kwargs else 150
    sortby = kwargs['sortby'] if 'sortby' in kwargs else 'weights'

    DF = pd.read_csv("cleaned.csv", index_col='Unnamed: 0')
    keys = DF.keys()

    W = results['W']
    cols = np.empty(W.shape[0], dtype='O')
    cols[0] = "Bias Weight"
    cols[1:] = keys

    # # un-normalize weights
    # W_un = W.copy()
    # # pdb.set_trace()
    # for i,col in enumerate(keys):
    #     x_j = DF[col].values
    #     xmin,xmax = x_j.min(),x_j.max()
    #     W_un[i+1] *= (xmax-xmin)
    #     W_un[i+1] += xmin

    # df = pd.DataFrame(dict(names=cols, weights=W, unNweight=W_un))
    df = pd.DataFrame(dict(names=cols, weights=W))
    df.sort_values(sortby)

    minorDF = df[df.weights <= lower_cutoff].sort_values(sortby)
    intermidiateDF = df[(df.weights > lower_cutoff) & (df.weights < upper_cutoff)].sort_values(sortby)
    majorDF = df[df.weights >= upper_cutoff].sort_values(sortby)

    plt.close('all')

    #===================================================================================================================
    # bar graphs
    #===================================================================================================================

    fig = plt.figure()
    fig.suptitle("A Look at How House Features Affect Sale Price")
    plt.title("Minor Features")
    index_mn = np.arange(minorDF.shape[0])
    plt.bar(index_mn, minorDF.weights.values.astype(float))
    # plt.bar(index_mn, minorDF.unNweight.values.astype(float))
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(index_mn, tuple(minorDF.names), rotation='vertical')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("minor.pdf")
    plt.close(fig)


    fig = plt.figure()
    fig.suptitle("A Look at How House Features Affect Sale Price")
    plt.title("Intermidiate Features")
    index_mn = np.arange(intermidiateDF.shape[0])
    plt.bar(index_mn, intermidiateDF.weights.values.astype(float))
    # plt.bar(index_mn, intermidiateDF.unNweight.values.astype(float))
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(index_mn, tuple(intermidiateDF.names), rotation='vertical')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("intermidiate.pdf")
    plt.close(fig)


    fig = plt.figure()
    fig.suptitle("A Look at How House Features Affect Sale Price")
    plt.title("Major Features")
    index_mn = np.arange(majorDF.shape[0])
    plt.bar(index_mn, majorDF.weights.values.astype(float))
    # plt.bar(index_mn, majorDF.unNweight.values.astype(float))
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(index_mn, tuple(majorDF.names), rotation='vertical')
    fig.subplots_adjust(bottom=0.3)
    fig.savefig("major.pdf")
    plt.close(fig)

    #===================================================================================================================
    # J vs lambda
    #===================================================================================================================

    J = results['J']
    i = results['i']
    j = results['j']
    L1 = results['L1']
    L2 = results['L2']

    fig,ax = plt.subplots(nrows=2)
    fig.suptitle("J vs $\\lambda$")

    # pdb.set_trace()
    ax[0].scatter(L1,J[0,:,j], color='r', label='Training')
    ax[0].scatter(L1,J[1,:,j], color='b', label='Validation')
    ax[0].set_ylabel("J")
    ax[0].set_xlabel("$\\lambda_1$")

    ax[1].scatter(L2,J[0,i,:], color='r', label='Training')
    ax[1].scatter(L2,J[1,i,:], color='b', label='Validation')
    ax[1].set_ylabel("J")
    ax[1].set_xlabel("$\\lambda_2$")

    # plt.tight_layout()
    plt.legend(loc='best')
    fig.subplots_adjust(hspace=.4, top=.9)
    fig.savefig("JvsLambda.pdf")
    plt.close(fig)

    #===================================================================================================================
    # J
    #===================================================================================================================

    # fig = plt.figure()

def add_hoc_model_adjustments(PHI,Y,**kwargs):
    """
    """
    L1 = kwargs['L1'] if 'L1' in kwargs else np.arange(0,500,50)
    L2 = kwargs['L2'] if 'L2' in kwargs else np.arange(0,100,10)
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else int(1e3)
    train_per = kwargs['train_per'] if 'train_per' in kwargs else .60
    validate_per = kwargs['val_per'] if 'val_per' in kwargs else .20

    # get lengths of L2 and L2
    Nl1 = L1.shape[0]
    Nl2 = L2.shape[0]
    P = PHI.shape[1]

    # create a mask to shuffle the observations
    mask = np.arange(PHI.shape[0])
    np.random.shuffle(mask)
    PHI,Y = PHI[mask],Y[mask]

    # get numbers of observations and numbers of observations for each stage
    N = PHI.shape[0]
    P = PHI.shape[1]
    N_tr = int(N*train_per)
    N_vl = int(N*validate_per)
    N_ts = N - N_tr - N_vl

    # split up the observations
    PHI1 = PHI[:N_tr]
    PHI2 = PHI[N_tr:N_tr+N_vl]
    PHI3 = PHI[N_tr+N_vl:]

    Y1 = Y[:N_tr]
    Y2 = Y[N_tr:N_tr+N_vl]
    Y3 = Y[N_tr+N_vl:]

    # initialize arrays to collect data
    J = np.zeros((2,Nl1,Nl2))
    R2 = np.zeros_like(J)
    W = np.zeros((P,Nl1,Nl2))

    for i,l1 in enumerate(L1):
        for j,l2 in enumerate(L2):
            print(f"Elastic Ridge {l1},{l2}")
            # train Elastic Net Regression at l1,l2
            solve = lr.solve_gradient_descent(PHI1,Y1, eta=eta, epochs=epochs, lambda1=l1, lambda2=l2)
            # validate
            W1 = solve['W']
            Y_hat1 = lr.get_y_hat(PHI1,W1)
            Y_hat2 = lr.get_y_hat(PHI2,W1)
            J1 = lr.J(PHI2,W,Y2,Y_hat2)
            # throw results into collection containers
            J[0,i,j] = solve['J'][-1]
            J[1,i,j] = J1
            # throw results into R2
            R2[0,i,j] = lr.r_squared(Y1,Y_hat1)
            R2[1,i,j] = lr.r_squared(Y2,Y_hat2)
            # throw W1 into W
            W[:,i,j] = W1.reshape([P,])

    # find l1 and l2 that maximize R2 for validation
    # get the indicies that maximize R2 in validation
    max_ij = np.where( R2[1,:,:] == R2[1,:,:].max() )
    i,j = max_ij[0][0],max_ij[1][0]

    Y_hat3 = lr.get_y_hat(PHI3,W[:,i,j]).reshape([N_ts,1])
    J3 = lr.J(PHI3,W,Y3,Y_hat3, lambda1=L1[i], lambda2=L2[j])
    R2_testing = lr.r_squared(Y3,Y_hat3)

    # collect the optimal results
    results = dict(
            R2 = np.hstack( (R2[:,i,j],R2_testing) ),
            W = W[:,i,j],
            J = J,
            parameters = dict(eta=eta, epochs=epochs, lambda1=L1[i], lambda2=L2[j]),
            L1 = L1,
            L2 = L2,
            i = i,
            j = j,
            PHI3 = PHI3,
            Y3 = Y3
            )
    return results
