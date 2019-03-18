# import dependencies
import numpy as np
import pandas as pd
import requests
import zipfile
import io
import os
import pdb
# import from my library
from DataScience.LinearRegression import LinearRegression

# make instance of LinearRegression
lr = LinearRegression()

def extract_data(startYear=2012):
    """
    """
    print("\nstarting data extraction from the web...\n")

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
    append_sale = kwargs['append_sale'] if 'append_sale' in kwargs else True
    append_real = kwargs['append_real'] if 'append_real' in kwargs else True

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
    if not os.path.isfile("sale.csv"): append_data_by_source(append_real=False)
    print("\ntransforming sale data...\n")
    # load file to DataFrame
    sale = pd.read_csv("sale.csv", index_col="Unnamed: 0", low_memory=False)

    # if 'real_csv' doesn't exist, create it
    if not os.path.isfile("sale.csv"): append_data_by_source(append_sale=False)
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

    # normalize
    for col in joinDF.keys():
        # check the column
        uniques = joinDF[col].unique()
        print(joinDF.shape)
        assert uniques.shape[0] > 1, f"since {col} is single valued, just drop it!"
        if uniques.shape[0] == 2: continue
        # normalize column
        try:
            x_j = joinDF[col].astype(float)
            min, max = x_j.min(), x_j.max()
            joinDF[col] = (x_j - min) / (max - min)
        except ValueError:
            print(f"couldn't convert {col} to numbers!")

    return joinDF

def make_PHI(**kwargs):
    """
    """
    # grab the cleaned, joined DataFrame
    cleanDF = kwargs['clean_data'] if 'clean_data' in kwargs else clean_and_transform(**kwargs)

    # return design matrix
    return np.column_stack( (np.ones((cleanDF.shape[0],1)), cleanDF.values) )
