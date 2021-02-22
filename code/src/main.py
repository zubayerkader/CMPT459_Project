import pandas as pd
import matplotlib
import scipy.stats
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper1 import *

def cleanImpute(dir):
    cases_train = pd.read_csv(dir)
    cases_train = cases_train.head(10000)

    # remove columns: additional_information and source
    cases_train = cases_train.drop(['additional_information', 'source'], axis=1)

    # Clean and impute Age column
    age, age_mean, age_std = cleanAge(cases_train['age'].copy())
    cases_train["age"] = age
    cases_train["age"] = cases_train.apply(imputeAge, axis = 1, age_mean= age_mean, age_std = age_std)

    # Impute Sex
    cases_train = imputeSex(cases_train)

    # Remove rows with missing latitude longitude
    latlong = cases_train[["latitude", "longitude"]]
    empty_idx = latlong[latlong.isnull().any(axis =1)].index
    cases_train = cases_train.drop(empty_idx)  #might crash.........................................

    # Impute Country
    taiwan = cases_train[cases_train.province == 'Taiwan'].index
    cases_train['country'].iloc[taiwan] = 'Taiwan'

    # Impute Province
    nan_province = cases_train[cases_train.province.isnull()]
    nan_province['province'] = nan_province.apply(imputeProvince, axis=1, cases_train=cases_train)
    # print(nan_province[nan_province.province == 'unknown'].shape[0])
    cases_train['province'].iloc[nan_province.index] = nan_province['province']

    # clean date confirmation
    date_confirm = cases_train.date_confirmation.dropna()
    date_index = date_confirm[date_confirm.str.contains("-")].index
    for i in date_index:
        date = date_confirm[i]
        date = date.split('-')[0].strip()
        # print (date)
        cases_train['date_confirmation'].iloc[i] = date
    nan_confirm = cases_train[cases_train.date_confirmation.isnull()]
    nan_confirm['date_confirmation'] = nan_confirm.apply(imputeDateConfrm, axis=1, cases_train=cases_train)
    cases_train['date_confirmation'].iloc[nan_confirm.index] = nan_confirm['date_confirmation']

    print(cases_train)

def main():
    cases_train = cleanImpute('./../data/cases_train.csv')
    cases_test = cleanImpute('./../data/cases_test.csv')


if __name__ == "__main__":
    main()
