import pandas as pd
import matplotlib
import scipy.stats
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper1 import *

def cleanImpute(dir, out):# pp = 0):
    cases_train = pd.read_csv(dir)

    # print ("!!!!!!!!!!!!!!",  cases_train.shape[0])
    # cases_train = cases_train.head(10000)

    # remove columns: additional_information and source
    cases_train = cases_train.drop(['additional_information', 'source'], axis=1)

    # Clean and impute Age column
    age, age_mean, age_std = cleanAge(cases_train['age'].copy())
    cases_train["age"] = age
    cases_train["age"] = cases_train.apply(imputeAge, axis = 1, age_mean= age_mean, age_std = age_std)

    # Impute Sex
    cases_train = imputeSex(cases_train)



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

    # Remove rows with missing latitude longitude
    latlong = cases_train[["latitude", "longitude"]]
    empty_idx = latlong[latlong.isnull().any(axis =1)].index
    print(empty_idx)
    cases_train = cases_train.drop(empty_idx)  #might crash.........................................

    print(cases_train)
    cases_train.to_csv(out,index=False)
    return cases_train
    # if pp == 1:
    #     ppp = cases_train.drop(['outcome'], axis=1)
    # else:
    #     ppp = cases_train
    # ppp.dropna()
    # print ("!!!!!!!!!!!!!!",  ppp.shape[0])

def main():
    # cases_train = cleanImpute('../data/cases_train.csv', '../results/cases_train_processed.csv')
    # cases_test = cleanImpute('../data/cases_test.csv', '../results/cases_test_processed.csv')#, pp= 1)

    cases_train = pd.read_csv('../results/cases_train_processed.csv')
    cases_test = pd.read_csv('../results/cases_test_processed.csv')

    probable_outliers = cases_train[cases_train['longitude'].between(-40, -20)]
    probable_outliers = probable_outliers[probable_outliers['latitude'].between(-40,-20)]
    probable_outliers_idx = probable_outliers.index
    for i in probable_outliers_idx:
        province = probable_outliers['province'][i]
        country = probable_outliers['country'][i]
        match_area = cases_train[((cases_train['province'] == province) & (cases_train['country'] == country))]
        lat_m = match_area['latitude'].mean()
        long_m = match_area['longitude'].mean()
        cases_train["latitude"].iloc[i] = lat_m
        cases_train["longitude"].iloc[i] = long_m

    # print(cases_train.iloc[probable_outliers.index])

if __name__ == "__main__":
    main()
