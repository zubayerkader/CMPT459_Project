import pandas as pd
import scipy.stats
import numpy as np
import re
import random

def cleanAge(age):

    nan_count = age.isna().sum()

    age = pd.DataFrame(age)

    for index, r in age.iterrows():
        row = r[0]

        if r.isnull().values.any():
            pass
        elif re.match('\d+ *- *\d+', row): #re.match('[0-9][0-9]-[0-9][0-9]', row) or re.match('[0-9]-[0-9][0-9]', row) or re.match('[0-9]-[0-9]', row) or re.match('[0-9][0-9] - [0-9][0-9]', row): # '54-56'
            val = str(row).split('-')
            avg = (int(val[0])+int(val[1]))/2
            age.loc[index] = int(avg)
        elif re.match('\d+ *months*', row): #re.match('[0-9] month', row): # '8 month'
            val = str(row).split('month')
            val = float(val[0].strip())
            age.loc[index] = int(val/12)
        elif re.match('[0-9][0-9][-|+]$', row): # '80+', '80-'
            age.loc[index] = int(row[0]+row[1])
        else: # converts floats to ints for all other cases
            try:
                age.loc[index] = int(float(row))
            except:
                age.loc[index] = None

    new_age_col_unimputed = age.copy()
    age = age.dropna()
    mean = age.mean()
    std = age.std()

    return [new_age_col_unimputed, mean, std]

def imputeAge(val, age_mean, age_std):
    if pd.isna(val.age):
        ret = np.random.normal(age_mean, age_std, 1)
        return int(ret[0])
    else:
        return val.age

def imputeSexValue(row, male_probability):
    r = random.uniform(0, 1)
    if (r < male_probability):
        return 'male'
    else:
        return 'female'

def imputeSex(cases_train):
    sex = cases_train.sex
    sex = sex.dropna()
    male_count = sex[sex == 'male'].shape[0]
    female_count = sex[sex == 'female'].shape[0]
    total = sex.shape[0]
    male_p = male_count/total
    female_p = female_count/total

    cases_train['sex'] = cases_train.apply(imputeSexValue, axis=1, male_probability=male_p)

    return cases_train

def imputeProvince(row, cases_train):
    country = row['country']
    country_table = cases_train[cases_train.country == country]
    province = country_table.province.mode()
    if province.shape[0] == 0:
        return "unknown"
    else:
        return province

def imputeDateConfrm(row, cases_train):
    # dc = row['date_confirmation']
    country = row['country']
    country_table = cases_train[cases_train.country == country]
    dc = country_table.date_confirmation.mode()
    if dc.shape[0] == 0:
        return "unknown"
    else:
        return dc
