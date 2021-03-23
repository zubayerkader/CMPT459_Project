import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from tune_params import *


def Knn(df):
    df = df.loc[:, df.columns != 'province']
    df = df.loc[:, df.columns != 'country']
    print(df)

    df_2 = df.loc[:, df.columns == 'sex']
    print (df_2)
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_2)
    onehotlabels = enc.transform(df_2).toarray()
    print (onehotlabels)
    df['female'] = onehotlabels[:, 0]
    df['male'] = onehotlabels[:, 1]
    print (df)
    df = df.loc[:, df.columns != 'sex']
    X = df.loc[:, df.columns != 'outcome']
    y = df['outcome']

    # from sklearn.preprocessing import StandardScaler
    # std_scaler = StandardScaler()
    # X = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

    X['outcome'] = y
    print (X)

    params = {
    	"n_neighbors": 1,
    	"n_neighbors_increment": 1,
    }
    tune_params(X, params, loops=20, model_name="KNeighborsClassifier")

def Random_forest(df):
    le = preprocessing.LabelEncoder()
    string_cols = ['sex','province','country','outcome']
    for col in string_cols:
        df[col] = le.fit_transform(df[col])
    print(df)
    params = {
    	"n_estimators": 100,
    	"max_depth": 10,
        "n_estimators_increment":100,
        "max_depth_increment":0
    }
    tune_params(df, params, loops=10, model_name="RandomForestClassifier")
    # remember there is a le.inverse_transform(y) to get back the string outcome

def Ada_Boosting(df):
    le = preprocessing.LabelEncoder()
    string_cols = ['sex','province','country','outcome']
    for col in string_cols:
        df[col] = le.fit_transform(df[col])

    params = {
    	"n_estimators": 100,
        "n_estimators_increment":100,
    }
    tune_params(df, params, loops=10, model_name="AdaBoostClassifier")
    # remember there is a le.inverse_transform(y) to get back the string outcome

def main():
    df = pd.read_csv("../data/cases_train_processed.csv")
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'])
    df['date_confirmation'] =((df['date_confirmation'] - dt.datetime(2020,1,1)).dt.total_seconds())/(3600)
    print (df)
    # Random_forest(df)
    # Ada_Boosting(df)
    Knn(df)





if __name__ == '__main__':
    main()



# import pickle
#
# #
# # Create your model here (same as above)
# #
#
# # Save to file in the current working directory
# pkl_filename = "pickle_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)
#
# # Load from file
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)
#
# # Calculate the accuracy score and predict target values
# score = pickle_model.score(Xtest, Ytest)
# print("Test score: {0:.2f} %".format(100 * score))
# Ypredict = pickle_model.predict(Xtest)
