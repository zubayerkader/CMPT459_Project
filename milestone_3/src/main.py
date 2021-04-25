import numpy as np
import pandas as pd
import datetime as dt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import metrics

def Knn_preprocess(df):
    df = df.loc[:, df.columns != 'province']
    df = df.loc[:, df.columns != 'country']
    df_2 = df.loc[:, df.columns == 'sex']
    enc = OneHotEncoder()
    enc.fit(df_2)
    onehotlabels = enc.transform(df_2).toarray()
    df['female'] = onehotlabels[:, 0]
    df['male'] = onehotlabels[:, 1]
    df = df.loc[:, df.columns != 'sex']
    X = df.loc[:, df.columns != 'outcome']
    y = df['outcome']
    min_max_scaler = MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
    X['outcome'] = y
    return X

def Knn_tuning(df):
    knn_data = Knn_preprocess(df)

    X = knn_data.loc[:, knn_data.columns != 'outcome']
    y = knn_data['outcome']

    # y = preprocessing.label_binarize(y, classes=['recovered', 'nonhospitalized', 'hospitalized', 'deceased'])
    # print(y)

    scoring = {
        'accuracy': make_scorer(metrics.accuracy_score),
        'recall_overall': 'recall_weighted',
        'recall_deceased': make_scorer(metrics.recall_score, labels=['deceased'], average=None),
        'f1_deceased': make_scorer(metrics.f1_score, labels=['deceased'], average=None),
        'precison': 'precision_weighted'
    }
    param_grid = {
        'n_neighbors': range(2, 15, 1),
        'weights': ['distance', 'uniform'],
        'p': [1,2]
    }
    gs = GridSearchCV(KNeighborsClassifier(),
                      param_grid=param_grid,
                      scoring=scoring, refit='recall_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("./results_knn.csv")

    pkl_filename = "../models/KNeighborsClassifier.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gs, file)

def Ada_Boosting_Random_forest_preprocess(df):
    le = LabelEncoder()
    string_cols = ['sex','province','country']      # ,'outcome'
    for col in string_cols:
        df[col] = le.fit_transform(df[col])

    return df

def Ada_Boosting_tuning(df):
    ada_data = Ada_Boosting_Random_forest_preprocess(df)

    X = ada_data.loc[:, ada_data.columns != 'outcome']
    y = ada_data['outcome']

    # y = preprocessing.label_binarize(y, classes=['recovered', 'nonhospitalized', 'hospitalized', 'deceased'])
    # print(y)

    scoring = {
        'accuracy': make_scorer(metrics.accuracy_score),
        'recall_overall': 'recall_weighted',
        'recall_deceased': make_scorer(metrics.recall_score, labels=['deceased'], average=None),
        'f1_deceased': make_scorer(metrics.f1_score, labels=['deceased'], average=None),
        'precison': 'precision_weighted'
    }
    param_grid = {
        'n_neighbors': range(2, 10, 2),
        'weights': ['distance', 'uniform']
    }

    param_grid = {
        "base_estimator__criterion" : ["gini", "entropy"],
        # "base_estimator__splitter" :   ["best", "random"],
        "base_estimator__max_depth": range(8, 22, 2),
        "n_estimators": [30,50,60]
    }

    DTC = DecisionTreeClassifier()
    ABC = AdaBoostClassifier(base_estimator = DTC)

    gs = GridSearchCV(ABC,
                      param_grid=param_grid,
                      scoring=scoring, refit='recall_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("./results_ada.csv")

    pkl_filename = "../models/AdaBoostClassifier.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gs, file)

def Random_forest_tuning(df):
    rf_data = Ada_Boosting_Random_forest_preprocess(df)

    X = rf_data.loc[:, rf_data.columns != 'outcome']
    y = rf_data['outcome']

    # y = preprocessing.label_binarize(y, classes=['recovered', 'nonhospitalized', 'hospitalized', 'deceased'])
    print(y)

    scoring = {
        'accuracy': make_scorer(metrics.accuracy_score),
        'recall_overall': 'recall_weighted',
        'recall_deceased': make_scorer(metrics.recall_score, labels=['deceased'], average=None),
        'f1_deceased': make_scorer(metrics.f1_score, labels=['deceased'], average=None),
        'precison': 'precision_weighted'
    }
    param_grid = {
        'n_estimators': range(10, 30, 2),
        'max_depth': range(10, 30, 2)
    }
    gs = GridSearchCV(RandomForestClassifier(),
                      param_grid=param_grid,
                      scoring=scoring, refit='recall_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("./results_rf.csv")

    pkl_filename = "../models/RandomForestClassifier.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gs, file)

def main():
    df = pd.read_csv("../data/cases_train_processed.csv")
    df = df.head(10000)
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'])
    df['date_confirmation'] = ((df['date_confirmation'] - dt.datetime(2020,1,1)).dt.total_seconds())/(3600)

    Knn_tuning(df)
    Random_forest_tuning(df)
    Ada_Boosting_tuning(df)

main()
