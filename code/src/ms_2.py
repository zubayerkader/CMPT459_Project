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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import datetime as dt
from tune_params import *
# def convert_to_number(data):
#     number = preprocessing.LabelEncoder()
#     data['outcome'] = number.fit_transform(data.outcome)
#     #data['outcome'] = number.inverse_transform(data.outcome)
#     #data['Source'] = number.fit_transform(data.Source)
#     data=data.fillna(-999)
#     return data
#
# def convert_back_to_string(data):
#     number = preprocessing.LabelEncoder()
#     #data['outcome'] = number.fit_transform(data.outcome)
#     data['outcome'] = number.inverse_transform(data.outcome)
#     #data['Source'] = number.fit_transform(data.Source)
#     data=data.fillna(-999)
#     return data

def evaluation(x,y):
    print("ACCURACY SCORE ",accuracy_score(x,y))
    print("PRECSION SCORE ",metrics.precision_score(x,y,average='macro'))
    print("RECALL SCORE ",metrics.recall_score(x,y,average='macro'))
    print("Confusing matrix ")
    print(confusion_matrix(x,y))

def main():
    df = pd.read_csv("../results/cases_train_processed.csv")
    # df = df.head(5000)
    df = df.loc[:, df.columns != 'province']
    df = df.loc[:, df.columns != 'country']
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'])
    df['date_confirmation'] =(df['date_confirmation'] - dt.datetime(2020,1,1)).dt.total_seconds()
    # df = df.loc[:, df.columns != 'date_confirmation']

    # df_2 = df.select_dtypes(include=[object])
    df_2 = df.loc[:, df.columns == 'sex']
    print (df_2)
    from sklearn import preprocessing
    # le = preprocessing.LabelEncoder()
    # df_3 = df_2.apply(le.fit_transform)
    # print (df_3)
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

    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

    X['outcome'] = y
    print (X)

    params = {
    	"n_neighbors": 5,
    	"n_neighbors_increment": 1,
    }
    tune_params(X, params, loops=10, model_name="KNeighborsClassifier")


































###---------------------------------------------------------------------------------ITRAZA
#
# #--------------------------------------------CHANE ENCODER------------------------------
#     le = preprocessing.LabelEncoder()
#     string_cols = ['sex','province','country','date_confirmation']
#     for col in string_cols:
#         X_train[col] = le.fit_transform(X_train[col])
#         X_validation[col] = le.fit_transform(X_validation[col])
# #--------------------------------------------CHANE ENCODER------------------------------
#
#
#
#     Y_train = le.fit_transform(Y_train)
#     Y_validation = le.fit_transform(Y_validation)
#
#     classifier = AdaBoostClassifier(n_estimators=100, random_state=0)
#     #classifier = RandomForestClassifier(n_estimators=1000,max_depth=4)
#     #classifier = KNeighborsClassifier(n_neighbors=3)
#     print("5 fold Cross-validation ",cross_val_score(classifier, X_train, Y_train, cv=5, scoring='recall_macro'))
#     classifier.fit(X_train,Y_train)
#     #print(classifier.score(X_validation,Y_validation))
#     Y_predict = classifier.predict(X_validation)
#
#     Y_validation = le.inverse_transform(Y_validation)
#     Y_predict = le.inverse_transform(Y_predict)
#
#     evaluation(Y_validation,Y_predict)

if __name__ == '__main__':
    main()
