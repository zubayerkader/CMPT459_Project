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

def analyze(input_file, results_file, best_hp_file):

    # with open(pkl_file, 'rb') as file:
    #     gs = pickle.load(file)
    #
    # results = gs.cv_results_
    # results_df = pd.DataFrame(results)
    # print(results_df)

    results_df = pd.read_csv(input_file)

    # results_df = results_df[["params", "mean_train_accuracy", "mean_test_accuracy", "mean_train_recall_overall", "mean_test_recall_overall", "mean_train_f1_deceased", "mean_test_f1_deceased", "mean_train_recall_deceased", "mean_test_recall_deceased", "mean_train_precison", "mean_test_precison" ]]
    # results_df.columns = ["params", "train_accuracy", "test_accuracy", "train_recall_overall", "test_recall_overall", "train_f1_deceased", "test_f1_deceased", "train_recall_deceased", "test_recall_deceased", "train_precison", "test_precison" ]

    results_df = results_df[["params", "mean_test_accuracy", "mean_test_recall_overall", "mean_test_f1_deceased", "mean_test_recall_deceased", "mean_test_precison" ]]
    results_df.columns = ["params", "test_accuracy", "test_recall_overall", "test_f1_deceased", "test_recall_deceased", "test_precison" ]
    results_df.to_csv(results_file)

    # final_df = pd.DataFrame(columns = ["params", "train_accuracy", "test_accuracy", "train_recall_overall", "test_recall_overall", "train_f1_deceased", "test_f1_deceased", "train_recall_deceased", "test_recall_deceased", "train_precison", "test_precison" ])
    final_df = pd.DataFrame(columns = ["params", "test_accuracy", "test_recall_overall", "test_f1_deceased", "test_recall_deceased", "test_precison" ])
    check_cols = [ "test_accuracy",  "test_recall_overall",  "test_f1_deceased",  "test_recall_deceased",  "test_precison" ]
    for attr in check_cols:
        sorted = (results_df.sort_values(by=[attr], ascending=False)).head(3)
        # print(sorted)
        final_df= final_df.append(sorted)

    final_df["params"] = final_df.params.apply(lambda x: str(x))
    final_df= final_df.drop_duplicates().sort_values(by=["test_f1_deceased"], ascending=False)
    # print(results_df.iloc[results_df['test_f1_deceased'].argmax()])
    final_df.to_csv(best_hp_file)
    return final_df

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
                      scoring=scoring, refit='f1_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("../results/results_knn.csv")

    # pkl_filename = "../models/KNeighborsClassifier.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(gs, file)

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
        "base_estimator__criterion" : ["gini", "entropy"],
        #### "base_estimator__splitter" :   ["best", "random"],
        # "base_estimator__max_depth": range(2, 30, 2),
        # "n_estimators": [10,20,30,40,50,60]
        "base_estimator__max_depth": range(8, 22, 2),
        "n_estimators": [30,50,60]
    }

    DTC = DecisionTreeClassifier()
    ABC = AdaBoostClassifier(base_estimator = DTC)

    gs = GridSearchCV(ABC,
                      param_grid=param_grid,
                      scoring=scoring, refit='f1_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("../results/results_ada.csv")

    # pkl_filename = "../models/AdaBoostClassifier.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(gs, file)

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
        # 'n_estimators': range(2, 30, 2),
        # 'max_depth': range(2, 30, 2)
    }
    gs = GridSearchCV(RandomForestClassifier(),
                      param_grid=param_grid,
                      scoring=scoring, refit='f1_deceased', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv("../results/results_rf.csv")

    # pkl_filename = "../models/RandomForestClassifier.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(gs, file)


def check_if_file_valid(filename):
    assert filename.endswith('predictions.txt'), 'Incorrect filename'
    f = open(filename).read()
    l = f.split('\n')
    # print (l)
    assert len(l) == 46500, 'Incorrect number of items'
    assert (len(set(l)) == 4), 'Wrong class labels'
    return 'Thepredictionsfile is valid'

def main():
    df = pd.read_csv("../data/cases_train_processed.csv")
    # df = df.head(90000)
    df['date_confirmation'] = pd.to_datetime(df['date_confirmation'])
    df['date_confirmation'] = ((df['date_confirmation'] - dt.datetime(2020,1,1)).dt.total_seconds())/(3600)

    print("Performing Grid Search For Random Forest")
    Random_forest_tuning(df)
    print("Performing Grid Search For Ada Boosting")
    Ada_Boosting_tuning(df)
    print("Performing Grid Search For KNN")
    Knn_tuning(df)

    results_rf_short = analyze("../results/results_rf.csv", "../results/results_rf_short.csv", "../results/best_hp_rf.csv")
    results_ada_short = analyze("../results/results_ada.csv", "../results/results_ada_short.csv", "../results/best_hp_ada.csv")
    results_knn_short = analyze("../results/results_knn.csv", "../results/results_knn_short.csv", "../results/best_hp_knn.csv")

    #reading prediction file
    test_df = pd.read_csv("../data/cases_test_processed.csv")
    test_df['date_confirmation'] = pd.to_datetime(test_df['date_confirmation'])
    test_df['date_confirmation'] = ((test_df['date_confirmation'] - dt.datetime(2020,1,1)).dt.total_seconds())/(3600)
    test_df = Ada_Boosting_Random_forest_preprocess(test_df)

    #use appropriate preprocessing
    training_data = Ada_Boosting_Random_forest_preprocess(df)
    X = training_data.loc[:, training_data.columns != 'outcome']
    y = training_data['outcome']

    # prediction using best hyperparameter
    # base = DecisionTreeClassifier(max_depth=2, criterion="gini")
    # model = AdaBoostClassifier(base_estimator=base, n_estimators=10)
    model = RandomForestClassifier(n_estimators=2, max_depth=24)
    model.fit(X,y)

    predictions = model.predict(test_df)
    with open('predictions.txt', 'w') as filehandle:
        for i in range(len(predictions)-1):
            filehandle.write(str(predictions[i])+"\n")
        filehandle.write(str(predictions[len(predictions)-1]))

    check_if_file_valid("../results/predictions.txt")

main()
