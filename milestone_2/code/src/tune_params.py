import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def tune_params(data, params, loops, model_name):

	X = data.loc[:, data.columns != 'outcome']
	y = data['outcome']
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

	if model_name == "RandomForestClassifier":

		resultset = [key for key, value in params.items() if key.endswith("_increment") == False]
		resultset.append("training_score")
		resultset.append("validation_score")
		# print(resultset)

		plot_df = pd.DataFrame(columns=resultset)
		# print(plot_df)

		n_estimators = params["n_estimators"]
		max_depth = params["max_depth"]

		for i in range(loops):

			print ("Iteration: ", i)
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
			model.fit(X_train,Y_train)

			# print ("training score: ", model.score(X_train,Y_train))
			# print ("validation score: ", model.score(X_validation, Y_validation))
			Y_train_predict = model.predict(X_train)
			Y_validation_predict = model.predict(X_validation)
			plot_df = plot_df.append({
				"n_estimators": n_estimators,
				"max_depth": max_depth,
				"training_score": model.score(X_train,Y_train),
				"validation_score": model.score(X_validation, Y_validation),
				"training_precision": metrics.precision_score(Y_train,Y_train_predict,average='macro'),
				"validation_precision": metrics.precision_score(Y_validation,Y_validation_predict,average='macro'),
				"training_recall": metrics.recall_score(Y_train,Y_train_predict,average='macro'),
				"validation_recall": metrics.recall_score(Y_validation,Y_validation_predict,average='macro'),
				"training_confusion_matrix": confusion_matrix(Y_train,Y_train_predict),
				"validation_confusion_matrix": confusion_matrix(Y_validation,Y_validation_predict),
			}, ignore_index=True)

			n_estimators += params["n_estimators_increment"]
			max_depth += params["max_depth_increment"]
			print (plot_df)

		# print (plot_df)
		plot_df.to_csv("./plots/" + model_name + ".csv" , index=False)
		params_tuple = "(" + plot_df["n_estimators"].astype(str) + "," + plot_df["max_depth"].astype(str) + ")"
		print(params_tuple)

		plt.plot(list(params_tuple), plot_df["training_score"], '-r', label="train")
		plt.plot(list(params_tuple), plot_df["validation_score"], '-b', label="valid")
		plt.legend(loc="upper left")
		plt.xticks(rotation=90)
		plt.savefig("../plots/" + model_name + ".jpg")

	if model_name == "KNeighborsClassifier":


		resultset = [key for key, value in params.items() if key.endswith("_increment") == False]
		resultset.append("training_score")
		resultset.append("validation_score")
		# print(resultset)

		plot_df = pd.DataFrame(columns=resultset)
		# print(plot_df)

		n_neighbors = params["n_neighbors"]


		for i in range(loops):
			print ("Iteration: ", i)
			model = KNeighborsClassifier(n_neighbors=n_neighbors, weights = 'distance')
			model.fit(X_train,Y_train)

			# print ("training score: ", model.score(X_train,Y_train))
			# print ("validation score: ", model.score(X_validation, Y_validation))
			Y_train_predict = model.predict(X_train)
			Y_validation_predict = model.predict(X_validation)
			plot_df = plot_df.append({
				"n_neighbors": n_neighbors,
				"training_score": model.score(X_train,Y_train),
				"validation_score": model.score(X_validation, Y_validation),
				"training_precision": metrics.precision_score(Y_train,Y_train_predict,average='macro'),
				"validation_precision": metrics.precision_score(Y_validation,Y_validation_predict,average='macro'),
				"training_recall": metrics.recall_score(Y_train,Y_train_predict,average='macro'),
				"validation_recall": metrics.recall_score(Y_validation,Y_validation_predict,average='macro'),
				"training_confusion_matrix": confusion_matrix(Y_train,Y_train_predict),
				"validation_confusion_matrix": confusion_matrix(Y_validation,Y_validation_predict),
			}, ignore_index=True)

			n_neighbors += params["n_neighbors_increment"]
			print (plot_df)

		# print (plot_df)
		plot_df.to_csv("./plots/" + model_name + ".csv" , index=False)
		plt.plot(plot_df["n_neighbors"], plot_df["training_score"], '-r', label="train")
		plt.plot(plot_df["n_neighbors"], plot_df["validation_score"], '-b', label="valid")
		plt.legend(loc="upper left")
		plt.savefig("../plots/" + model_name + ".jpg")


	if model_name == "AdaBoostClassifier":

		resultset = [key for key, value in params.items() if key.endswith("_increment") == False]
		resultset.append("training_score")
		resultset.append("validation_score")
		# print(resultset)

		plot_df = pd.DataFrame(columns=resultset)
		# print(plot_df)

		n_estimators = params["n_estimators"]


		for i in range(loops):
			print ("Iteration: ", i)
			model = AdaBoostClassifier(n_estimators=n_estimators)
			model.fit(X_train,Y_train)

			# print ("training score: ", model.score(X_train,Y_train))
			# print ("validation score: ", model.score(X_validation, Y_validation))
			Y_train_predict = model.predict(X_train)
			Y_validation_predict = model.predict(X_validation)
			plot_df = plot_df.append({
				"n_estimators": n_estimators,
				"training_score": model.score(X_train,Y_train),
				"validation_score": model.score(X_validation, Y_validation),
				"training_precision": metrics.precision_score(Y_train,Y_train_predict,average='macro'),
				"validation_precision": metrics.precision_score(Y_validation,Y_validation_predict,average='macro'),
				"training_recall": metrics.recall_score(Y_train,Y_train_predict,average='macro'),
				"validation_recall": metrics.recall_score(Y_validation,Y_validation_predict,average='macro'),
				"training_confusion_matrix": confusion_matrix(Y_train,Y_train_predict),
				"validation_confusion_matrix": confusion_matrix(Y_validation,Y_validation_predict),
			}, ignore_index=True)

			n_estimators += params["n_estimators_increment"]
			print (plot_df)

		# print (plot_df)
		plot_df.to_csv("./plots/" + model_name + ".csv" , index=False)
		plt.plot(plot_df["n_estimators"], plot_df["training_score"], '-r', label="train")
		plt.plot(plot_df["n_estimators"], plot_df["validation_score"], '-b', label="valid")
		plt.legend(loc="upper left")
		plt.savefig("../plots/" + model_name + ".jpg")


# df = pd.read_csv("../results/cases_train_processed.csv")
# df = df.head(5000)
# df = df.drop(columns=["province", "country", "date_confirmation", "sex"])
# print(df)

# params = {
# 	"n_estimators": 10,
# 	"max_depth": 10,
# 	"n_estimators_increment": 10,
# 	"max_depth_increment": 0,
# }
# tune_params(df, params, loops=10, model_name="RandomForestClassifier")

# params = {
# 	"n_neighbors": 5,
# 	"n_neighbors_increment": 1,
# }
# tune_params(df, params, loops=10, model_name="KNeighborsClassifier")

# params = {
# 	"n_estimators": 5,
# 	"n_estimators_increment": 1,
# }
# tune_params(df, params, loops=10, model_name="AdaBoostClassifier")
