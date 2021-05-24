COVID Outcome Predictor Jan – April 2021
Data Mining, CMPT 459, SFU
o Two different datasets were used to train machine-learning models to predict the medical outcome of a COVID patient.
o Exploratory data analysis was performed using histograms and box plots on the datasets to understand the data distribution and detect outliers.
o The datasets were cleaned and missing values were imputed based on their specific distributions or modes. The 2 datasets were joined based on closed location.
o Grid Search parameter tuning with 5-fold cross-validation was performed to tune parameters for K-nearest-neighbor, Random Forest and ADA Boost classifiers.
o A model of Random Forest was trained on the training dataset using the best parameters. The model was tested on an unseen test dataset to achieve an accuracy of 81% over all outcomes.
