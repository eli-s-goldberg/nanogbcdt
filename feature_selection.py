#!/usr/bin/env python

'''
Notes:
1) if you have problems with matplotlib (i.e, RuntimeError: Python is not installed as a framework.), it's likely due
to matplotlib being installed by pip. Navigate to a directory in your (cmd+space and enter: ~/.matplotlib) and create
a text file ~/.matplotlib/matplotlibrc. The only text in the file shoudl be 'backend: TkAgg'. Save, exit, and rerun.
'''


# --- package import ---
import multiprocessing
import numpy as np
multiprocessing.cpu_count()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from biokit.viz import corrplot
import os


# --- controlling randomness, number of iterations, and number of cross-folds
SEED = 69  # always use a seed for randomized procedures
iterator1 = 1 # this is the number of model iterations to go through.
kNum = 5 # cross folds

# --- container initialization for speed ---
X_train = []
X_holdout = []
y_train = []
y_holdout = []
a = []
rfc_all_f1 = []
nameListAll = pd.DataFrame()
optimumLengthAll = pd.DataFrame()
classScoreAll = pd.DataFrame()
classScoreAll2 = pd.DataFrame()
classScoreAll3 = pd.DataFrame()
featureImportancesAll = pd.DataFrame()
rfecvGridScoresAll = pd.DataFrame()


# --- function definitions and class re-definitions ---
# Define a method for counting the number of particles in each bin.
def techVnatCount (target_feature):
    import pandas as pd
    target_feature= list(target_feature)
    technicalCount = target_feature.count(0)
    naturalCount = target_feature.count(1)
    return (technicalCount,naturalCount)

# Re-definition of the GBC to employ feature importance as a proxy for weighting to employ RFECV.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
class GradientBoostingClassifierrWithCoef(GradientBoostingClassifier):
    def fit(self, *args, **kwargs):
        super(GradientBoostingClassifierrWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


# --- data import and screening ---
# Set technical training file path and import data from csv into a dataframe.
path = os.getcwd()

technicalFilePath = os.path.join(path,'newData','02_CeO2_technicalTraining0.csv') # CSV Database File Name
technicalFile = pd.DataFrame.from_csv(technicalFilePath,header=0, sep=',', index_col=0, parse_dates=True,
                                   encoding=None,tupleize_cols=False, infer_datetime_format=False)

# drop the background isotopes '27Al' and '138Ba' were removed previously
technicalFile.drop(['25Mg','55Mn'], axis=1, inplace=True)

# Set natural training file path and import data from csv into a dataframe.
naturalFilePath = os.path.join(path,'newData','00_SPK0_naturalTraining0and1.csv')# CSV Database File Name
naturalFile = pd.DataFrame.from_csv(naturalFilePath,header=0, sep=',', index_col=0, parse_dates=True,
                                   encoding=None,tupleize_cols=False, infer_datetime_format=False)

# drop the background isotopes
naturalFile.drop(['25Mg','55Mn'], axis=1, inplace=True)

# define an array of the remaining features to iterate through (can I delete this??)
listOfFeatures = list(naturalFile)

# add classification: 0 for the technical particle, 1 for natural particle
technicalFile['Classification'] = 0
naturalFile['Classification'] = 1

# Combine the technical and natural data files into a single dataframe
combinedData = pd.concat([technicalFile,naturalFile])
trainingData = combinedData.copy(deep=True)

# Apply the cerium thresholding ON!!!
trainingData = trainingData[trainingData['140Ce']>=0]

# Store the 'Classification' column as the target data set.
targetData = trainingData.Classification

# Drop the 'DwellTimeofNPEvent' and the target data set 'Classification'
trainingData = trainingData.drop(['Classification'],1)

# Store the training data and target data as a matrices for import into ML.
trainingDataMatrix = trainingData.as_matrix()
targetDataMatrix = targetData.as_matrix()

# save the training data to a csv for inspection
trainingData.to_csv('./allIsotopes.csv')

# Determine the number of particles left in each class after the data filtering.
[technicalCount,naturalCount] = techVnatCount (targetDataMatrix)
print "%s Technical Particles " %technicalCount
print "%s Natural Particles " %naturalCount

# The Tech/nat split is unevenly weighted in favor of natural particles by ~4:1.
# Classification must include therefore include a stratified splitting to avoid skew during classification.

# Store the feature names by using the headers in the trainingData DataFrame.
feature_names = list(trainingData.columns.values)
print feature_names


# --- GBC initialization ---




# --- RFECV to automatically investigate the best conformation of features ---

# for 0 to the number of iterations specified above
for kk in range(0,iterator1):

    # print the run number to keep track for long runs
    print kk+1

    # Shuffle and split the dataset using a stratified approach to minimize the influence of class imbalance.
    SSS = StratifiedShuffleSplit(targetDataMatrix,n_iter=1,test_size=0.20,random_state=SEED*kk)
    for train_index,test_index in SSS:
        X_train, X_holdout = trainingDataMatrix[train_index],trainingDataMatrix[test_index]
        y_train, y_holdout = targetDataMatrix[train_index],targetDataMatrix[test_index]

    # Call the RFECV function. Additional splitting is done by stratification shuffling and splitting
    kfold = cross_validation.StratifiedKFold(y_train, n_folds=kNum, shuffle=True, random_state=None)

    # Set the initial parameters for GBC to use in the initial tuning round
    initParams = {'loss': 'exponential',
                  'learning_rate': 0.1,
                  'min_samples_leaf': 50,
                  'n_estimators': 100,
                  'random_state': None,
                  'max_features': 'auto'}

    # initialize GBC as rfc with initial params above
    rfc = GradientBoostingClassifierrWithCoef(**initParams)

    def heldout_score(clf, X_test, y_test):
        """compute deviance scores on ``X_test`` and ``y_test``. """
        clf.fit(X_test, y_test)
        score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            score[i] = clf.loss_(y_test, y_pred)
        return score

    # determine minimum number of estimators with least overfitting
    n_estimators = 1000 # choose a large amount of iterators to begin
    x = np.arange(n_estimators) + 1
    test_score = heldout_score(rfc, X_train, y_train)

    # min loss according to test (normalize such that first loss is 0)
    test_score -= test_score[0]
    test_best_iter = x[np.argmin(test_score)]
    print test_best_iter, "optimum number of iterations"

    updatedParams = {'loss': 'exponential',
                  'learning_rate': 0.1,
                  'min_samples_leaf': 50,
                  'n_estimators': test_best_iter,
                  'random_state': None,
                  'max_features': 'auto'}

    # initialize GBC as rfc with initial params above
    rfc = GradientBoostingClassifierrWithCoef(**updatedParams)

    # Define RFECV function, can  use 'accuracy' or 'f1' f1_weighted, f1_macro
    rfecv = RFECV(estimator=rfc, step=1, cv = kfold, scoring='f1_weighted')

    # First, the recursive feature elimination model is trained. This fits to the optimum model and begins recursion.
    rfecv = rfecv.fit(X_train, y_train)

    # Second, the cross-validation scores are calculated such that grid_scores_[i] corresponds to the CV score
    # of the i-th subset of features. In other words, from all the features to a single feature, the cross validation
    # score is recorded.
    rfecvGridScoresAll = rfecvGridScoresAll.append([rfecv.grid_scores_])

    # Third, the .support_ attribute reports whether the feature remains after RFECV or not. The possible parameters are
    # inspected by their ranking. Low ranking features are removed.
    supPort = rfecv.support_ # True/False values, where true is a parameter of importance identified by recursive alg.
    possParams = rfecv.ranking_
    min_feature_params = rfecv.get_params(deep=True)
    optimumLengthAll = optimumLengthAll.append([rfecv.n_features_])
    featureSetIDs = list(supPort)
    featureSetIDs = list(featureSetIDs)
    feature_names = list(feature_names)
    namedFeatures = list(trainingData.columns.values)
    namedFeatures = np.array(namedFeatures)

    # Loop over each item in the list of true/false values, if true, pull out the corresponding feature name and store
    # it in the appended namelist. This namelist is rewritten each time, but the information is retained.
    nameList = []   # Initialize a blank array to accept the list of names for features identified as 'True',
                    # or important.
    for i in range(0,len(featureSetIDs)):
        if featureSetIDs[i]:
            nameList.append(feature_names[i])
        else:
            a=1
            # print("didn't make it")
            # print(feature_names[i])
    nameList = pd.DataFrame(nameList)
    nameListAll = nameListAll.append(nameList) # append the name list
    nameList = list(nameList)
    nameList = np.array(nameList)

    # Fourth, the training process begins anew, with the objective to trim to the optimum feature and retrain the model
    # without cross validation i.e., test the holdout set. The new training test set size for the holdout validation
    # should be the entire 90% of the training set (X_trimTrainSet). The holdout test set also needs to be
    # trimmed. The same transformation is performed on the holdout set (X_trimHoldoutSet).
    X_trimTrainSet = rfecv.transform(X_train)
    X_trimHoldoutSet = rfecv.transform(X_holdout)

    # Fifth, no recursive feature elimination is needed (it has already been done and the poor features removed).
    # Here the model is trained against the trimmed training set X's and corresponding Y's.
    rfc.fit(X_trimTrainSet,y_train)

    # Holdout test results are generated here.
    preds = rfc.predict(X_trimHoldoutSet) # Predict the class from the holdout dataset. Previous call: rfecv.predict(X_holdout)
    rfc_all_f1 = metrics.f1_score(y_holdout,preds,pos_label=None,average = 'weighted') # determine the F1
    rfc_all_f2 = metrics.r2_score(y_holdout,preds) # determine the R^2 Score
    rfc_all_f3 = metrics.mean_absolute_error(y_holdout,preds) # determine the MAE - Do this because we want to determine sign.

    classScoreAll = classScoreAll.append([rfc_all_f1]) # append the previous scores for aggregated analysis.
    classScoreAll2 = classScoreAll2.append([rfc_all_f2])
    classScoreAll3 = classScoreAll3.append([rfc_all_f3])
    refinedFeatureImportances = rfc.feature_importances_ # determine the feature importances for aggregated analysis.
    featureImportancesAll = featureImportancesAll.append([refinedFeatureImportances])
    # append the previous scores for aggregated analysis

## Output file creation
print("List of Important Features Identified by Recursive Selection Method:")
print(nameListAll)
nameListAll.to_csv('class_IFIRS.csv')
nameListAll.count()

print("f1 weighted score for all runs:")
print(classScoreAll)
classScoreAll.to_csv('f1_score_all.csv')

print("R^2 score for all runs:")
print(classScoreAll2)
classScoreAll2.to_csv('class_Rsq_score_all.csv')

print("MAE score for all runs:")
print(classScoreAll3)
classScoreAll3.to_csv('class_MAE_score_all.csv')

print("Optimal number of features:")
print(optimumLengthAll)
optimumLengthAll.to_csv('class_optimum_length.csv')

print("Selected Feature Importances:")
print(featureImportancesAll)
featureImportancesAll.to_csv('class_sel_feature_importances.csv')

print("mean_squared_error Grid Score for Increasing Features")
print(rfecvGridScoresAll)
rfecvGridScoresAll.to_csv('class_rfecv_grid_scores.csv')

## Output file summary
fileName1 = 'f1_score_all.csv'
fileName2 = 'class_IFIRS.csv'
fileName3 = 'class_optimum_length.csv'
fileName4 = 'class_sel_feature_importances.csv'
fileName5 = 'class_rfecv_grid_scores.csv'