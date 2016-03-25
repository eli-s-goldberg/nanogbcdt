import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import RFECV

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from biokit.viz import corrplot
import os

DATABASES_BASEPATH = os.path.join(os.path.dirname(__file__), 'databases')
NATURAL_TRAINING_DATABASE_NAME_ = 'natural_training_data.csv'
TECHNICAL_TRAINING_DATABASE_NAME_ = 'technical_training_data.csv'
IMPORT_TRAINING_DATABASE_PATH = os.path.join(DATABASES_BASEPATH,
                                             'training_data')
IMPORT_TESTING_DATABASE_PATH = os.path.join(DATABASES_BASEPATH,
                                            'test_data')
OUTPUT_DATA_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), 'output')
ISOTOPE_LIST_ = ['107Ag', '109Ag', '139La', '140Ce', '141Pr', '143Nd',
                 '146Nd', '147Sm', '149Sm', '153Eu', '157Gd', '159Tb',
                 '182W', '206Pb', '208Pb', '232Th', '238U', '25Mg',
                 '55Mn', '59Co', '60Ni', '65Cu', '66Zn', '88Sr',
                 '90Zr', '93Nb', '95Mo']

BACKGROUND_ISOTOPES_ = ['25Mg', '55Mn']

# Set the initial parameters for GBC to use in the initial tuning round
GBC_INIT_PARAMS = {'loss': 'exponential', 'learning_rate': 0.3,
                   'min_samples_leaf': 50, 'n_estimators': 1000,
                   'random_state': None, 'max_features': 'auto'}

GBC_GRID_SEARCH_PARAMS = {'loss': ['exponential'],
                          'learning_rate': [0.01, 0.1, 0.3],
                          'min_samples_leaf': [50, 75, 100],
                          'random_state': [None],
                          'max_features': ['auto'],
                          'max_depth': [3]}  # note n_estimators automatically set


class GradientBoostingClassifierrWithCoef(GradientBoostingClassifier):
    def fit(self, *args, **kwargs):
        super(GradientBoostingClassifierrWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


def main(path='.', databases_filepath=DATABASES_BASEPATH, iterations=1,
         crossfolds=5,
         filter_negative=True, detection_threshold=True,
         threshold_value=0, isotope_trigger='140Ce',
         corrplot_training_show=True, max_n_estimators=1000):
    """
    Parameters
    ----------


    """

    os.chdir(IMPORT_TRAINING_DATABASE_PATH)

    training_files = glob.glob('*.csv')

    for file in training_files:
        if fnmatch.fnmatchcase(file, NATURAL_TRAINING_DATABASE_NAME_):
            natural_training_database = pd.DataFrame.from_csv(
                os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
                header=0, index_col=False)
            natural_training_database['Classification'] = 1
            print 'total number of natural particles pre threshold: ', \
                len(natural_training_database)

        elif fnmatch.fnmatchcase(file, TECHNICAL_TRAINING_DATABASE_NAME_):
            technical_training_database = pd.DataFrame.from_csv(
                os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
                header=0, index_col=False)
            technical_training_database['Classification'] = 0
            print 'total number of technincal particles pre threshold: ', \
                len(technical_training_database)
        else:
            print "database not found"

    training_data = pd.concat([natural_training_database, technical_training_database])
    training_data = training_data.dropna()
    training_data = training_data.drop(BACKGROUND_ISOTOPES_, axis=1)

    if filter_negative:
        for isotopes in list(training_data):
            training_data = training_data[training_data[isotopes] >= 0]

    if detection_threshold:
        training_data = training_data[training_data[isotope_trigger] >= threshold_value]

    if corrplot_training_show:
        c = corrplot.Corrplot(training_data).plot(
            upper='text', lower='circle', fontsize=8,
            colorbar=False, shrink=.9)
        plt.savefig('corrplot_rfecv_data.eps', format='eps')
        plt.show()

    print '# natural particles post threshold: ', \
        list(training_data['Classification']).count(1)

    print '# technical particles post threshold: ', \
        list(training_data['Classification']).count(0)

    target_data = training_data['Classification']
    training_data = training_data.drop('Classification', axis=1)

    # initialize gbc to determine max estimators with least overfitting
    gbc = GradientBoostingClassifier(**GBC_INIT_PARAMS)

    # change format for machine learning
    X = training_data.as_matrix()
    y = np.array(target_data)

    # determine minimum number of estimators with least overfitting
    x = np.arange(max_n_estimators) + 1
    test_score = heldout_score(gbc, X, y, max_n_estimators)

    # min loss according to test (normalize such that first loss is 0)
    test_score -= test_score[0]
    test_best_iter = x[np.argmin(test_score)]
    print "optimum number of boosting stages: ", test_best_iter

    GBC_GRID_SEARCH_PARAMS['n_estimators'] = test_best_iter

    for run in xrange(iterations):
        y_train = np.array(target_data)
        x_train = training_data.as_matrix()

        # assign the target data as y_all and the training data as x_all. Notice
        # that we train AND test on the same data. This is not commmon, but
        # we're employing the decision tree for a descriptive evaluation, not
        # its generic prediction performance

        if stratified_holdout:
            random_state = _SEED if deterministic else None
            sss = StratifiedShuffleSplit(
                y_train, n_iter=1, test_size=holdout_size,
                random_state=random_state)

            for train_index, test_index in sss:
                x_train, x_holdout = x_train[train_index], x_train[test_index]
                y_train, y_holdout = y_train[train_index], y_train[test_index]

            x_train_or_holdout = x_holdout
            y_train_or_holdout = y_holdout

            # if you want to seperate training data into holdout set to examine performance.
            x_train_or_holdout = x_train
            y_train_or_holdout = y_train

#     # --- controlling randomness, number of iterations, and number of cross-folds
#     SEED = 69  # always use a seed for randomized procedures
#     iterator1 = 1  # this is the number of model iterations to go through.
#     kNum = 5  # cross folds
#
#     # --- container initialization for speed ---
#     X_train = []
#     X_holdout = []
#     y_train = []
#     y_holdout = []
#     a = []
#     rfc_all_f1 = []
#     nameListAll = pd.DataFrame()
#     optimumLengthAll = pd.DataFrame()
#     classScoreAll = pd.DataFrame()
#     classScoreAll2 = pd.DataFrame()
#     classScoreAll3 = pd.DataFrame()
#     featureImportancesAll = pd.DataFrame()
#     rfecvGridScoresAll = pd.DataFrame()
#
#     # --- function definitions and class re-definitions ---
#     # Define a method for counting the number of particles in each bin.
#     def techVnatCount(target_feature):
#         import pandas as pd
#         target_feature = list(target_feature)
#         technicalCount = target_feature.count(0)
#         naturalCount = target_feature.count(1)
#         return (technicalCount, naturalCount)
#
#     # --- data import and screening ---
#     # Set technical training file path and import data from csv into a dataframe.
#     path = os.getcwd()
#
#     technicalFilePath = os.path.join(path, 'newData', '02_CeO2_technicalTraining0.csv')  # CSV Database File Name
#     technicalFile = pd.DataFrame.from_csv(technicalFilePath, header=0, sep=',', index_col=0, parse_dates=True,
#                                           encoding=None, tupleize_cols=False, infer_datetime_format=False)
#
#     # drop the background isotopes '27Al' and '138Ba' were removed previously
#     technicalFile.drop(['25Mg', '55Mn'], axis=1, inplace=True)
#
#     # Set natural training file path and import data from csv into a dataframe.
#     naturalFilePath = os.path.join(path, 'newData', '00_SPK0_naturalTraining0and1.csv')  # CSV Database File Name
#     naturalFile = pd.DataFrame.from_csv(naturalFilePath, header=0, sep=',', index_col=0, parse_dates=True,
#                                         encoding=None, tupleize_cols=False, infer_datetime_format=False)
#
#     # drop the background isotopes
#     naturalFile.drop(['25Mg', '55Mn'], axis=1, inplace=True)
#
#     # define an array of the remaining features to iterate through (can I delete this??)
#     listOfFeatures = list(naturalFile)
#
#     # add classification: 0 for the technical particle, 1 for natural particle
#     technicalFile['Classification'] = 0
#     naturalFile['Classification'] = 1
#
#     # Combine the technical and natural data files into a single dataframe
#     combinedData = pd.concat([technicalFile, naturalFile])
#     trainingData = combinedData.copy(deep=True)
#
#     # Apply the cerium thresholding ON!!!
#     trainingData = trainingData[trainingData['140Ce'] >= 0]
#
#     # Store the 'Classification' column as the target data set.
#     targetData = trainingData.Classification
#
#     # Drop the 'DwellTimeofNPEvent' and the target data set 'Classification'
#     trainingData = trainingData.drop(['Classification'], 1)
#
#     # Store the training data and target data as a matrices for import into ML.
#     trainingDataMatrix = trainingData.as_matrix()
#     targetDataMatrix = targetData.as_matrix()
#
#     # save the training data to a csv for inspection
#     trainingData.to_csv('./allIsotopes.csv')
#
#     # Determine the number of particles left in each class after the data filtering.
#     [technicalCount, naturalCount] = techVnatCount(targetDataMatrix)
#     print "%s Technical Particles " % technicalCount
#     print "%s Natural Particles " % naturalCount
#
#     # The Tech/nat split is unevenly weighted in favor of natural particles by ~4:1.
#     # Classification must include therefore include a stratified splitting to avoid skew during classification.
#
#     # Store the feature names by using the headers in the trainingData DataFrame.
#     feature_names = list(trainingData.columns.values)
#     print feature_names
#
#     # --- GBC initialization ---
#
#
#
#
#     # --- RFECV to automatically investigate the best conformation of features ---
#
#
#
#     # for 0 to the number of iterations specified above
#     for kk in range(0, iterator1):
#
#         # print the run number to keep track for long runs
#         print kk + 1
#
#         # Shuffle and split the dataset using a stratified approach to minimize the influence of class imbalance.
#         SSS = StratifiedShuffleSplit(targetDataMatrix, n_iter=1, test_size=0.20, random_state=SEED * kk)
#         for train_index, test_index in SSS:
#             X_train, X_holdout = trainingDataMatrix[train_index], trainingDataMatrix[test_index]
#             y_train, y_holdout = targetDataMatrix[train_index], targetDataMatrix[test_index]
#
#         # Call the RFECV function. Additional splitting is done by stratification shuffling and splitting
#         kfold = cross_validation.StratifiedKFold(y_train, n_folds=kNum, shuffle=True, random_state=None)
#
#         # Set the initial parameters for GBC to use in the initial tuning round
#         initParams = {'loss': 'exponential',
#                       'learning_rate': 0.1,
#                       'min_samples_leaf': 50,
#                       'n_estimators': 100,
#                       'random_state': None,
#                       'max_features': 'auto'}
#
#         # initialize GBC as rfc with initial params above
#         rfc = GradientBoostingClassifierrWithCoef(**initParams)
#
#         def heldout_score(clf, X_test, y_test):
#             """compute deviance scores on ``X_test`` and ``y_test``. """
#             clf.fit(X_test, y_test)
#             score = np.zeros((n_estimators,), dtype=np.float64)
#             for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
#                 score[i] = clf.loss_(y_test, y_pred)
#             return score
#
#         # determine minimum number of estimators with least overfitting
#         n_estimators = 1000  # choose a large amount of iterators to begin
#         x = np.arange(n_estimators) + 1
#         test_score = heldout_score(rfc, X_train, y_train)
#
#         # min loss according to test (normalize such that first loss is 0)
#         test_score -= test_score[0]
#         test_best_iter = x[np.argmin(test_score)]
#         print test_best_iter, "optimum number of iterations"
#
#         updatedParams = {'loss': 'exponential',
#                          'learning_rate': 0.1,
#                          'min_samples_leaf': 50,
#                          'n_estimators': test_best_iter,
#                          'random_state': None,
#                          'max_features': 'auto'}
#
#         # initialize GBC as rfc with initial params above
#         rfc = GradientBoostingClassifierrWithCoef(**updatedParams)
#
#         # Define RFECV function, can  use 'accuracy' or 'f1' f1_weighted, f1_macro
#         rfecv = RFECV(estimator=rfc, step=1, cv=kfold, scoring='f1_weighted')
#
#         # First, the recursive feature elimination model is trained. This fits to the optimum model and begins recursion.
#         rfecv = rfecv.fit(X_train, y_train)
#
#         # Second, the cross-validation scores are calculated such that grid_scores_[i] corresponds to the CV score
#         # of the i-th subset of features. In other words, from all the features to a single feature, the cross validation
#         # score is recorded.
#         rfecvGridScoresAll = rfecvGridScoresAll.append([rfecv.grid_scores_])
#
#         # Third, the .support_ attribute reports whether the feature remains after RFECV or not. The possible parameters are
#         # inspected by their ranking. Low ranking features are removed.
#         supPort = rfecv.support_  # True/False values, where true is a parameter of importance identified by recursive alg.
#         possParams = rfecv.ranking_
#         min_feature_params = rfecv.get_params(deep=True)
#         optimumLengthAll = optimumLengthAll.append([rfecv.n_features_])
#         featureSetIDs = list(supPort)
#         featureSetIDs = list(featureSetIDs)
#         feature_names = list(feature_names)
#         namedFeatures = list(trainingData.columns.values)
#         namedFeatures = np.array(namedFeatures)
#
#         # Loop over each item in the list of true/false values, if true, pull out the corresponding feature name and store
#         # it in the appended namelist. This namelist is rewritten each time, but the information is retained.
#         nameList = []  # Initialize a blank array to accept the list of names for features identified as 'True',
#         # or important.
#         for i in range(0, len(featureSetIDs)):
#             if featureSetIDs[i]:
#                 nameList.append(feature_names[i])
#             else:
#                 a = 1
#                 # print("didn't make it")
#                 # print(feature_names[i])
#         nameList = pd.DataFrame(nameList)
#         nameListAll = nameListAll.append(nameList)  # append the name list
#         nameList = list(nameList)
#         nameList = np.array(nameList)
#
#         # Fourth, the training process begins anew, with the objective to trim to the optimum feature and retrain the model
#         # without cross validation i.e., test the holdout set. The new training test set size for the holdout validation
#         # should be the entire 90% of the training set (X_trimTrainSet). The holdout test set also needs to be
#         # trimmed. The same transformation is performed on the holdout set (X_trimHoldoutSet).
#         X_trimTrainSet = rfecv.transform(X_train)
#         X_trimHoldoutSet = rfecv.transform(X_holdout)
#
#         # Fifth, no recursive feature elimination is needed (it has already been done and the poor features removed).
#         # Here the model is trained against the trimmed training set X's and corresponding Y's.
#         rfc.fit(X_trimTrainSet, y_train)
#
#         # Holdout test results are generated here.
#         preds = rfc.predict(
#             X_trimHoldoutSet)  # Predict the class from the holdout dataset. Previous call: rfecv.predict(X_holdout)
#         rfc_all_f1 = metrics.f1_score(y_holdout, preds, pos_label=None, average='weighted')  # determine the F1
#         rfc_all_f2 = metrics.r2_score(y_holdout, preds)  # determine the R^2 Score
#         rfc_all_f3 = metrics.mean_absolute_error(y_holdout,
#                                                  preds)  # determine the MAE - Do this because we want to determine sign.
#
#         classScoreAll = classScoreAll.append([rfc_all_f1])  # append the previous scores for aggregated analysis.
#         classScoreAll2 = classScoreAll2.append([rfc_all_f2])
#         classScoreAll3 = classScoreAll3.append([rfc_all_f3])
#         refinedFeatureImportances = rfc.feature_importances_  # determine the feature importances for aggregated analysis.
#         featureImportancesAll = featureImportancesAll.append([refinedFeatureImportances])
#         # append the previous scores for aggregated analysis
#
#     ## Output file creation
#     print("List of Important Features Identified by Recursive Selection Method:")
#     print(nameListAll)
#     nameListAll.to_csv('1_ifirs.csv')
#     nameListAll.count()
#
#     print("f1 weighted score for all runs:")
#     print(classScoreAll)
#     classScoreAll.to_csv('2_f1.csv')
#
#     print("R^2 score for all runs:")
#     print(classScoreAll2)
#     classScoreAll2.to_csv('3_r2.csv')
#
#     print("MAE score for all runs:")
#     print(classScoreAll3)
#     classScoreAll3.to_csv('4_mae.csv')
#
#     print("Optimal number of features:")
#     print(optimumLengthAll)
#     optimumLengthAll.to_csv('5_optimum_length.csv')
#
#     print("Selected Feature Importances:")
#     print(featureImportancesAll)
#     featureImportancesAll.to_csv('6_feature_importance.csv')
#
#     print("mean_squared_error Grid Score for Increasing Features")
#     print(rfecvGridScoresAll)
#     rfecvGridScoresAll.to_csv('7_rfecv_grid.csv')
#
#     ## Output file summary
#     fileName1 = 'f1_score_all.csv'
#     fileName2 = 'class_IFIRS.csv'
#     fileName3 = 'class_optimum_length.csv'
#     fileName4 = 'class_sel_feature_importances.csv'
#     fileName5 = 'class_rfecv_grid_scores.csv'
#
#
# if __name__ == '__main__':  # wrap inside to prevent parallelize errors on windows.
#     main(path='.', databases_filepath=DATABASES_BASEPATH,
#          filter_negative=True, detection_threshold=True,
#          threshold_value=0, isotope_trigger='140Ce',
#          corrplot_training_show=True, max_n_estimators=1000)
