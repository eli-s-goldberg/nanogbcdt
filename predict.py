import json
import os
import fnmatch
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from biokit.viz import corrplot
from helper_functions import (heldout_score, techVnatCount)

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

CRITICAL_ISOTOPE_LIST_ = ['140Ce', '139La', '206Pb', '208Pb',
                          '88Sr', '90Zr', '66Zn', '107Ag']

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


def main(path='.', databases_filepath=DATABASES_BASEPATH,
         filter_negative=True, detection_threshold=True,
         threshold_value=0, isotope_trigger='140Ce',
         corrplot_training_show=True, max_n_estimators=1000):
    NON_CRITICAL_ISOTOPES_ = set(ISOTOPE_LIST_) - set(CRITICAL_ISOTOPE_LIST_)

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
    training_data = training_data.drop(NON_CRITICAL_ISOTOPES_, axis=1)

    if filter_negative:
        for isotopes in list(training_data):
            training_data = training_data[training_data[isotopes] >= 0]

    if detection_threshold:
        training_data = training_data[training_data[isotope_trigger] >= threshold_value]

    if corrplot_training_show:
        c = corrplot.Corrplot(training_data).plot(
            upper='text', lower='circle', fontsize=8,
            colorbar=False, shrink=.9)
        plt.savefig('corrplot_training_data.eps', format='eps')
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

    gbc.fit(X, y)

    # examine the testing database
    os.chdir(IMPORT_TESTING_DATABASE_PATH)
    test_files = glob.glob('*.csv')

    X_test_predicted_track = []
    X_test_predicted_proba_track = []
    X_test_data_track = pd.DataFrame()

    for test in test_files:
        run_name = str(test)

        test_data = pd.read_csv(os.path.join(
            IMPORT_TESTING_DATABASE_PATH, test), header=0, index_col=0)
        test_data.reset_index(drop=True, inplace=True)
        test_data = test_data.dropna()
        test_data = test_data.drop(NON_CRITICAL_ISOTOPES_, axis=1)

        if filter_negative:
            for isotopes in list(test_data):
                test_data = test_data[test_data[isotopes] >= 0]

        if detection_threshold:
            for isotopes in list(test_data):
                test_data = test_data[test_data[isotopes] >= threshold_value]

        # assign orignal data a new name for later analysis
        X_test_preserved = test_data.copy(deep=True)

        # change format for machine learning
        X_test = test_data.as_matrix()

        # use trained classifier to predict imported data
        X_test_predicted = gbc.predict(X_test)
        X_test_predicted_track.append(X_test_predicted)

        # use trained classifier to output trained probabilities
        X_test_predicted_proba = gbc.predict_proba(X_test)
        X_test_predicted_proba_track.append(X_test_predicted_proba)

        # keep track of particle counts in predictions
        X_test_nat_count = list(X_test_predicted).count(1).__float__()
        X_test_tec_count = list(X_test_predicted).count(0).__float__()

        # keep track of the ratio - if zero handle with 'pure' label
        try:
            X_test_tech_nat = X_test_tec_count / X_test_nat_count
        except ZeroDivisionError:
            X_test_tech_nat = 'pure'

        # Organize and track data for table
        X_test_data = pd.DataFrame()
        X_test_data['run_name'] = [run_name]
        X_test_data['total_particle_count'] = [X_test_nat_count + X_test_tec_count]
        X_test_data['nat_particle_count'] = [X_test_nat_count]
        X_test_data['tec_particle_count'] = [X_test_tec_count]
        X_test_data['tec_to_nat_ratio'] = [X_test_tech_nat]
        X_test_data_track = X_test_data_track.append(X_test_data)

        X_test_nat_proba = pd.DataFrame(X_test_predicted_proba)
        X_test_preserved['natural_class_proba'] = np.array(X_test_nat_proba[1])

    X_test_data_track.to_csv(os.path.join(OUTPUT_DATA_SUMMARY_PATH, 'summary.csv'),
                             index = False)


if __name__ == '__main__':  # wrap inside to prevent parallelize errors on windows.
    main(path='.', databases_filepath=DATABASES_BASEPATH,
         filter_negative=True, detection_threshold=True,
         threshold_value=0, isotope_trigger='140Ce',
         corrplot_training_show=True, max_n_estimators=1000)
