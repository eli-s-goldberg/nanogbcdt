import json
import os
import fnmatch
import glob
import numpy as np
from sklearn import grid_search
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from biokit.viz import corrplot
from helper_functions import heldout_score

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

CRITICAL_ISOTOPE_LIST_ = ['140Ce', '139La',
                          '88Sr']

MAX_N_ESTIMATORS_ = 1000
# Set the initial parameters for GBC to use in the initial tuning round
GBC_INIT_PARAMS = {'loss': 'deviance', 'learning_rate': 0.1,
                   'min_samples_leaf': 100, 'n_estimators': MAX_N_ESTIMATORS_, 'max_depth': 5,
                   'random_state': None, 'max_features': 'sqrt'}

GBC_GRID_SEARCH_PARAMS = {'loss': ['exponential', 'deviance'],
                          'learning_rate': [0.01, 0.1],
                          'min_samples_leaf': [50, 100],
                          'random_state': [None],
                          'max_features': ['sqrt', 'log2'],
                          'max_depth': [5]}  # note n_estimators automatically set


def main(path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         critical_isotope_list=CRITICAL_ISOTOPE_LIST_,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5, threshold_value=0,
         isotope_trigger=['140Ce'], corrplot_training_show=False, max_n_estimators=MAX_N_ESTIMATORS_,
         track_class_probabilities=[0.1, 0.1], output_summary_name='output_summary.csv'):
    NON_CRITICAL_ISOTOPES_ = set(ISOTOPE_LIST_) - set(critical_isotope_list)

    os.chdir(IMPORT_TRAINING_DATABASE_PATH)

    training_files = glob.glob('*.csv')

    for file in training_files:
        if fnmatch.fnmatchcase(file, NATURAL_TRAINING_DATABASE_NAME_):
            natural_training_database = pd.DataFrame.from_csv(
                os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
                header=0, index_col=None)
            natural_training_database['Classification'] = 1
        elif fnmatch.fnmatchcase(file, TECHNICAL_TRAINING_DATABASE_NAME_):
            technical_training_database = pd.DataFrame.from_csv(
                os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
                header=0, index_col=None)
            technical_training_database['Classification'] = 0
        else:
            print
            "database not found"

    training_data = pd.concat(
        [natural_training_database, technical_training_database])
    training_data = training_data.dropna()

    if filter_negative:
        training_data = training_data[training_data >= 0.0]

    if detection_threshold:
        training_data = training_data[training_data >= threshold_value]

    training_data = training_data.drop(NON_CRITICAL_ISOTOPES_, axis=1)

    if corrplot_training_show:
        corrplot.Corrplot(training_data).plot(
            upper='text', lower='circle', fontsize=8,
            colorbar=False, shrink=.9)
        plt.savefig('corrplot_training_data.eps', format='eps')
        plt.show()

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
    print
    "optimum number of boosting stages: ", test_best_iter

    GBC_GRID_SEARCH_PARAMS['n_estimators'] = [test_best_iter]

    # investigate the best possible set of parameters using a cross
    # validation loop and the given grid. The cross-validation does not do
    # random shuffles, but the estimator does use randomness (and
    # takes random_state via dpgrid).
    if perform_gridsearch:
        grid_searcher = grid_search.GridSearchCV(estimator=gbc,
                                                 cv=crossfolds,
                                                 param_grid=GBC_GRID_SEARCH_PARAMS,
                                                 n_jobs=-1)

        # call the grid search fit using the data
        grid_searcher.fit(X, y)

        # store and print the best parameters
        best_params = grid_searcher.best_params_
        print
        best_params

        gbc = GradientBoostingClassifier(**best_params)

    gbc.fit(X, y)

    # examine the testing database
    os.chdir(IMPORT_TESTING_DATABASE_PATH)
    test_files = glob.glob('*.csv')

    X_test_predicted_track = []
    X_test_predicted_proba_track = []
    # total_nat_above_proba_thresh_track = []
    # total_tech_above_proba_thresh_track = []
    X_test_data_track = pd.DataFrame()

    for test in test_files:
        run_name = str(test)
        # print run_name

        test_data = pd.read_csv(os.path.join(
            IMPORT_TESTING_DATABASE_PATH, test), header=0, index_col=0)
        test_data.reset_index(drop=True, inplace=True)

        if filter_negative:
            for isotopes in list(ISOTOPE_LIST_):
                test_data = test_data[test_data[isotopes] >= 0]

        if detection_threshold:
            for isotopes in isotope_trigger:
                test_data = test_data[test_data[isotopes] >= threshold_value]

        test_data = test_data.dropna()
        test_data = test_data.drop(NON_CRITICAL_ISOTOPES_, axis=1)
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

        # determine the number of natural particles with prediction proba < 90%
        if track_class_probabilities:
            class_proba = X_test_predicted_proba
            total_natural_by_proba = len(np.where(class_proba[:, 0] <= 0.5)[0])
            total_technical_by_proba = len(
                np.where(class_proba[:, 1] <= 0.5)[0])

            nat_above_thresh = len(
                np.where(class_proba[:, 0] <= track_class_probabilities[0])[0])
            tech_above_thresh = len(
                np.where(class_proba[:, 1] <= track_class_probabilities[1])[0])

            total_nat_above_proba_thresh = total_natural_by_proba - nat_above_thresh
            total_tech_above_proba_thresh = total_technical_by_proba - tech_above_thresh

        else:
            total_nat_above_proba_thresh = 0
            total_tech_above_proba_thresh = 0

        # keep track of particle counts in predictions
        X_test_nat_count = list(X_test_predicted).count(1).__float__()
        X_test_tec_count = list(X_test_predicted).count(0).__float__()

        # Organize and track data for table
        X_test_data = pd.DataFrame()
        X_test_data['run_name'] = [run_name]
        X_test_data['total_particle_count'] = [
            X_test_nat_count + X_test_tec_count]
        X_test_data['nat_particle_count'] = [X_test_nat_count]
        X_test_data['tec_particle_count'] = [X_test_tec_count]
        X_test_data['nat_above_proba_thresh'] = [total_nat_above_proba_thresh]
        X_test_data['tech_above_proba_thresh'] = [
            total_tech_above_proba_thresh]

        X_test_data_track = X_test_data_track.append(X_test_data)

        X_test_nat_proba = pd.DataFrame(X_test_predicted_proba)
        X_test_preserved['natural_class_proba'] = np.array(X_test_nat_proba[1])
        X_test_preserved.to_csv(os.path.join(OUTPUT_DATA_SUMMARY_PATH,
                                             'filtered_data', str('filtered_' + run_name[:-4] +
                                                                  output_summary_name)))

    X_test_data_track.to_csv(os.path.join(OUTPUT_DATA_SUMMARY_PATH, 'data_summaries',
                                          output_summary_name), index=False)


if __name__ == '__main__':
    CRITICAL_ISOTOPE_LIST_1 = ['140Ce']
    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_1,
         threshold_value=0,
         output_summary_name='summary_Ce_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_1,
         threshold_value=20,
         output_summary_name='summary_Ce_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    CRITICAL_ISOTOPE_LIST_2 = ['140Ce', '139La']
    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_2,
         threshold_value=0,
         output_summary_name='summary_Ce_La_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_2,
         threshold_value=20,
         output_summary_name='summary_Ce__La_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    CRITICAL_ISOTOPE_LIST_3 = ['140Ce', '139La', '88Sr']
    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_3,
         threshold_value=0,
         output_summary_name='summary_Ce_La_Sr_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_3,
         threshold_value=20,
         output_summary_name='summary_Ce__La_Sr_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    CRITICAL_ISOTOPE_LIST_4 = ['140Ce', '139La', '206Pb', '208Pb',
                               '88Sr', '90Zr', '66Zn', '107Ag']
    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_4,
         threshold_value=0,
         output_summary_name='summary_9_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_4,
         threshold_value=20,
         output_summary_name='summary_9_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    CRITICAL_ISOTOPE_LIST_5 = ['140Ce',
                               '139La',
                               '208Pb',
                               '88Sr',
                               '90Zr',
                               '206Pb',
                               '66Zn',
                               '65Cu',
                               '107Ag',
                               '182W',
                               '59Co',
                               '159Tb',
                               '143Nd',
                               '232Th',
                               '146Nd',
                               '141Pr',
                               '153Eu']

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_5,
         threshold_value=0,
         output_summary_name='summary_17_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_5,
         threshold_value=20,
         output_summary_name='summary_17_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    CRITICAL_ISOTOPE_LIST_6 = ISOTOPE_LIST_
    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_6,
         threshold_value=0,
         output_summary_name='summary_25_no_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=False, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])

    main(critical_isotope_list=CRITICAL_ISOTOPE_LIST_6,
         threshold_value=20,
         output_summary_name='summary_25_20_thresh.csv',
         path='.', databases_filepath=DATABASES_BASEPATH, filter_negative=False,
         perform_gridsearch=True, detection_threshold=True, crossfolds=5,
         isotope_trigger=['140Ce'], corrplot_training_show=False,
         track_class_probabilities=[0.1, 0.1])
