import os
import fnmatch
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from biokit.viz import corrplot
from sklearn.model_selection import GridSearchCV

# from helper_functions import heldout_score

GBC_GRID_SEARCH_PARAMS = {'loss': ['exponential', 'deviance'],
                          'learning_rate': [0.01, 0.1],
                          'min_samples_leaf': [50, 100],
                          'random_state': [None],
                          'max_features': ['sqrt', 'log2'],
                          'max_depth': [5]}  # note n_estimators automatically set


class distinguish_nat_vs_tech():
    def __init__(self,
                 training_data=[],
                 target_data=[]):
        self.training_data = training_data
        self.target_data = target_data

    def filter_negative(self, data):
        data = data[data >= 0].dropna()
        return data

    def apply_detection_threshold(self, data,
                                  isotope_trigger='140Ce',
                                  threshold_value=0):
        data = data[data[isotope_trigger] >= threshold_value].dropna()
        self.threshold_value = threshold_value
        return data

    def show_correlation(self):
        corrplot.Corrplot(self.training_data).plot(
            upper='text',
            lower='circle',
            fontsize=8,
            colorbar=False,
            shrink=.9)

        plt.savefig('corrplot_training_data.eps', format='eps')
        plt.show()

    def conform_data_for_ML(self, training_data, target_data):
        X = training_data.as_matrix()
        y = np.array(target_data)
        return X, y

    def set_training_target_data(self, X, y):
        self.X = X
        self.y = y
        return self

    def split_target_from_training_data(self, data):
        self.target_data = data['Classification']
        self.training_data = data.drop('Classification', axis=1)
        return self

    def find_min_boosting_stages(self, gbc_base_params):
        def heldout_score(clf, X_test, y_test, max_n_estimators):
            """compute deviance scores on ``X_test`` and ``y_test``. """
            clf.fit(X_test, y_test)
            score = np.zeros((max_n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                score[i] = clf.loss_(y_test, y_pred)
            return score

        # determine minimum number of estimators with least overfitting
        gbc = GradientBoostingClassifier(**gbc_base_params)
        self.gbc_base = gbc

        x_range = np.arange(gbc_base_params['n_estimators'] + 1)
        test_score = heldout_score(gbc, self.X, self.y, gbc_base_params['n_estimators'])

        # min loss according to test (normalize such that first loss is 0)
        test_score -= test_score[0]
        test_best_iter = x_range[np.argmin(test_score)]
        self.optimum_boosting_stages = test_best_iter
        return self

    def find_optimum_gbc_parameters(self, crossfolds=5, method='perform_search',
                                    gbc_base=[],
                                    gbc_search_params=[]):
        gbc = self.gbc_base

        if method == 'perform_search':
            grid_searcher = GridSearchCV(estimator=gbc_base,
                                         cv=crossfolds,
                                         param_grid=gbc_search_params,
                                         n_jobs=-1)

            # call the grid search fit using the data
            grid_searcher.fit(self.X, self.y)

            # store and print the best parameters
            self.gbc_best_params = grid_searcher.best_params_

            # re-initialize classifier with best params and fit
            if self.optimum_boosting_stages:
                self.gbc_best_params['n_estimators'] = self.optimum_boosting_stages

            gbc = GradientBoostingClassifier(**self.gbc_best_params)

        gbc_fitted = gbc.fit(self.X, self.y)
        return gbc_fitted

    def track_class_probabilitie(self):
        pass
        
    def apply_trained_classification(self,
                                     gbc = [],
                                     test_data_path='',
                                     filter_neg=True,
                                     apply_threshold=True,
                                     critical_isotopes=[]):
        X_test_predicted_track = []
        X_test_predicted_proba_track = []
        X_test_data_track = pd.DataFrame()
        os.chdir(test_data_path)
        test_data_names = glob.glob('*.csv')

        for test in test_data_names:
            run_name = str(test)
            test_data = pd.read_csv(os.path.join(
                test_data_path, test), header=0, index_col=0)
            test_data.reset_index(drop=True, inplace=True)

            if filter_neg:
                test_data = self.filter_negative(data=test_data)

            if apply_threshold:
                test_data = self.apply_detection_threshold(data=test_data)

            if critical_isotopes:
                non_crit_isotopes = set(list(self.training_data)) - set(critical_isotopes)
                test_data = test_data.drop(non_crit_isotopes, axis=1)

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

        print(test_data)
        #     test_data = test_data.dropna()
        #
        #     # assign orignal data a new name for later analysis
        #     X_test_preserved = test_data.copy(deep=True)
        #
        #     # change format for machine learning
        #     X_test = test_data.as_matrix()
        #
        #     # use trained classifier to predict imported data
        #     X_test_predicted = gbc.predict(X_test)
        #     X_test_predicted_track.append(X_test_predicted)
        #
        #     # use trained classifier to output trained probabilities
        #     X_test_predicted_proba = gbc.predict_proba(X_test)
        #     X_test_predicted_proba_track.append(X_test_predicted_proba)
        #
        #     # determine the number of natural particles with prediction proba < 90%
        #     if track_class_probabilities:
        #         class_proba = X_test_predicted_proba
        #         total_natural_by_proba = len(np.where(class_proba[:, 0] <= 0.5)[0])
        #         total_technical_by_proba = len(
        #             np.where(class_proba[:, 1] <= 0.5)[0])
        #
        #         nat_above_thresh = len(
        #             np.where(class_proba[:, 0] <= track_class_probabilities[0])[0])
        #         tech_above_thresh = len(
        #             np.where(class_proba[:, 1] <= track_class_probabilities[1])[0])
        #
        #         total_nat_above_proba_thresh = total_natural_by_proba - nat_above_thresh
        #         total_tech_above_proba_thresh = total_technical_by_proba - tech_above_thresh
        #
        #     else:
        #         total_nat_above_proba_thresh = 0
        #         total_tech_above_proba_thresh = 0
        #
        #     # keep track of particle counts in predictions
        #     X_test_nat_count = list(X_test_predicted).count(1).__float__()
        #     X_test_tec_count = list(X_test_predicted).count(0).__float__()
        #
        #     # Organize and track data for table
        #     X_test_data = pd.DataFrame()
        #     X_test_data['run_name'] = [run_name]
        #     X_test_data['total_particle_count'] = [
        #         X_test_nat_count + X_test_tec_count]
        #     X_test_data['nat_particle_count'] = [X_test_nat_count]
        #     X_test_data['tec_particle_count'] = [X_test_tec_count]
        #     X_test_data['nat_above_proba_thresh'] = [total_nat_above_proba_thresh]
        #     X_test_data['tech_above_proba_thresh'] = [
        #         total_tech_above_proba_thresh]
        #
        #     X_test_data_track = X_test_data_track.append(X_test_data)
        #
        #     X_test_nat_proba = pd.DataFrame(X_test_predicted_proba)
        #     X_test_preserved['natural_class_proba'] = np.array(X_test_nat_proba[1])
        #     X_test_preserved.to_csv(os.path.join(OUTPUT_DATA_SUMMARY_PATH,
        #                                          'filtered_data', str('filtered_' + run_name[:-4] +
        #                                                               output_summary_name)))
        #
        # X_test_data_track.to_csv(os.path.join(OUTPUT_DATA_SUMMARY_PATH, 'data_summaries',
        #                                       output_summary_name), index=False)


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

os.chdir(IMPORT_TRAINING_DATABASE_PATH)
training_files = glob.glob('*.csv')
print(training_files)

for file in training_files:
    if fnmatch.fnmatchcase(file, TECHNICAL_TRAINING_DATABASE_NAME_):
        technical_training_database = pd.DataFrame.from_csv(
            os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
            header=0, index_col=None)

        # assign technical ID as = 0
        technical_training_database['Classification'] = 0

    elif fnmatch.fnmatchcase(file, NATURAL_TRAINING_DATABASE_NAME_):
        natural_training_database = pd.DataFrame.from_csv(
            os.path.join(IMPORT_TRAINING_DATABASE_PATH, file),
            header=0, index_col=None)

        # assign natural ID as = 1
        natural_training_database['Classification'] = 1

training_data = pd.concat([natural_training_database, technical_training_database])
training_data = training_data.dropna()

nat_v_tech = distinguish_nat_vs_tech()
neg_filt_training_data = nat_v_tech.filter_negative(data=training_data)
thresh_neg_filt_training_data = nat_v_tech.apply_detection_threshold(data=neg_filt_training_data, threshold_value=5)
nat_v_tech.split_target_from_training_data(data=thresh_neg_filt_training_data)
ML_data = nat_v_tech.conform_data_for_ML(training_data=nat_v_tech.training_data, target_data=nat_v_tech.target_data)

nat_v_tech.set_training_target_data(X=ML_data[0], y=ML_data[1])

# initialize gbc to determine max estimators with least overfitting
GBC_INIT_PARAMS = {'loss': 'deviance', 'learning_rate': 0.1,
                   'min_samples_leaf': 100, 'n_estimators': 1000,
                   'max_depth': 5, 'random_state': None, 'max_features': 'sqrt'}

nat_v_tech.find_min_boosting_stages(gbc_base_params=GBC_INIT_PARAMS)
print(nat_v_tech.optimum_boosting_stages)

gbc_fitted = nat_v_tech.find_optimum_gbc_parameters(gbc_base=nat_v_tech.gbc_base,
                                                    method=False,
                                                    gbc_search_params=GBC_GRID_SEARCH_PARAMS)
nat_v_tech.apply_trained_classification(test_data_path=IMPORT_TESTING_DATABASE_PATH,
                                        gbc= gbc_fitted,
                                        critical_isotopes=False)
