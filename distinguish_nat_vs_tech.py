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

    def filter_negative(self, training_data):
        self.training_data = training_data[training_data >= 0].dropna()
        return self

    def apply_detection_threshold(self, training_data,
                                  isotope_trigger='140Ce',
                                  threshold_value=0):
        self.training_data = training_data[training_data[isotope_trigger] >= threshold_value].dropna()
        return self

    def show_correlation(self):
        corrplot.Corrplot(self.training_data).plot(
            upper='text',
            lower='circle',
            fontsize=8,
            colorbar=False,
            shrink=.9)

        plt.savefig('corrplot_training_data.eps', format='eps')
        plt.show()


    def prepare_data_for_ML(self,training_data, target_data):
        self.X = self.training_data.as_matrix()
        self.y = np.array(self.target_data)
        return self

    def split_target_from_training_data(self, training_data):
        self.target_data = training_data['Classification']
        self.training_data = training_data.drop('Classification', axis=1)
        return self


    def find_min_boosting_stages(self, gbc_init_params):


        def heldout_score(clf, X_test, y_test, max_n_estimators):
            """compute deviance scores on ``X_test`` and ``y_test``. """
            clf.fit(X_test, y_test)
            score = np.zeros((max_n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                score[i] = clf.loss_(y_test, y_pred)
            return score

        # determine minimum number of estimators with least overfitting
        gbc = GradientBoostingClassifier(**gbc_init_params)

        x_range = np.arange(gbc_init_params['n_estimators'] + 1)
        test_score = heldout_score(gbc, self.X, self.y, gbc_init_params['n_estimators'])

        # min loss according to test (normalize such that first loss is 0)
        test_score -= test_score[0]
        test_best_iter = x_range[np.argmin(test_score)]
        self.optimum_boosting_stages = test_best_iter
        return self

    def find_optimum_isotopes(self, crossfolds=5,
                              gbc_grid_search_params=GBC_GRID_SEARCH_PARAMS):




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

# Set the initial parameters for GBC to use in the initial tuning round


###### ####### ######
###### EXAMPLE ######
###### ####### ######
# Step 1
# import datadi
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
nat_v_tech.filter_negative(training_data = training_data)
nat_v_tech.apply_detection_threshold(training_data = nat_v_tech.training_data,threshold_value= 5)
nat_v_tech.split_target_from_training_data(training_data=nat_v_tech.training_data)
nat_v_tech.prepare_data_for_ML(training_data = nat_v_tech.training_data,target_data=nat_v_tech.target_data)

# initialize gbc to determine max estimators with least overfitting
GBC_INIT_PARAMS = {'loss': 'deviance', 'learning_rate': 0.1,
                   'min_samples_leaf': 100, 'n_estimators': 1000,
                   'max_depth': 5, 'random_state': None, 'max_features': 'sqrt'}
nat_v_tech.find_min_boosting_stages(gbc_init_params=GBC_INIT_PARAMS)
print (nat_v_tech.optimum_boosting_stages)
