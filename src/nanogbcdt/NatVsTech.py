# -*- encoding: utf-8 -*-
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from nanogbcdt.DataUtil import DataUtil
from nanogbcdt.RFECVResult import RFECVResult


class NatVsTech:
    def __init__(self,
                 output_summary_data_path=os.path.join(os.path.dirname(__file__), 'output'),
                 output_summary_base_name='output_summary.csv'):
        self.output_summary_data_path = output_summary_data_path
        self.output_summary_base_name = output_summary_base_name

    # compatibility between python2 and 3
    if sys.version_info >= (3, 0):
        def xrange(*args, **kwargs):
            return iter(range(*args, **kwargs))

    def find_min_boosting_stages(self, training_df, target_df, gbc_base_params):
        """
        :param training_df: dataframe; dataframe containing isotope data
        :param target_df: dataframe; dataframe containing classification data
        :param gbc_base_params: dict; dict containing initial parameters for the gbc
        :return: integer; returns optimal number of boosting stages
        """

        def heldout_score(clf, X_test, y_test, max_n_estimators):
            """compute deviance scores on ``X_test`` and ``y_test``. """
            clf.fit(X_test, y_test)
            score = np.zeros((max_n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                score[i] = clf.loss_(y_test, y_pred)
            return score

        # conform src
        conformed_data = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        # determine minimum number of estimators with least overfitting
        gbc = GradientBoostingClassifier(**gbc_base_params)
        self.gbc_base = gbc

        x_range = np.arange(gbc_base_params['n_estimators'] + 1)
        test_score = heldout_score(gbc, conformed_data[0], conformed_data[1], gbc_base_params['n_estimators'])

        # min loss according to test (normalize such that first loss is 0)
        test_score -= test_score[0]
        test_best_iter = x_range[np.argmin(test_score)]
        self.optimum_boosting_stages = test_best_iter
        return self, test_best_iter

    def find_optimum_gbc_parameters(self, crossfolds=5,
                                    training_df=[],
                                    target_df=[],
                                    gbc_search_params=[]):
        """
        :param crossfolds: int; number of crossfolds
        :param training_df: dataframe; dataframe containing isotope data
        :param target_df: dataframe; dataframe containing classification data
        :param gbc_search_params: dict; dict containing gbc search space
        :return: dict; dict contains optimal parameters for gbc
        """

        grid_searcher = GridSearchCV(estimator=self.gbc_base,
                                     cv=crossfolds,
                                     param_grid=gbc_search_params,
                                     n_jobs=-1)

        # conform src
        (X, y) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        # call the grid search fit using the src
        grid_searcher.fit(X, y)

        # store and print the best parameters
        gbc_best_params = grid_searcher.best_params_

        # re-initialize classifier with best params and fit
        if self.optimum_boosting_stages:
            gbc_best_params['n_estimators'] = self.optimum_boosting_stages

        gbc_fitted = GradientBoostingClassifier(**gbc_best_params)

        return gbc_fitted

    def filter_noncritical_isotopes(self, training_df, critical_isotopes):
        """
        :param training_df: dataframe; dataframe containing isotope data
        :param critical_isotopes: list; list of isotope names to remain
        :return: dataframe; dataframe containing only critical isotopes
        """
        # drop all but critical isotopes (occurs after critical isotopes are found)
        non_crit_isotopes = set(list(training_df)) - set(critical_isotopes)
        return training_df.drop(non_crit_isotopes, axis=1)

    def apply_trained_classification(self,
                                     test_data_path='',
                                     output_summary_data_path='',
                                     output_summary_base_name='',
                                     gbc_fitted='gbc_fitted',
                                     track_class_probabilities=[0.1, 0.1],
                                     isotope_trigger='140Ce',
                                     training_df=pd.DataFrame,
                                     target_df=pd.DataFrame,
                                     filter_neg=True,
                                     apply_threshold=True,
                                     threshold_value=0,
                                     critical_isotopes=False,  # provide an array
                                     track_particle_counts=True):
        """
        :param test_data_path: string; string test data path. Preferred to pass os.path()
        :param output_summary_data_path: string
        :param output_summary_base_name: string
        :param gbc_fitted: scikit-learn abc object;
        :param track_class_probabilities: 2-element array; [0.1,0.1] is remove values between 10-90% likelihood.
        :param isotope_trigger: string; string containing isotope name
        :param training_df: dataframe
        :param target_df: dataframe
        :param filter_neg: boolean; True is filter negatives
        :param apply_threshold: boolean; True is apply threshold
        :param threshold_value: float
        :param critical_isotopes: list; list of critical isotope names
        :param track_particle_counts: boolean; True means track class probability threshold
        :return:
        """
        X_test_predicted_track = []
        X_test_predicted_proba_track = []
        X_test_data_track = pd.DataFrame()

        os.chdir(test_data_path)
        test_data_names = glob.glob('*.csv')

        (X, y) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        gbc = gbc_fitted.fit(X, y)

        for test in test_data_names:

            # initialize a variable to track the csv src names
            run_name = str(test)

            # import in test src for particular run into dataframe.
            test_data = pd.read_csv(os.path.join(
                test_data_path, test), header=0, index_col=0)
            test_data.reset_index(drop=True, inplace=True)

            # filter negative, if assigned
            if filter_neg:
                test_data = DataUtil.filter_negative(data=test_data)

            # apply threshold, if assigned
            if apply_threshold:
                test_data = DataUtil.apply_detection_threshold(data=test_data,
                                                               isotope_trigger=isotope_trigger,
                                                               threshold_value=threshold_value)

            # drop all but critical isotopes (occurs after critical isotopes are found)
            if critical_isotopes:
                non_crit_isotopes = set(list(training_df)) - set(test_data)
                test_data = test_data.drop(non_crit_isotopes, axis=1)

            # assign original src a new name for later analysis
            X_test_preserved = test_data.copy(deep=True)

            # change format for machine learning
            X_test = test_data.as_matrix()

            # use trained classifier to predict imported src
            X_test_predicted = gbc.predict(X_test)
            X_test_predicted_track.append(X_test_predicted)

            # use trained classifier to output trained probabilities
            X_test_predicted_proba = gbc.predict_proba(X_test)
            X_test_predicted_proba_track.append(X_test_predicted_proba)

            total_nat_above_proba_thresh = []
            total_tech_above_proba_thresh = []

            # determine the number of natural particles with prediction proba < 90%
            if track_class_probabilities:
                class_proba = X_test_predicted_proba
                print(np.where(class_proba[:, 0] <= 0.5))
                total_natural_by_proba = len(np.where(class_proba[:, 0] <= 0.5)[0])
                total_technical_by_proba = len(np.where(class_proba[:, 1] <= 0.5)[0])

                nat_above_thresh = len(np.where(class_proba[:, 0] <= track_class_probabilities[0])[0])
                tech_above_thresh = len(np.where(class_proba[:, 1] <= track_class_probabilities[1])[0])

                total_nat_above_proba_thresh = total_natural_by_proba - nat_above_thresh
                total_tech_above_proba_thresh = total_technical_by_proba - tech_above_thresh

            # keep track of particle counts in predictions
            if track_particle_counts:
                X_test_nat_count = list(X_test_predicted).count(1).__float__()
                X_test_tec_count = list(X_test_predicted).count(0).__float__()

                # Organize and track src for table
                X_test_data = pd.DataFrame()
                X_test_data['run_name'] = [run_name]
                X_test_data['total_particle_count'] = [
                    X_test_nat_count + X_test_tec_count]
                X_test_data['nat_particle_count'] = [X_test_nat_count]
                X_test_data['tec_particle_count'] = [X_test_tec_count]
                X_test_data['nat_above_proba_thresh'] = [total_nat_above_proba_thresh]
                X_test_data['tech_above_proba_thresh'] = [total_tech_above_proba_thresh]

                X_test_data_track = X_test_data_track.append(X_test_data)

                X_test_nat_proba = pd.DataFrame(X_test_predicted_proba)
                X_test_preserved['natural_class_proba'] = np.array(X_test_nat_proba[1])
                X_test_preserved.to_csv(os.path.join(output_summary_data_path,
                                                     'filtered_data', str('filtered_' + run_name[:-4] +
                                                                          output_summary_base_name)))

        X_test_data_track.to_csv(os.path.join(output_summary_data_path, 'data_summaries',
                                              output_summary_base_name), index=False)
        return X_test_data_track

    def rfecv_feature_identify(self, training_df=[], target_df=[],
                               n_splits=30, test_size=0.15, random_state=0,
                               cv_grid_search=False,
                               gbc_init_params=[], kfolds=5, gbc_grid_params=[],
                               find_min_boosting_stages=False):
        """
        :param training_df: dataframe
        :param target_df: dataframe
        :param n_splits: integer
        :param test_size: float; value between 0 and 1
        :param random_state: float; 0 is random
        :param cv_grid_search: dict
        :param gbc_init_params: dict
        :param kfolds: int
        :param gbc_grid_params: dict
        :param find_min_boosting_stages: boolean; True means find min boosting stages
        :return:
        """

        # containerize for speed
        result = RFECVResult()

        # Store the feature names by using the headers in the trainingData DataFrame.
        feature_names = list(training_df.columns.values)

        # conform src
        (X_all, y_all) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        # initialize split cross validation
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        # modify class function to enable gbc with rfecv
        class GradientBoostingClassifierrWithCoef(GradientBoostingClassifier):
            def fit(self, *args, **kwargs):
                super(GradientBoostingClassifierrWithCoef, self).fit(*args, **kwargs)
                self.coef_ = self.feature_importances_

        # initialize gbc with altered class definition
        gbc = GradientBoostingClassifierrWithCoef(**gbc_init_params)

        # loop through splits
        run = 0
        for train_index, test_index in sss.split(X=X_all, y=y_all):
            X_train, X_holdout = X_all[train_index], X_all[test_index]
            y_train, y_holdout = y_all[train_index], y_all[test_index]

            run += 1
            print('Runs: ', run, ' of ', n_splits)

            optimum_boosting_stages = 1
            if find_min_boosting_stages:
                optimum_boosting_stages = self.find_min_boosting_stages(gbc_base_params=gbc_init_params,
                                                                        training_df=training_df, target_df=target_df)[1]
                gbc_init_params['n_estimators'] = optimum_boosting_stages
                print('optimum_boosting_stages=', optimum_boosting_stages)

            if cv_grid_search:
                gbc_grid_fitted_params = self.find_optimum_gbc_parameters(crossfolds=5,
                                                                          training_df=training_df, target_df=target_df,
                                                                          gbc_search_params=gbc_grid_params)
                if find_min_boosting_stages:
                    gbc_grid_fitted_params['n_estimators'] = optimum_boosting_stages

                # re-initialize GBC with grid-fit parameters
                gbc = GradientBoostingClassifierrWithCoef(**gbc_grid_fitted_params)

            # Define RFECV function, can  use 'accuracy' or 'f1' f1_weighted, f1_macro
            rfecv = RFECV(estimator=gbc, step=1, cv=kfolds, scoring='f1_weighted')  # , n_jobs=-1)

            # First, the recursive feature elimination model is trained. This fits to the optimum model and begins
            # recursion.
            rfecv = rfecv.fit(X_train, y_train)

            # Second, the cross-validation scores are calculated such that grid_scores_[i] corresponds to the CV score
            # of the i-th subset of features. In other words, from all the features to a single feature, the cross
            # validation score is recorded.
            result.grid_scores_ = result.grid_scores_.append([rfecv.grid_scores_])

            # Third, the .support_ attribute reports whether the feature remains after RFECV or not. The possible
            # parameters are inspected by their ranking. Low ranking features are removed.
            rfecv_support = rfecv.support_

            #  True/False values, where true is a parameter of importance identified by
            # recursive alg.

            # possible_params = rfecv.ranking_
            # min_feature_params = rfecv.get_params(deep=True)
            # optimum_lengths = optimum_lengths.append([rfecv.n_features_])
            feature_set_ids = list(rfecv_support)
            feature_set_ids = list(feature_set_ids)
            feature_names = list(feature_names)
            # named_features = list(training_df.columns.values)
            # named_features = np.array(named_features)

            # Loop over each item in the list of true/false values, if true, pull out the corresponding feature name
            # and store it in the appended namelist. This namelist is rewritten each time, but the information is
            # retained.
            name_list = []  # Initialize a blank array to accept the list of names for features identified as 'True',
            # or important.
            for i in range(0, len(feature_set_ids)):
                if feature_set_ids[i]:
                    name_list.append(feature_names[i])

            result.name_list_ = result.name_list_.append([pd.DataFrame(name_list)])  # append the name list

            # Fourth, the training process begins anew, with the objective to trim to the optimum feature and retrain
            # the model without cross validation i.e., test the holdout set. The new training test set size for the
            # holdout validation should be the entire 90% of the training set (X_trimTrainSet). The holdout test
            # set also needs to be trimmed. The same transformation is performed on the holdout set (X_trimHoldoutSet).
            X_trim_training_set = rfecv.transform(X_train)
            X_trim_holdout_set = rfecv.transform(X_holdout)

            # Fifth, no recursive feature elimination is needed (it has already been done and the poor features
            # removed).
            # Here the model is trained against the trimmed training set X's and corresponding Y's.
            gbc.fit(X_trim_training_set, y_train)

            # Holdout test results are generated here.
            # Predict the class from the holdout dataset
            holdout_predictions = rfecv.predict(X_holdout)

            # determine the F1
            rfc_f1 = metrics.f1_score(y_holdout, holdout_predictions, pos_label=None, average='weighted')
            # determine the R^2 Score
            rfc_r2 = metrics.r2_score(y_holdout, holdout_predictions)
            # determine the MAE - Do this because we want to determine sign.
            rfc_mae = metrics.mean_absolute_error(y_holdout, holdout_predictions)

            # append the previous scores for aggregated analysis.
            result.holdout_predictions_ = result.holdout_predictions_.append([holdout_predictions])
            result.class_scores_f1_ = result.class_scores_f1_.append([rfc_f1])
            result.class_scores_r2_ = result.class_scores_r2_.append([rfc_r2])
            result.class_scores_mae_ = result.class_scores_mae_.append([rfc_mae])

            # determine the feature importances for aggregated analysis.
            result.feature_importances_ = result.feature_importances_.append([gbc.feature_importances_])

        return result

    def plot_feature_frequency(self, result=''):

        # copy result into a new name to avoid over-writing
        result_copy = result.name_list_.copy(deep=True)
        # rename the column for convenience
        result_copy.name_list_.columns = ["isotope"]

        # count runs by output
        optimum_count = pd.DataFrame(result_copy.feature_importances_.count(axis=1),
                                     columns=['optimal_length'])

        # isolate uniques
        unique_featur_list = result_copy.name_list_['isotope'].unique()
        name_count = list(result_copy.name_list_['isotope'].values)

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        for feature_id in unique_featur_list:
            df11 = name_count.count(feature_id)
            df1 = df1.append([df11], ignore_index=True)

            df22 = [feature_id]
            df2 = df2.append(df22, ignore_index=True)

        # turn into a percent
        df1 = df1 / pd.Series(len(optimum_count)) * pd.Series(100)

        # sort and plot
        df3 = pd.concat((df2, df1), axis=1)
        df3.index = unique_featur_list
        df3.columns = ['feature', 'observation %']
        df3.sort_values(by=['observation %', 'feature'], ascending=[True, True], inplace=True)
        df3.plot(kind='barh', title='Feature Frequency (%) Inclusion')
        # plt.savefig(os.path.join(feature_selection_filepath, 'ffi.eps'))
        plt.show()

    def plot_holdout_scores(self, result=[], scorer=pd.DataFrame()):
        # result: result_class
        # scorer: result.class_scores_f1_

        # copy to avoid overwriting

        #  calculate holdout groups using optimum count
        optimum_count = pd.DataFrame(result.feature_importances_.count(axis=1), columns=['optimal_length'])

        # copy to avoid over-writing
        result_copy = scorer.copy(deep=True)

        # rename the column for convenience
        result_copy.columns = ['holdout_f1_scores']

        holdout_groups = result_copy
        holdout_groups['optimal_length'] = optimum_count
        print(holdout_groups)

        # calculate Q1, median, Q3, and variance
        holdout_groups = holdout_groups.groupby(by=['optimal_length'])

        holdout_group_names = list(holdout_groups.groups)
        print(holdout_group_names)
        holdout_group_score_track = pd.DataFrame()
        for group_ in holdout_group_names:
            holdout_group_score = pd.DataFrame()
            holdout_group_score_ = np.array(holdout_groups.get_group(group_)['holdout_f1_scores'])
            holdout_group_score['group'] = [group_]
            holdout_group_score['quartile_1'] = [np.percentile(holdout_group_score_, 25)]
            holdout_group_score['median'] = [np.percentile(holdout_group_score_, 50)]
            holdout_group_score['quartile_3'] = [np.percentile(holdout_group_score_, 75)]
            holdout_group_score['inner_quartile_range'] = [float(holdout_group_score['quartile_3'] - \
                                                                 holdout_group_score['quartile_1'])]
            holdout_group_score['number_observed_in_group'] = [len(holdout_groups.groups.get(group_))]
            if holdout_group_score['inner_quartile_range'].all() > 0:
                holdout_group_score['optimality_score'] = [float(
                    (1 / holdout_group_score['group']) * (1 / holdout_group_score['median']) * (
                        1 / holdout_group_score['inner_quartile_range']) * (
                        holdout_group_score['number_observed_in_group']))]

            holdout_group_score_track = holdout_group_score_track.append(holdout_group_score)

        print(holdout_group_score_track)
