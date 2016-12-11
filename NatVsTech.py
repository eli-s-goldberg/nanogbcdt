import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biokit.viz import corrplot
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from DataUtil import DataUtil


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
        def heldout_score(clf, X_test, y_test, max_n_estimators):
            """compute deviance scores on ``X_test`` and ``y_test``. """
            clf.fit(X_test, y_test)
            score = np.zeros((max_n_estimators,), dtype=np.float64)
            for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
                score[i] = clf.loss_(y_test, y_pred)
            return score

        # conform data
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

        grid_searcher = GridSearchCV(estimator=self.gbc_base,
                                     cv=crossfolds,
                                     param_grid=gbc_search_params,
                                     n_jobs=-1)

        # conform data
        (X, y) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        # call the grid search fit using the data
        grid_searcher.fit(X, y)

        # store and print the best parameters
        self.gbc_best_params = grid_searcher.best_params_

        # re-initialize classifier with best params and fit
        if self.optimum_boosting_stages:
            self.gbc_best_params['n_estimators'] = self.optimum_boosting_stages

        gbc_fitted = GradientBoostingClassifier(**self.gbc_best_params)

        return gbc_fitted

    def filter_noncritical_isotopes(self, training_df, critical_isotopes):
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
                                     critical_isotopes=False,  # provide an array
                                     track_particle_counts=True):
        X_test_predicted_track = []
        X_test_predicted_proba_track = []
        X_test_data_track = pd.DataFrame()

        os.chdir(test_data_path)
        test_data_names = glob.glob('*.csv')

        (X, y) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        gbc = gbc_fitted.fit(X, y)

        for test in test_data_names:

            # initialize a variable to track the csv data names
            run_name = str(test)

            # import in test data for particular run into dataframe.
            test_data = pd.read_csv(os.path.join(
                test_data_path, test), header=0, index_col=0)
            test_data.reset_index(drop=True, inplace=True)

            # filter negative, if assigned
            if filter_neg:
                test_data = DataUtil.filter_negative(data=test_data)

            # apply threshold, if assigned
            if apply_threshold:
                test_data = DataUtil.apply_detection_threshold(data=test_data, isotope_trigger=isotope_trigger)

            # drop all but critical isotopes (occurs after critical isotopes are found)
            if critical_isotopes:
                non_crit_isotopes = set(list(training_df)) - set(test_data)
                test_data = test_data.drop(non_crit_isotopes, axis=1)

            # assign original data a new name for later analysis
            X_test_preserved = test_data.copy(deep=True)

            # change format for machine learning
            X_test = test_data.as_matrix()

            # use trained classifier to predict imported data
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

                # Organize and track data for table
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

    def rfecv_feature_identify(self, training_df=[],
                               target_df=[], n_splits=30,
                               test_size=0.15, random_state=0,
                               cv_grid_search=False,
                               gbc_init_params=[],
                               kfolds=5,
                               gbc_grid_params=[],
                               find_min_boosting_stages=False):

        # todo(create more elegant output handling)
        # containerize for speed
        nameListAll = pd.DataFrame()
        optimumLengthAll = pd.DataFrame()
        classScoreAll = pd.DataFrame()
        classScoreAll2 = pd.DataFrame()
        classScoreAll3 = pd.DataFrame()
        featureImportancesAll = pd.DataFrame()
        rfecvGridScoresAll = pd.DataFrame()

        # Store the feature names by using the headers in the trainingData DataFrame.
        feature_names = list(training_df.columns.values)

        # conform data
        (X_all, y_all) = DataUtil.conform_data_for_ml(training_df=training_df, target_df=target_df)

        # initialize split cross validation
        sss = StratifiedShuffleSplit(n_splits=n_splits,
                                     test_size=test_size,
                                     random_state=random_state)

        # get splits (silly initialization in new version of scikit).
        sss.get_n_splits(X=X_all, y=y_all)

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

            run = run + 1
            print(run, 'of: ', n_splits)

            if find_min_boosting_stages:
                optimum_boosting_stages = self.find_min_boosting_stages(gbc_base_params=gbc_init_params,
                                                                        training_df=training_df,
                                                                        target_df=target_df)[1]

                gbc_init_params['n_estimators'] = optimum_boosting_stages
                print(optimum_boosting_stages)

            if cv_grid_search:

                gbc_grid_fitted_params = self.find_optimum_gbc_parameters(crossfolds=5,
                                                                          training_df=training_df,
                                                                          target_df=target_df,
                                                                          gbc_search_params=gbc_grid_params)
                if find_min_boosting_stages:
                    gbc_grid_fitted_params['n_estimators'] = optimum_boosting_stages

                # re-initialize GBC with grid-fit parameters
                gbc = GradientBoostingClassifierrWithCoef(**gbc_grid_fitted_params)

            # Define RFECV function, can  use 'accuracy' or 'f1' f1_weighted, f1_macro
            rfecv = RFECV(estimator=gbc, step=1, cv=kfolds, scoring='f1_weighted')

            # First, the recursive feature elimination model is trained. This fits to the optimum model and begins
            # recursion.
            rfecv = rfecv.fit(X_train, y_train)

            # Second, the cross-validation scores are calculated such that grid_scores_[i] corresponds to the CV score
            # of the i-th subset of features. In other words, from all the features to a single feature, the cross
            # validation score is recorded.
            rfecvGridScoresAll = rfecvGridScoresAll.append([rfecv.grid_scores_])

            # Third, the .support_ attribute reports whether the feature remains after RFECV or not. The possible
            # parameters are inspected by their ranking. Low ranking features are removed.
            supPort = rfecv.support_

            #  True/False values, where true is a parameter of importance identified by
            # recursive alg.

            possParams = rfecv.ranking_
            min_feature_params = rfecv.get_params(deep=True)
            optimumLengthAll = optimumLengthAll.append([rfecv.n_features_])
            featureSetIDs = list(supPort)
            featureSetIDs = list(featureSetIDs)
            feature_names = list(feature_names)
            namedFeatures = list(training_df.columns.values)
            namedFeatures = np.array(namedFeatures)

            # Loop over each item in the list of true/false values, if true, pull out the corresponding feature name
            # and store it in the appended namelist. This namelist is rewritten each time, but the information is
            # retained.
            nameList = []  # Initialize a blank array to accept the list of names for features identified as 'True',
            # or important.
            for i in range(0, len(featureSetIDs)):
                if featureSetIDs[i]:
                    nameList.append(feature_names[i])
                else:
                    a = 1
            nameList = pd.DataFrame(nameList)
            nameListAll = nameListAll.append([nameList])  # append the name list ###
            nameList = list(nameList)
            nameList = np.array(nameList)

            # Fourth, the training process begins anew, with the objective to trim to the optimum feature and retrain
            # the model without cross validation i.e., test the holdout set. The new training test set size for the
            # holdout validation should be the entire 90% of the training set (X_trimTrainSet). The holdout test
            # set also needs to be trimmed. The same transformation is performed on the holdout set (X_trimHoldoutSet).
            X_trimTrainSet = rfecv.transform(X_train)
            X_trimHoldoutSet = rfecv.transform(X_holdout)

            # Fifth, no recursive feature elimination is needed (it has already been done and the poor features
            # removed).
            # Here the model is trained against the trimmed training set X's and corresponding Y's.
            gbc.fit(X_trimTrainSet, y_train)

            # Holdout test results are generated here.
            # Predict the class from the holdout dataset
            rfecv.n_jobs = -1
            holdout_preds = rfecv.predict(X_holdout)  ###

            # determine the F1
            rfc_all_f1 = metrics.f1_score(y_holdout, holdout_preds, pos_label=None,
                                          average='weighted')  ###

            # determine the R^2 Score
            rfc_all_f2 = metrics.r2_score(y_holdout, holdout_preds)  ###
            # determine the MAE - Do this because we want to determine sign.
            rfc_all_f3 = metrics.mean_absolute_error(y_holdout, holdout_preds)  ###

            # append the previous scores for aggregated analysis.
            classScoreAll = classScoreAll.append([rfc_all_f1])  ###
            classScoreAll2 = classScoreAll2.append([rfc_all_f2])  ###
            classScoreAll3 = classScoreAll3.append([rfc_all_f3])  ###
            # determine the feature importances for aggregated analysis.
            refinedFeatureImportances = gbc.feature_importances_
            featureImportancesAll = featureImportancesAll.append([refinedFeatureImportances])

        return nameListAll, rfecvGridScoresAll, holdout_preds, classScoreAll, classScoreAll2, classScoreAll3, featureImportancesAll
