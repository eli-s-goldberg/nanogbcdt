# -*- encoding: utf-8 -*-
from __future__ import print_function
import pandas as pd


class RFECVResult:
    def __init__(self):
        self.name_list_ = pd.DataFrame()
        self.optimum_lengths_ = pd.DataFrame()
        self.class_scores_f1_ = pd.DataFrame()
        self.class_scores_r2_ = pd.DataFrame()
        self.class_scores_mae_ = pd.DataFrame()
        self.feature_importances_ = pd.DataFrame()
        self.grid_scores_ = pd.DataFrame()
        self.holdout_predictions_ = []

#    def __str__(self):
#        return print("{} [name_list_={}, optimum_lengths_={}, class_scores_f1_={}, class_scores_r2_={}, "
#                     "class_scores_mae_={}, feature_importances_={}, grid_scores_={}, holdout_predictions_={}]".format(
#                         self.__class__.__name__, self.name_list_, self.optimum_lengths_, self.class_scores_f1_,
#                         self.class_scores_r2_, self.class_scores_r2_, self.feature_importances_, self.grid_scores_,
#                         self.holdout_predictions_))
