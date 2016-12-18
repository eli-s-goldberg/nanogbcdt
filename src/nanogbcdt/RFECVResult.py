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
