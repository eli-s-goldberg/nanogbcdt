import numpy as np

class DataUtil:
    @staticmethod
    def filter_na(data):
        data = data.dropna()
        return data

    @staticmethod
    def filter_negative(data):
        data = data[data >= 0].dropna()
        return data

    @staticmethod
    def apply_detection_threshold(data, isotope_trigger='140Ce', threshold_value=0):
        data = data[data[isotope_trigger] >= threshold_value].dropna()
        return data

    @staticmethod
    def split_target_from_training_data(df, target_name='classification'):
        target_df = df[target_name]
        training_df = df.drop(target_name, axis=1)
        return training_df, target_df

    @staticmethod
    def conform_data_for_ml(training_df, target_df):
        x = training_df.as_matrix()
        y = np.array(target_df)
        return x, y
