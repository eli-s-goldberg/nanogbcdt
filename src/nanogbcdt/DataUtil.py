import numpy as np

class DataUtil:
    @staticmethod
    def filter_na(data):
        """
        :param data: dataframe
        :return: dataframe
        """
        data = data.dropna()
        return data

    @staticmethod
    def filter_negative(data):
        """
        :param data: dataframe
        :return: dataframe
        """
        data = data[data >= 0].dropna()
        return data

    @staticmethod
    def apply_detection_threshold(data, isotope_trigger='140Ce', threshold_value=0):
        """
        :param data: dataframe
        :param isotope_trigger: string; isotope name
        :param threshold_value: float; isotope hit value
        :return: dataframe
        """
        data = data[data[isotope_trigger] >= threshold_value].dropna()
        return data

    @staticmethod
    def split_target_from_training_data(df, target_name='classification'):
        """
        :param df: dataframe; dataframe containing isotope data and column for classification
        :param target_name: string; name of the target data column
        :return: training_df, target_df; training dataframe, target dataframe
        """
        target_df = df[target_name]
        training_df = df.drop(target_name, axis=1)
        return training_df, target_df

    @staticmethod
    def conform_data_for_ml(training_df, target_df):
        """
        :param training_df: dataframe; dataframe containing only training data
        :param target_df: dataframe; dataframe containing only target data
        :return: x, y; training data matrix; target data array
        """
        x = training_df.as_matrix()
        y = np.array(target_df)
        return x, y
