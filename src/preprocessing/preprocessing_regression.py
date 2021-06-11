import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump


class Preprocessing:
    def __init__(self, path_to_raw_data, path_to_preprocessed_data):
        self.path_to_raw_data = path_to_raw_data
        self.path_to_preprocessed_data = path_to_preprocessed_data
        self.target_variable = "ACS"
        self.list_of_features = ["well_name",
                                 "DEPTH",
                                 "AC",
                                 "AI",
                                 "BS",
                                 "CALI",
                                 "DEN",
                                 "GR",
                                 "NEU",
                                 "RMED"] # well_name needed for per well scaling of features
        self.numerical_variables = [col for col in self.list_of_features if col != "well_name"]
        self.list_of_variables = [self.target_variable] + self.list_of_features
        self.columns_to_engineer = [col for col in self.list_of_features if col not in ["well_name", "DEPTH", self.target_variable]]
        self.add_windows = True
        self.add_gradients = True
        self.window = 4

    def load_data(self):
        """
        Load raw dataset

        :return: pandas dataframe
        """
        data = pd.read_csv(self.path_to_raw_data, sep=",")
        return data

    def feature_selection(self, df):
        """
        Select relevant features from the dataset

        :param df: pandas dataframe
        :return: pandas dataframe
        """
        cols_in_df = df.columns.values
        assert all([col in cols_in_df for col in self.list_of_variables]), \
            "One/more of the selected features are not in the dataset"
        return df[self.list_of_variables]

    def clean_rows_target(self, df):
        """
        Clean the dataset, remove rows with poor measurements

        :param df: pandas dataframe
        :return: pandas dataframe
        """
        df.dropna(subset=[self.target_variable], inplace=True)
        df.drop(df[(df['BADACS'] == 1) | (df['BADACS'].isnull())].index, axis=0, inplace=True)
        return df

    def scale_features_wellwise(self, df):
        """
        Perform the well-wise scaling of the features

        :param df: pandas dataframe
        :return: pandas dataframe
        """
        new_df = df.copy(deep=True)
        new_df.drop(["well_name", "DEPTH"], axis=1, inplace=True)
        wells = df['well_name'].unique()
        for well in wells:
            scaler = StandardScaler()
            new_df[df['well_name'] == well] = scaler.fit_transform(new_df[df['well_name'] == well])
        new_df['well_name'] = df['well_name'] # just for validating the scaling and splitting data further down the pipeline

        return new_df

    def validate_scaling(self, df):
        """
        Helper function to validate the scaling

        :param df: pandas dataframe
        :return: None
        """
        wells = df['well_name'].unique()
        for well in wells:
            data_pr_well = df[df['well_name'] == well]

            for col in df.columns:
                if col != 'well_name':
                    print('Column: {}, Well: {}'.format(col, well))
                    mean = np.mean(data_pr_well[col])
                    std = np.std(data_pr_well[col])
                    print('mean: {}, std: {}'.format(mean, std))
                    print('')

    def mean_imputer(self, df):
        """
        Impute missing values using the mean value of each feature

        :param df: pandas dataframe
        :return: pandas dataframe
        """
        for col in self.numerical_variables:
            assert col in df.columns.values, "Column {} not in dataset".format(col)
            if df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)

        return df

    def feature_engineering_add_gradients(self, df, columns=None):
        """
        Add gradients of each feature to the feature set

        :param df: pandas dataframe
        :param columns: list of feature to add gradient to
        :return: pandas dataframe
        """
        if columns is None:
            columns = df.columns.values

        gradient_cols = [col+"_gradient" for col in columns]
        for col in columns:
            if col != "well_name" or col != self.target_variable:
                df[col+"_gradient"] = np.gradient(df[col])

        return df, gradient_cols

    def feature_engineering_add_windows(self, df, columns=None):
        """
        Add rolling window features for the mean, max and min value to the feature set

        :param df: pandas dataframe
        :param columns: list of features to add rolling window functions to
        :return:
        """
        if columns is None:
            columns = df.columns.values

        mean_cols = [col+"_window_mean" for col in columns if col != "well_name" or col != self.target_variable]
        min_cols = [col+"_window_min" for col in columns if col not in ["well_name", self.target_variable]]
        max_cols = [col+"_window_max" for col in columns if col not in ["well_name", self.target_variable]]
        for col in columns:
            df[col+"_window_mean"] = df[col].rolling(center=False, window=self.window, min_periods=1).mean()
            df[col+"_window_min"] = df[col].rolling(center=False, window=self.window, min_periods=1).min()
            df[col+"_window_max"] = df[col].rolling(center=False, window=self.window, min_periods=1).max()

        window_cols = mean_cols + min_cols + max_cols
        return df, window_cols

    def save_dataset(self, df):
        """
        Save dataset

        :param df: pandas dataframe
        :return: None
        """
        df.to_csv(self.path_to_preprocessed_data, sep=";", index=False)

    def preprocessing_main(self):
        """
        Main function for performing the preprocessing scheme

        :return: pandas dataframe
        """
        data = self.load_data()
        print('Dimensions of original data: {}'.format(data.shape))
        data = self.clean_rows_target(data)
        print('Dimension of data cleaned for bad/missing ACS: {}'.format(data.shape))
        data = self.feature_selection(data)
        data = self.mean_imputer(data)
        data = self.scale_features_wellwise(data)
        added_cols = []
        if self.add_gradients:
            data, gradient_cols = self.feature_engineering_add_gradients(data, columns=self.columns_to_engineer)
            added_cols += gradient_cols

        if self.add_windows:
            data, window_cols = self.feature_engineering_add_windows(data, columns=self.columns_to_engineer)
            added_cols += window_cols

        added_plus_selected = added_cols + self.list_of_variables
        assert all([col in added_plus_selected for col in data.columns.values]), \
            "Variables not properly added to dataset"
        print('Dimensions of preprocessed data: {} '.format(data.shape))
        return data


if __name__ == "__main__":
    raw_data_fname = "raw_regression.csv"
    preprocessed_data_fname = "preprocessed_regression.csv"
    data_folder = "./data"
    path_to_raw_data = os.path.join(data_folder, raw_data_fname)
    path_to_preprocessed_data = os.path.join(data_folder, preprocessed_data_fname)
    os.chdir("../..")

    preprocessor = Preprocessing(path_to_raw_data=path_to_raw_data, path_to_preprocessed_data=path_to_preprocessed_data)
    preprocessed_data = preprocessor.preprocessing_main()
    preprocessor.save_dataset(preprocessed_data)





