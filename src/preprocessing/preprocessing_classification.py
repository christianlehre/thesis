import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, path_to_raw_dataset, path_to_preprocessed_dataset):
        self.path_to_dataset = path_to_raw_dataset
        self.path_to_preprocessed_dataset = path_to_preprocessed_dataset
        self.target_variable = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        self.list_of_features = ["WELL",
                                 "DEPTH_MD",
                                 "CALI",
                                 "RMED",
                                 "RDEP",
                                 "GR",
                                 "PEF",
                                 "RHOB",
                                 "NPHI",
                                 "DTC"] #TODO: check if other features should be used
        self.numerical_variables = [var for var in self.list_of_features if var != 'WELL']
        self.list_of_variables = [self.target_variable] + self.list_of_features
        self.add_windows = True
        self.add_gradients = True
        self.window = 4

    def load_dataset(self):
        data = pd.read_csv(self.path_to_dataset, sep=";")
        return data

    def filter_high_confidence(self, df): #TODO: discuss with peder if this should be done
        return df[df["FORCE_2020_LITHOFACIES_CONFIDENCE"] == 1]

    def scale_features_wellwise(self, df): #scale features per well, not per dataset
        new_df = df.copy(deep=True)
        new_df.drop(['WELL', "DEPTH_MD", self.target_variable], axis=1, inplace=True)
        scaler = StandardScaler()
        wells = df['WELL'].unique()
        for well in wells:
            new_df[df['WELL'] == well] = scaler.fit_transform(new_df[df['WELL'] == well])

        new_df['WELL'] = df['WELL']
        new_df["DEPTH_MD"] = df["DEPTH_MD"]
        new_df[self.target_variable] = df[self.target_variable]
        return new_df

    def validate_scaling(self, df):
        wells = df['WELL'].unique()
        for well in wells:
            data_pr_well = df[df['WELL'] == well]

            for col in df.columns:
                if col != 'WELL':
                    print('Column: {}, Well: {}'.format(col, well))
                    mean = np.mean(data_pr_well[col])
                    std = np.std(data_pr_well[col])
                    print('mean: {}, std: {}'.format(mean, std))
                    print('')

    def feature_selection(self, df):
        return df[self.list_of_variables]

    def mean_imputer(self, df):
        for col in self.numerical_variables:
            assert col in df.columns.values, "Column {} not in dataset".format(col)
            if df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)

        return df

    def feature_engineering_add_gradients(self, df, columns=None):
        if columns is None:
            columns = df.columns.values

        gradient_cols = [col + "_gradient" for col in columns if col != 'WELL' or col != self.target_variable]
        for col in columns:
            if col != 'WELL' or col != self.target_variable:
                df[col+"_gradient"] = np.gradient(df[col])

        return df, gradient_cols

    def feature_engineering_add_windows(self, df, columns=None):
        if columns is None:
            columns = df.columns.values

        mean_cols = [col+"_window_mean" for col in columns if col != 'WELL' or col != self.target_variable]
        min_cols = [col+"_window_min" for col in columns if col != 'WELL' or col != self.target_variable]
        max_cols = [col+"_window_max" for col in columns if col != 'WELL' or col != self.target_variable]
        for col in columns:
            if col != 'WELL' or col != self.target_variable:
                df[col+"_window_mean"] = df[col].rolling(center=False, window=self.window, min_periods=1).mean()
                df[col+"_window_min"] = df[col].rolling(center=False, window=self.window, min_periods=1).min()
                df[col+"_window_max"] = df[col].rolling(center=False, window=self.window, min_periods=1).max()

        window_cols = mean_cols + min_cols + max_cols
        return df, window_cols

    def convert_target(self, df):
        lithology_numbers = {30000: 0,
                             65030: 1,
                             65000: 2,
                             80000: 3,
                             74000: 4,
                             70000: 5,
                             70032: 6,
                             88000: 7,
                             86000: 8,
                             99000: 9,
                             90000: 10,
                             93000: 11}
        df[self.target_variable] = df[self.target_variable].map(lithology_numbers)
        return df

    def save_dataset(self, df):
        df.to_csv(self.path_to_preprocessed_dataset, sep=";")

    def preprocessing_main(self):
        data = self.load_dataset()
        data = self.feature_selection(data)
        data = self.mean_imputer(data)
        data = self.scale_features_wellwise(data)
        if self.add_gradients: # Do this before or after scaling?
            data, gradient_cols = self.feature_engineering_add_gradients(data, columns=self.numerical_variables)

        if self.add_windows: # Do this before/after scaling?
            data, window_cols = self.feature_engineering_add_windows(data, columns=self.numerical_variables)

        data = self.convert_target(data)
        return data


if __name__ == "__main__":
    raw_data_fname = "raw_classification.csv"
    preprocessed_data_fname = "preprocessed_classification.csv"
    data_folder = "./data/"
    path_to_raw_dataset = os.path.join(data_folder, raw_data_fname)
    path_to_preprocessed_dataset = os.path.join(data_folder, preprocessed_data_fname)
    os.chdir("../..")

    preprocessor = Preprocessing(path_to_raw_dataset=path_to_raw_dataset,
                                 path_to_preprocessed_dataset=path_to_preprocessed_dataset)
    preprocessed_data = preprocessor.preprocessing_main()
    preprocessor.save_dataset(preprocessed_data)

