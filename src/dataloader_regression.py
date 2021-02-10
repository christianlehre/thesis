import numpy as np
import pandas as pd
import random
import os
import torch


class Train_test_split:
    def __init__(self, path_to_preprocessed_data, test_size):
        self.path_to_preprocessed_data = path_to_preprocessed_data
        self.test_size = test_size
        self.well_column = "well_name"
        self.target_variable = "ACS"

    def load_preprocessed_data(self):
        data = pd.read_csv(self.path_to_preprocessed_data, sep=";")
        N = len(data.index)
        N_wells = len(list(set(data[self.well_column])))
        setattr(Train_test_split, "N", N)
        setattr(Train_test_split, "N_wells", N_wells)
        return data

    def list_of_dataframes_wellwise(self, df):
        wells = list(set(df[self.well_column]))
        list_of_dataframes = []
        for well in wells:
            well_data = df[df[self.well_column] == well]
            assert len(list(set(well_data[self.well_column]))) == 1, "multiple wells extracted"
            list_of_dataframes.append(well_data)

        return list_of_dataframes

    def shuffle_wells_return_df(self, list_of_wells):
        random.seed(69)
        random.shuffle(list_of_wells)
        shuffled_df = pd.concat(list_of_wells)
        return shuffled_df

    def split_wells(self, df):
        random.seed(69)
        wells = list(set(df[self.well_column]))
        test_wells = random.sample(wells, int(self.test_size*self.N_wells))
        training_wells = [well for well in wells if well not in test_wells]
        return wells, training_wells, test_wells

    def train_test_split_df(self, df, test_wells):
        df_test = df.loc[df[self.well_column].isin(test_wells)]
        df_train = df.loc[~df[self.well_column].isin(test_wells)]
        wells_in_test_df = list(set(df_test[self.well_column]))
        assert all([well in test_wells for well in wells_in_test_df]), "train/test split is incorrect in terms of wells"
        return df_train, df_test, test_wells

    def torch_dataset_train_test(self, df_train, df_test):
        variables = df_train.columns.values
        explanatory_variables = [var for var in variables if var not in [self.target_variable, self.well_column]]
        y_train, y_test = df_train[self.target_variable], df_test[self.target_variable]
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        X_train, X_test = df_train[explanatory_variables], df_test[explanatory_variables]
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        training_set = torch.utils.data.TensorDataset(X_train, y_train)
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
        return training_set, test_set

    def train_test_split(self):
        df = self.load_preprocessed_data()
        list_of_dfs = self.list_of_dataframes_wellwise(df)
        df_shuffled_wellwise = self.shuffle_wells_return_df(list_of_dfs)
        _, _, test_wells = self.split_wells(df_shuffled_wellwise)
        df_train, df_test, _ = self.train_test_split_df(df_shuffled_wellwise, test_wells)
        training_set, test_set = self.torch_dataset_train_test(df_train, df_test)
        return training_set, test_set


class Dataloader:
    def __init__(self, training_set, test_set, batch_size):
        self.training_set = training_set
        self.test_set = test_set
        self.batch_size = batch_size

    def training_loader(self):
        return torch.utils.data.DataLoader(dataset=self.training_set,
                                           batch_size=self.batch_size,
                                           shuffle=True)

    def test_loader(self):
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def unpack_full_test_set(self):
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                           batch_size=len(self.test_set),
                                           shuffle=False)


if __name__ == "__main__":
    preprocessed_data_fname = "preprocessed_regression.csv"
    data_folder = "./data"
    path_to_preprocessed_data = os.path.join(data_folder, preprocessed_data_fname)
    print(os.getcwd())
    os.chdir("..")
    print(os.getcwd())

    test_size = 0.25
    train_test_splitter = Train_test_split(path_to_preprocessed_data=path_to_preprocessed_data, test_size=test_size)
    training_set, test_set = train_test_splitter.train_test_split()

