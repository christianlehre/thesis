import pandas as pd
import random
import os
import torch

from src.dataloader.dataloader import Dataloader


def create_torch_dataset(df, target, predictors):
    y = df[target]
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    x = df[predictors]
    x = torch.tensor(x.values, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


class Train_test_split:
    """
    A class for splitting the dataset into a training, validation and test set

    Parameters:
        - path_to_preprocessed_data (string): relative path to preprocessed data
        - test_size (float): fraction of wells in the test set

    Attributes:
        - path_to_preprocessed_data
        - test_size
        - validation_size (float): fraction of samples in the training set used for validation of the model
        - well_column (string): specifying the column containing wells
        - target_variable (string): specifying the target variable in the dataset
    """
    def __init__(self, path_to_preprocessed_data, test_size):
        self.path_to_preprocessed_data = path_to_preprocessed_data
        self.test_size = test_size
        self.validation_size = 0.10
        self.well_column = "well_name"
        self.target_variable = "ACS"

    def load_preprocessed_data(self):
        """
        Loads and return the preprocessed dataset

        :return: pandas dataframe containing the preprocessed regression dataset
        """
        data = pd.read_csv(self.path_to_preprocessed_data, sep=";")
        N = len(data.index)
        N_wells = len(list(set(data[self.well_column])))
        setattr(Train_test_split, "N", N)
        setattr(Train_test_split, "N_wells", N_wells)
        return data

    def list_of_dataframes_wellwise(self, df):
        """
        create a list of pandas dataframes based on wells

        :param df: pandas dataframe containing the preprocessed dataset
        :return: list of dataframes for each well in the dataset
        """
        wells = list(set(df[self.well_column]))
        list_of_dataframes = []
        for well in wells:
            well_data = df[df[self.well_column] == well]
            assert len(list(set(well_data[self.well_column]))) == 1, "multiple wells extracted"
            list_of_dataframes.append(well_data)

        return list_of_dataframes

    def shuffle_wells_return_df(self, list_of_wells):
        """
        Randomly shuffle the list of dataframes and concatenate the shuffled list

        :param list_of_wells: list of dataframes for each well in the dataset
        :return: dataframe containing all wells, randomly shuffled
        """
        random.Random(42).shuffle(list_of_wells)
        shuffled_df = pd.concat(list_of_wells)
        return shuffled_df

    def split_wells(self, df):
        """
        Randomly select a fraction of the wells for training and testing.
        Randomly select a fraction of the training wells for validation

        :param df: pandas dataframe of the dataset (shuffled by wells)
        :return: list of wells, training wells, validation wells and test wells
        """
        wells = list(set(df[self.well_column]))
        #test_wells = random.Random(42).sample(wells, int(self.test_size*self.N_wells))
        #training_wells = [well for well in wells if well not in test_wells]
        #validation_wells = random.Random(42).sample(training_wells, int(self.validation_size*len(training_wells)))
        #training_wells = [well for well in training_wells if well not in validation_wells]

        training_wells = ['25/8-14 ST2',
                             '30/11-6 S',
                             '30/11-8 S',
                             '25/4-7',
                             '25/5-9',
                             '30/5-2',
                             '30/11-9 A',
                             '25/11-24',
                             '25/10-15 S',
                             '25/2-18 A',
                             '25/8-8 S',
                             '25/10-16 S',
                             '25/10-16 A',
                             '25/4-9 S',
                             '25/8-12 S',
                             '25/2-18 ST2',
                             '25/7-4 S',
                             '25/6-4 S',
                             '25/10-16 C',
                             '25/4-13 A',
                             '30/9-22',
                             '25/10-12 ST2',
                             '25/8-12 A',
                             '25/5-6']

        validation_wells = ['25/5-5', '30/11-7 A']

        test_wells = ['25/7-6',
                         '30/11-11 S',
                         '30/11-9 ST2',
                         '30/6-26',
                         '30/8-5 T2',
                         '25/4-10 S',
                         '30/11-7',
                         '30/11-10']

        return wells, training_wells, validation_wells, test_wells

    def train_val_test_split_df(self, df, train_wells, val_wells, test_wells):
        """
        Split dataset into training, validation and test set

        :param df: pandas dataframe of the dataset (shuffled wellwise)
        :param train_wells: list of training wells
        :param val_wells: list of validation wells
        :param test_wells: list of test wells
        :return: pandas dataframes for the training, validation and test sets
        """
        df_test = df.loc[df[self.well_column].isin(test_wells)]
        df_train = df.loc[df[self.well_column].isin(train_wells)]
        df_val = df.loc[df[self.well_column].isin(val_wells)]

        wells_in_test_df = list(set(df_test[self.well_column]))
        assert all([well in test_wells for well in wells_in_test_df]), "train/test split is incorrect in terms of wells"

        setattr(Train_test_split, "test_wells", test_wells)
        setattr(Train_test_split, "train_wells", train_wells)
        setattr(Train_test_split, "val_wells", val_wells)
        return df_train, df_val, df_test

    def save_train_val_test_split(self, df_train, df_val,  df_test):
        """
        Saves the different datasets to disc

        :param df_train: pandas dataframe for training set
        :param df_val: pandas dataframe for validatino set
        :param df_test: pandas dataframe for test set
        :return: None
        """
        df_train.to_csv("./data/train_regression_scaled_response_wellwise.csv", sep=';', index=False)
        df_val.to_csv("./data/val_regression_scaled_response_wellwise.csv", sep=";", index=False)
        df_test.to_csv("./data/test_regression_scaled_response_wellwise.csv", sep=';', index=False)

    def load_train_val_test_split(self):
        """
        Loads the datasets from disc

        :return: triplet of pandas dataframes for the different datasets
        """
        df_train = pd.read_csv("./data/train_regression.csv", sep=";")
        df_val = pd.read_csv("./data/val_regression.csv", sep=";")
        df_test = pd.read_csv("./data/test_regression.csv", sep=";")

        return df_train, df_val, df_test

    def torch_dataset_train_val_test(self, df_train, df_val, df_test):
        """
        Creates torch datasets from dataframes of the different datasets

        :param df_train: pandas dataframe for training set
        :param df_val: pandas dataframe for validation set
        :param df_test: pandas dataframe for test set
        :return: triplet of torch datasets for the training, validation and test set
        """
        variables = df_train.columns.values
        explanatory_variables = [var for var in variables if var not in [self.target_variable, self.well_column]]
        setattr(Train_test_split, "explanatory_variables", explanatory_variables)

        training_set = create_torch_dataset(df_train, self.target_variable, self.explanatory_variables)
        validation_set = create_torch_dataset(df_val, self.target_variable, self.explanatory_variables)
        test_set = create_torch_dataset(df_test, self.target_variable, self.explanatory_variables)

        return training_set, validation_set, test_set

    def train_val_test_split(self):
        """
        Main function that randomly splits the dataset into the training and test set, and the training set further
        into training and validation set. The function saves the datasets to disc, and creates torch datasets for the
        different splits.
        This function is only called once, as the split is done randomly and will affect the training and testing of
        the model. Further down the pipeline the different datasets are loaded as dataframes and needs to be transformed
        into torch datasets, before being fed to the dataloader class.

        :return: triplet containing torch datasets for the different splits
        """
        df = self.load_preprocessed_data()
        list_of_dfs = self.list_of_dataframes_wellwise(df)
        df_shuffled_wellwise = self.shuffle_wells_return_df(list_of_dfs)
        _, train_wells, val_wells, test_wells = self.split_wells(df_shuffled_wellwise)
        df_train, df_val, df_test = self.train_val_test_split_df(df_shuffled_wellwise, train_wells, val_wells, test_wells)
        self.save_train_val_test_split(df_train, df_val, df_test)

        training_set, validation_set, test_set = self.torch_dataset_train_val_test(df_train, df_val, df_test)
        return training_set, validation_set, test_set


if __name__ == "__main__":
    preprocessed_data_fname = "preprocessed_regression_scaled_response_wellwise.csv"
    data_folder = "./data"
    path_to_preprocessed_data = os.path.join(data_folder, preprocessed_data_fname)
    print(os.getcwd())
    os.chdir("./../..")
    print(os.getcwd())

    test_size = 0.25
    train_test_splitter = Train_test_split(path_to_preprocessed_data=path_to_preprocessed_data, test_size=test_size)
    training_set, validation_set, test_set = train_test_splitter.train_val_test_split()
    print("n_train: {}".format(len(training_set)))
    print("Training wells: {}".format(train_test_splitter.train_wells))
    print("n_val: {}".format(len(validation_set)))
    print("Validation wells: {}".format(train_test_splitter.val_wells))
    print("n_test: {}".format(len(test_set)))
    print("Test wells: {}".format(train_test_splitter.test_wells))

    # The below is simply for testing the dataloader class
    batch_size = 50
    dataloader = Dataloader(training_set=training_set, validation_set=validation_set, test_set=test_set, batch_size=batch_size)
    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    # simple test to ensure that the dataloader iterates over batches specified by the batch size
    for x_train, y_train in training_loader:
        assert x_train.shape[0] == batch_size, "dataloader does not work as intended"
        assert y_train.shape[0] == batch_size, "dataloader does not work as intended"
        assert y_train.shape[1] == 1, "multiple or none values for scalar target"

    for x_val, y_val in validation_loader:
        assert x_val.shape[0] == batch_size, "dataloader does not work as intended"
        assert y_val.shape[0] == batch_size, "dataloader does not work as intended"
        assert y_val.shape[1] == 1, "multiple or none values for scalar target"

    for x_test, y_test in test_loader:
        assert x_test.shape[0] == len(test_set), "dataloder does not work as intended"
        assert y_test.shape[0] == len(test_set), "dataloader does not work as intended"
        assert y_train.shape[1] == 1, "multiple or none values for scalar target"

