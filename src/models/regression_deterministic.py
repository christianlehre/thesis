import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor

from src.dataloader.dataloader import Dataloader


def unpack_dataset_shuffle(dataset):
    """
    Unpack a torch dataset into vector of target variable and design matrix

    :param dataset: torch dataset
    :return: tuple (X, y) of the design matrix X and target variable y
    """
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=len(dataset),
                                       shuffle=True)
    for X, y in loader:
        X, y = X, y

        return X, y


def create_torch_dataset(df, target, predictors):
    """
    Create a torch dataset based on a dataframe, target variable and predictors

    :param df: pandas dataframe containing the dataset
    :param target: target variable (string)
    :param predictors: explanatory variables/predictors (list of strings)
    :return: torch dataset
    """
    y = df[target]
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    x = df[predictors]
    x = torch.tensor(x.values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


class RegressionModel(nn.Module):
    """
    Class for the deterministic regression model

    :param layers: list of the width of each layer in the network, e.g. [num_input, num_hidden, num_output]
    :param num_epochs: number of epochs to train the model for, e.g. 10
    :param batch_size: size of each mini-batch, e.g. 100
    :dropout_rate: dropout rate, e.g 0.10
    :learning_rate: learning rate for the Adam optimizer, e.g. 1e-3
    """
    def __init__(self, layers, num_epochs, batch_size, dropout_rate, learning_rate):
        super(RegressionModel, self).__init__()
        self.layers = layers
        linear_layers = []
        for i in range(len(layers)-1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = torch.nn.Linear(n_in, n_out)

            # Initialize weights (He) and biases (0)
            a = 1 if i == 0 else 2
            layer.weight.data = torch.rand((n_out, n_in)) * np.sqrt(a/n_in)
            layer.bias.data = torch.zeros(n_out)

            linear_layers.append(layer)

        self.linear_layers = torch.nn.ModuleList(linear_layers)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, input):
        """
        Forward pass of the model, calculating the output of the model

        :param input: matrix of samples, dim = (num samples, num predictors)
        :return: output of the neural network, dim = (1,number of samples in input)
        """
        x = input
        for l in self.linear_layers[:-1]:
            x = l(x)
            x = self.dropout(x)
            x = self.relu(x)
        output_layer = self.linear_layers[-1]
        x = output_layer(x)
        return x

    def train_validate_model(self, train_loader, val_loader):
        """
        Train the model using the partial training set,i.e. excluding the validation set,
        estimate the test loss during training using the validation set

        :param train_loader: torch dataloader object for the training set, excluding the validation set
        :param val_loader: torch dataloader object for the validation set
        :return: tuple containig the training and validation loss for each epoch
        """
        train_loss = []
        val_loss = []

        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                outputs = self.forward(x_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()

            train_loss.append(loss.item())

            for x_val, y_val in val_loader:
                outputs_val = self.forward(x_val)
                loss_val = self.criterion(outputs_val, y_val)
            val_loss.append(loss_val.item())
            print("Epoch {}; Train MSE: {},  Val MSE: {}".format(epoch, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

    def train_model(self, train_loader):
        """
        Train the model using the full training set, i.e. including the validation set

        :param train_loader: torch dataloader object for the full training set, including the validation set
        :return: list of training loss for each epoch
        """
        train_loss = []

        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                outputs = self.forward(x_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()

            train_loss.append(loss.item())
            print("Epoch {}; Train MSE: {}".format(epoch, train_loss[epoch]))
        return train_loss

    def evaluate_performance(self, test_loader):
        """
        Evaluate model performance on the full test set in terms of the following performance metrics
        - MSE
        - MAE
        - MAPE

        :param test_loader: torch dataloader object for the test set
        :return: triplet containing the performance metrics, (mse, mae, mape)
        """
        for x, y in test_loader:
            x_test, y_test = x, y
        predictions_test = self.forward(x_test).detach()
        mse = torch.mean(torch.pow(predictions_test - y_test, 2))
        mse = mse.item()
        mae = torch.mean(torch.abs(predictions_test - y_test))
        mae = mae.item()
        mape = 100 * torch.mean(torch.abs(torch.div(predictions_test - y_test, y_test)))
        mape = mape.item()

        return mse, mae, mape

    def wellwise_performance(self, test_df):
        """
        Evaluate model performance for each well in the test dataset in terms of the following performance metrics
        - MSE
        - MAE
        - MAPE

        :param test_df: pandas dataframe containing the test dataset
        :return: 2-dimensional dictionary with wells as keys, and values as dictionaries containing
        triplets of the performance metrics
        """
        test_wells = list(set(test_df["well_name"]))
        performance_dict = {}

        for well in test_wells:
            well_dict = {}
            test_well_data = df_test[df_test["well_name"] == well]
            x_test = test_well_data[explanatory_variables]
            x_test = torch.tensor(x_test.values, dtype=torch.float32)
            y_test = test_well_data["ACS"]
            y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

            predictions_test = self.forward(x_test).detach()

            mse = torch.mean(torch.pow(predictions_test - y_test, 2))
            mae = torch.mean(torch.abs(predictions_test - y_test))
            mape = 100 * torch.mean(torch.abs(torch.div(predictions_test - y_test, y_test)))

            well_dict['mse'] = mse.item()
            well_dict['mae'] = mae.item()
            well_dict['mape'] = mape.item()

            performance_dict[well] = well_dict

        for key, value in performance_dict.items():
            print("well {}: {}".format(key, value))

        return performance_dict

    def plot_predictions_test_wells(self, test_df):
        """
        Plot predicted and true target variable in each well in the test set against depth.

        :param test_df: pandas dataframe containing the test set
        :return: None
        """
        test_wells = list(set(test_df["well_name"]))

        fig, axs = plt.subplots(1, len(test_wells), figsize=(15, 10), sharey=False)
        for ic, well in enumerate(test_wells):
            test_well_data = test_df[test_df["well_name"] == well]

            x_test_ = test_well_data[explanatory_variables]
            depths = x_test_["DEPTH"]
            x_test = torch.tensor(x_test_.values, dtype=torch.float32)
            y_predictions = self.forward(x_test).detach().numpy()
            y_test = test_well_data["ACS"]

            axs[ic].set_ylim(depths.values[-1], depths.values[0])
            axs[ic].plot(y_predictions, depths, ".")
            axs[ic].plot(y_test, depths, ".")
            axs[ic].set_title(well)
            axs[ic].set_xlabel("ACS")


        axs[0].set_ylabel("Depth")
        plt.tight_layout()


if __name__ == "__main__":
    preprocessed_data_fname = "preprocessed_regression.csv"
    data_folder = "./data"
    path_to_preprocessed_data = os.path.join(data_folder, preprocessed_data_fname)
    print(os.getcwd())
    os.chdir("../..")
    print(os.getcwd())

    # Load train/val/test from file
    df_train = pd.read_csv("./data/train_regression.csv", sep=";")
    df_test = pd.read_csv("./data/test_regression.csv", sep=";")
    df_val = pd.read_csv("./data/val_regression.csv", sep=";")

    variables = df_train.columns.values
    target_variable = "ACS"
    well_variable = "well_name"
    explanatory_variables = [var for var in variables if var not in [target_variable, well_variable]]

    # Create torch dataset using train/set split
    training_set = create_torch_dataset(df_train, target_variable, explanatory_variables)
    validation_set = create_torch_dataset(df_val, target_variable, explanatory_variables)
    test_set = create_torch_dataset(df_test, target_variable, explanatory_variables)

    # Set (hyper)parameters for the neural network model
    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    layers = [input_dim, hidden_dim, output_dim]

    num_epochs = 10
    batch_size = 100
    learning_rate = 1e-3
    dropout_rate = 0.10

    # Initialize model object
    model = RegressionModel(layers=layers, num_epochs=num_epochs, batch_size=batch_size,
                            dropout_rate=dropout_rate, learning_rate=learning_rate)

    # Initialize dataloader object
    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    training_loader_full = dataloader.training_loader_full()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    train = False
    grid_search = False

    if train:
        print("Training model on training set (excluding validation set)...")
        start_time = time.time()
        model.train()
        train_loss, val_loss = model.train_validate_model(training_loader, validation_loader)
        model.eval()
        end_time = time.time()
        print("Training time: {:.2f} s".format(end_time - start_time))

        mse, mae, mape = model.evaluate_performance(test_loader)
        print("\nPerformance on test set: ")
        print("MSE: {:.3f}".format(mse))
        print("MAE: {:.3f}".format(mae))
        print("MAPE: {:.3f}%".format(mape))

        plt.figure()
        plt.title("Loss curves")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.plot(range(model.num_epochs), train_loss, label="train loss")
        plt.plot(range(model.num_epochs), val_loss, label="validation loss")
        plt.legend()

        print("Training model on full training set (train + val)...")
        # Initialize new model object
        model = RegressionModel(layers=layers, num_epochs=num_epochs, batch_size=batch_size,
                                dropout_rate=dropout_rate, learning_rate=learning_rate)
        model.train()
        start_time = time.time()
        train_loss = model.train_model(training_loader_full)
        end_time = time.time()
        model.eval()
        print("Training time: {:.2f} s".format(end_time - start_time))
        torch.save(model.state_dict(), "./data/models/regression/regression_deterministic.pt")

        mse, mae, mape = model.evaluate_performance(test_loader)
        print("\nPerformance on test set: ")
        print("MSE: {:.3f}".format(mse))
        print("MAE: {:.3f}".format(mae))
        print("MAPE: {:.3f}%".format(mape))

        plt.figure()
        plt.title("Loss curve for full training set")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.plot(range(model.num_epochs), train_loss, label="train loss")
        plt.legend()

    elif grid_search:
        net_regr = NeuralNetRegressor(RegressionModel,
                                      module__layers=layers,
                                      module__dropout_rate=0.10,
                                      module__num_epochs=10,
                                      module__batch_size=1000,
                                      module__learning_rate=1e-3,
                                      train_split=False,
                                      criterion=nn.MSELoss,
                                      optimizer=torch.optim.Adam)
        params = {
            'max_epochs': [10],
            'lr': [1e-3, 1e-4, 1e-5],
            'module__layers': [[input_dim, 50, output_dim],
                               [input_dim, 100, output_dim],
                               [input_dim, 250, output_dim]],
            'module__batch_size': [100, 500, 1000]
        }

        gs = GridSearchCV(net_regr, params, refit=False, cv=3, scoring='neg_mean_squared_error', verbose=3)
        X, y = unpack_dataset_shuffle(training_set)
        start_time_gs = time.time()
        gs.fit(X, y)
        end_time_gs = time.time()

        print('Best score, Best parameter configuration')
        print(gs.best_score_, gs.best_params_)
        print("Elapsed time: {} s".format(end_time_gs - start_time_gs))

    else:
        model.load_state_dict(torch.load("./data/models/regression/regression_deterministic.pt"))
        model.eval()
        mse, mae, mape = model.evaluate_performance(test_loader)
        print("Performance on full test set: ")
        print("MSE: {:.3f}".format(mse))
        print("MAE: {:.3f}".format(mae))
        print("MAPE: {:.3f}%".format(mape))

    # well-wise performance
    performance_dict = model.wellwise_performance(df_test)

    # plot predictions for all wells in test set
    model.plot_predictions_test_wells(df_test)

    plt.show()
