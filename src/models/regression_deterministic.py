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
    def __init__(self, layers, num_epochs, batch_size, N, dropout_rate, learning_rate, heteroscedastic):
        super(RegressionModel, self).__init__()
        self.heteroscedastic = heteroscedastic
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()

        # set weight decay for non-output and output layer
        self.precision = 1.0
        self.length_scale = 1.0
        self.N = N
        self.reg_dropout = (1 - self.dropout_rate) * self.length_scale ** 2 / (2 * self.N * self.precision)
        self.reg_final = self.length_scale ** 2 / (2 * self.N * self.precision)

        self.layers = layers
        linear_layers = []
        bn_layers = []
        reg_list = []
        for i in range(len(layers)-1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = torch.nn.Linear(n_in, n_out)
            bn_layer = torch.nn.BatchNorm1d(num_features=n_out, track_running_stats=False)

            # Initialize weights (He) and biases (0)
            a = 1 if i == 0 else 2
            layer.weight.data = torch.rand((n_out, n_in)) * np.sqrt(a/n_in)
            layer.bias.data = torch.zeros(n_out)

            linear_layers.append(layer)
            bn_layers.append(bn_layer)

            reg_dict_linear = {}
            bn_dict = {}
            reg_dict_linear['params'] = layer.parameters()
            reg_dict_linear['weight_decay'] = self.reg_dropout if i < (len(layers)-1) else self.reg_final
            bn_dict['params'] = bn_layer.parameters()

            reg_list.append(reg_dict_linear)
            reg_list.append(bn_dict)

        self.linear_layers = torch.nn.ModuleList(linear_layers)
        self.bn_layers = bn_layers

        if heteroscedastic:
            self.log_var = nn.Linear(n_in, n_out)
            self.log_var.weight.data = torch.rand((n_out, n_in)) * np.sqrt(a / n_in)
            self.log_var.bias.data = torch.zeros(n_out)
            reg_list.append({'params': self.log_var.parameters(), 'weight_decay': self.reg_final})

            self.forward = self.forward_heteroscedastic
            self.loss = self.loss_heteroscedastic
        else:
            self.log_var = nn.Parameter(torch.FloatTensor(1, ).normal_(mean=-2.5, std=0.001), requires_grad=True)
            reg_list.append({'params': self.log_var, 'weight_decay': self.reg_final})

            self.forward = self.forward_homoscedastic
            self.loss = self.loss_homoscedastic


        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(reg_list, lr=self.learning_rate)

    def forward_homoscedastic(self, input):
        """
        Forward pass of the model, calculating the output of the model

        :param input: matrix of samples, dim = (num samples, num predictors)
        :return: output of the neural network, dim = (1,number of samples in input)
        """
        x = input
        for i, l in enumerate(self.linear_layers[:-1]):
            bn_layer = self.bn_layers[i]
            x = l(x)
            x = bn_layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        output_layer = self.linear_layers[-1]
        x = output_layer(x)
        return x

    def loss_homoscedastic(self, y, y_pred):
        neg_loglik = 0.5*len(y)*(np.log(2*np.pi) + self.log_var) + 0.5*torch.div(torch.pow(y - y_pred, 2),
                                                                                 torch.exp(self.log_var)).sum()
        return neg_loglik/self.precision

    def forward_heteroscedastic(self, input):
        x = input
        for i, l in enumerate(self.linear_layers[:-1]):
            bn_layer = self.bn_layers[i]
            x = l(x)
            x = bn_layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        output_layer = self.linear_layers[-1]
        y_hat = output_layer(x)
        log_var = self.log_var(x)
        return y_hat, log_var

    def loss_heteroscedastic(self, y, y_pred, log_var):
        neg_loglik = 0.5 * len(y) * np.log(2 * np.pi) + 0.5 * (log_var.sum() +
                                                               torch.div(torch.pow(y - y_pred, 2),
                                                                         torch.exp(log_var)).sum())
        return neg_loglik

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
                if self.heteroscedastic:
                    outputs, log_var = self.forward(x_train)
                    loss = self.loss(y_train, outputs, log_var)
                else:
                    outputs = self.forward(x_train)
                    loss = self.loss(y_train, outputs)
                loss.backward()
                self.optimizer.step()

            train_loss.append(loss.item())

            for x_val, y_val in val_loader:
                if self.heteroscedastic:
                    outputs_val, log_var_val = self.forward(x_val)
                    loss_val = self.loss(y_val, outputs_val, log_var_val)
                else:
                    outputs_val = self.forward(x_val)
                    loss_val = self.loss(y_val, outputs_val)
            val_loss.append(loss_val.item())
            print("Epoch {}; Train NLL: {},  Val NLL: {}".format(epoch, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

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
        if self.heteroscedastic:
            predictions, _ = self.forward(x_test)
            predictions_test = predictions.detach()
        else:
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

            if self.heteroscedastic:
                predictions, _ = self.forward(x_test)
                predictions_test = predictions.detach()
            else:
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
            if self.heteroscedastic:
                predictions, _ = self.forward(x_test)
                y_predictions = predictions.detach()
            else:
                y_predictions = self.forward(x_test).detach()
            y_test = test_well_data["ACS"]

            axs[ic].set_ylim(depths.values[-1], depths.values[0])
            axs[ic].plot(y_test, depths, ".", label="True")
            axs[ic].plot(y_predictions, depths, ".", label="Predictions")

            axs[ic].set_title(well)
            axs[ic].set_xlabel("ACS")


        axs[0].set_ylabel("Depth")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='best')
        plt.tight_layout()

    def plot_predictions_velocity_ratio_test_wells(self, test_df):
        test_wells = list(set(test_df["well_name"]))

        fig, axs = plt.subplots(1, len(test_wells), figsize=(15, 10), sharey=False)
        for ic, well in enumerate(test_wells):
            test_data_well = test_df[test_df["well_name"] == well]
            acs_test = test_data_well["ACS"]
            x_test = test_data_well[explanatory_variables]
            depths = x_test["DEPTH"]
            ac_test = x_test["AC"]

            ratio_test = np.divide(np.array(ac_test), np.array(acs_test))

            x_test = torch.tensor(x_test.values, dtype=torch.float32)

            if self.heteroscedastic:
                predictions, _ = self.forward(x_test)
                acs_predictions = predictions.detach().numpy()
            else:
                acs_predictions = self.forward(x_test).detach().numpy()

            ratio_predictions = np.divide(np.array(ac_test), np.array(acs_predictions[:, 0]))

            axs[ic].set_ylim(depths.values[-1], depths.values[0])
            axs[ic].plot(ratio_test, depths, ".", label="True")
            axs[ic].plot(ratio_predictions, depths, ".", label="Predictions")

            axs[ic].set_title(well)
            axs[ic].set_xlabel("AC/ACS")
        axs[0].set_ylabel("Depths")
        plt.legend(loc="best",bbox_to_anchor=(1.05, 1))
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
    layers = [input_dim, hidden_dim, hidden_dim, output_dim]

    num_epochs = 10
    batch_size = 100
    learning_rate = 1e-4
    dropout_rate = 0.10
    N = len(training_set)

    # Initialize model object
    model = RegressionModel(layers=layers, num_epochs=num_epochs, batch_size=batch_size,
                            N=N, dropout_rate=dropout_rate, learning_rate=learning_rate, heteroscedastic=False)

    # Initialize dataloader object
    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    training_loader_full = dataloader.training_loader_full()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    train = False
    grid_search = False

    if model.heteroscedastic:
        path_to_model = "./data/models/regression/regression_deterministic_heteroscedastic.pt"
        path_to_loss = "./data/loss/regression/regession_deterministic_heteroscedastic.npz"
    else:
        path_to_model = "./data/models/regression/regression_deterministic_homoscedastic.pt"
        path_to_loss = "./data/loss/regression/regession_deterministic_homoscedastic.npz"

    if train:
        print("Training model on training set (excluding validation set)...")
        start_time = time.time()
        model.train()
        training_loss, validation_loss = model.train_validate_model(training_loader, validation_loader)
        model.eval()
        end_time = time.time()
        training_time = end_time - start_time
        print("Training time: {:.2f} s".format(training_time))
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=training_loss, validation_loss=validation_loss, training_time=training_time)

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
        model.load_state_dict(torch.load(path_to_model))
        model.train(mode=False)

        with np.load(path_to_loss) as data:
            training_loss = data["training_loss"]
            validation_loss = data["validation_loss"]
            training_time = data["training_time"]

    plt.figure()
    plt.title("Loss curves, training time: {:.2f}s".format(training_time))
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.plot(range(model.num_epochs), training_loss, label="train loss")
    plt.plot(range(model.num_epochs), validation_loss, label="validation loss")
    plt.legend()

    mse, mae, mape = model.evaluate_performance(test_loader)
    print("Performance on full test set: ")
    print("MSE: {:.5f}".format(mse))
    print("MAE: {:.5f}".format(mae))
    print("MAPE: {:.5f}%".format(mape))

    # well-wise performance
    performance_dict = model.wellwise_performance(df_test)

    # plot predictions for all wells in test set
    model.plot_predictions_test_wells(df_test)
    model.plot_predictions_velocity_ratio_test_wells(df_test)

    plt.show()
