import torch
import pandas as pd
import numpy as np
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt
from torchinfo import summary
from bayesianlinear import BayesianLinear


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


class KL:
    accumulated_kl_div = 0

class BayesianRegressor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_batches):
        super().__init__()
        self.kl_loss = KL

        #self.layers = nn.Sequential(
        #    BayesianLinear(in_size, hidden_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1),
        #    nn.ReLU(),
        #    BayesianLinear(hidden_size, out_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        #)

        # define layers of the network, call sequentially in forward
        self.bfc1 = BayesianLinear(in_size, hidden_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.bfc2 = BayesianLinear(hidden_size, out_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.relu = nn.ReLU()

        self.num_epochs = 10
        self.n_batches = n_batches

        self.lr = 0.01
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.bfc1(x)
        x = self.relu(x)
        x = self.bfc2(x)

        return x

    def det_loss(self, y, y_pred):
        reconstruction_error = -torch.distributions.normal.Normal(y_pred, 0.1).log_prob(y).sum()  # self.criterion??
        kl = self.accumulated_kl_div
        self.reset_kl_div()
        return reconstruction_error + kl

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def train_model(self, train_loader):
        train_loss = []

        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                #x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                y_pred = self.forward(x_train)
                loss = self.det_loss(y_train, y_pred)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.item())
            print("Epoch {}; Train loss: {}".format(epoch, train_loss[epoch]))
        return train_loss

    def print_trainable_parameters(self):
        print("Trainable parameters: ")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("../..")
    print(os.getcwd())

    df_train = pd.read_csv("./data/train_regression.csv", sep=";")
    df_test = pd.read_csv("./data/test_regression.csv", sep=";")
    variables = df_train.columns.values
    target_variable = "ACS"
    well_variable = "well_name"
    explanatory_variables = [var for var in variables if var not in [target_variable, well_variable]]

    training_set = create_torch_dataset(df_train, target_variable, explanatory_variables)

    # test for a single well
    wells = list(set(df_test["well_name"]))
    df_test_single_well = df_test[df_test["well_name"] == wells[0]]
    test_set = create_torch_dataset(df_test_single_well, target_variable, explanatory_variables)

    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    batch_size = 100
    N = len(training_set)
    M = int(N/batch_size) # number of mini-batches in training set

    training_loader = torch.utils.data.DataLoader(dataset=training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=len(test_set),
                                              shuffle=False,
                                              )

    for x, y in test_loader:
        x_test = x
        y_test = y

    model = BayesianRegressor(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)

    train = False
    if train:
        print("Training Bayesian neural network...")
        start_time = time.time()
        train_loss = model.train_model(training_loader)
        end_time = time.time()
        print("Training time: {:.2f} s".format(end_time - start_time))
        torch.save(model.state_dict(), "./data/models/regression/regression_bayesian_test.pt")
    else:
        model.load_state_dict(torch.load("./data/models/regression/regression_bayesian_test.pt"))
        print("Trained model loaded..\n")

    model.print_trainable_parameters()
    print("")
    summary(model, input_size=(1, input_dim), verbose=2, col_names=["kernel_size", "output_size", "num_params"])

    B = 100
    predictions_B = np.zeros((B, len(y_test)))
    for b in range(B):
        predictions = model.forward(x_test).detach()
        predictions_B[b, :] = predictions.numpy()

    q_05, q_95 = np.nanquantile(predictions_B, [0.05, 0.95], axis=0)
    depths = df_test_single_well["DEPTH"]
    plt.figure(figsize=(16, 6))
    plt.plot(depths, y_test, label="true")
    plt.plot(depths, predictions_B.mean(axis=0), label="predicted")
    plt.fill_between(depths, q_05, q_95, alpha=0.2)
    plt.legend(loc="best")

    plt.show()
