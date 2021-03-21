import torch
import pandas as pd
import numpy as np
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt
from torchinfo import summary
from bayesianlinear import BayesianLinear
from src.dataloader.dataloader import Dataloader

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


def unpack_dataset(dataloader_object):
    for x, y in dataloader_object:
        x, y = x, y
    return x, y

def credible_interval(mean, variance, std_multiplier):
    upper_ci = [m + np.sqrt(v)*std_multiplier for m, v in zip(mean, variance)]
    lower_ci = [m - np.sqrt(v)*std_multiplier for m, v in zip(mean, variance)]

    return lower_ci, upper_ci


def coverage_probability(test_loader, lower_ci, upper_ci):
    _, y_test = unpack_dataset(test_loader)
    num_samples = len(y_test)
    num_samples_inside_ci = 0

    for i in range(num_samples):
        if upper_ci[i] > y_test[i] > lower_ci[i]:
            num_samples_inside_ci += 1
    coverage = num_samples_inside_ci / num_samples

    return coverage


class KL:
    accumulated_kl_div = 0

class BayesianRegressor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_batches):
        super().__init__()
        self.kl_loss = KL

        # define layers of the network, call sequentially in forward
        self.bfc1 = BayesianLinear(in_size, hidden_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.bfc2 = BayesianLinear(hidden_size, hidden_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.bfc3 = BayesianLinear(hidden_size, out_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.bfc_logvar = BayesianLinear(hidden_size, out_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_size, track_running_stats=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_size, track_running_stats=False)

        self.num_epochs = 10
        self.n_batches = n_batches

        self.lr = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x_ = self.bfc1(x)
        x_ = self.batchnorm1(x_)
        x_ = self.relu(x_)

        x_ = self.bfc2(x_)
        x_ = self.batchnorm2(x_)
        x_ = self.relu(x_)

        y_hat = self.bfc3(x_)
        log_var = self.bfc_logvar(x_)

        return y_hat, log_var

    def det_loss(self, y, y_pred, log_var):
        reconstruction_error = 0.5*len(y)*np.log(2*np.pi) + (torch.log(torch.sqrt(torch.exp(log_var)))).sum() + \
                               0.5*(torch.div(torch.pow(y-y_pred, 2), torch.exp(log_var))).sum()
        kl = self.accumulated_kl_div
        self.reset_kl_div()
        return reconstruction_error + kl

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def train_model(self, train_loader, val_loader):
        train_loss = []
        val_loss = []
        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                y_pred, log_var = self.forward(x_train)
                loss = self.det_loss(y_train, y_pred, log_var)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.item())

            #print("\nChecking for finite gradients:")
            #for name, param in model.named_parameters():
            #    print(name, torch.isfinite(param.grad).all())

            for x_val, y_val in val_loader:
                output_val, log_var_val = self.forward(x_val)
                loss_val = self.det_loss(y_val, output_val, log_var_val)
            val_loss.append(loss_val.item())
            print("\nEpoch {}; Train loss: {}, Val loss: {}".format(epoch+1, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

    def print_trainable_parameters(self):
        print("Trainable parameters: ")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def evaluate_performance(self, test_loader, B=100):
        x_test, y_test = unpack_dataset(test_loader)
        mse, mae = [], []
        for _ in range(B):
            predictions, _ = self.forward(x_test)
            predictions = predictions.detach()
            mse_test = torch.mean(torch.pow(predictions - y_test, 2))
            mae_test = torch.mean(torch.abs(predictions - y_test))

            mse.append(mse_test.item())
            mae.append(mae_test.item())

        mean_mse = np.mean(mse)
        std_mse = np.std(mse)
        mse_tuple = (mean_mse, std_mse)

        mean_mae = np.mean(mae)
        std_mae = np.std(mae)
        mae_tuple = (mean_mae, std_mae)

        return mse_tuple, mae_tuple

    def aleatoric_epistemic_variance(self, test_loader, B=100):
        x_test, y_test = unpack_dataset(test_loader)
        predictions_B = np.zeros((B, len(y_test)))
        log_vars_B = np.zeros_like(predictions_B)
        for b in range(B):
            predictions, log_var = self.forward(x_test)
            predictions_B[b, :] = predictions.detach().flatten().numpy()
            log_vars_B[b, :] = log_var.detach().flatten().numpy()
        mean_predictions = np.mean(predictions_B, axis=0)
        mean_log_var = np.mean(log_vars_B, axis=0)
        var_epistemic = np.std(predictions_B, axis=0)**2
        var_aleatoric = np.exp(mean_log_var)
        var_total = var_epistemic + var_aleatoric

        return mean_predictions, var_epistemic, var_aleatoric, var_total


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("../..")
    print(os.getcwd())

    df_train = pd.read_csv("./data/train_regression.csv", sep=";")
    df_val = pd.read_csv("./data/val_regression.csv", sep=";")
    df_test = pd.read_csv("./data/test_regression.csv", sep=";")

    variables = df_train.columns.values
    target_variable = "ACS"
    well_variable = "well_name"
    explanatory_variables = [var for var in variables if var not in [target_variable, well_variable]]
    training_set = create_torch_dataset(df_train, target_variable, explanatory_variables)
    validation_set = create_torch_dataset(df_val, target_variable, explanatory_variables)
    test_set = create_torch_dataset(df_test, target_variable, explanatory_variables)

    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    batch_size = 100
    N = len(training_set)
    M = int(N/batch_size)  # number of mini-batches in training set

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set, test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    model = BayesianRegressor(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)

    training_conf = "bayesian_heteroscedastic_lr"+str(model.lr)+"_numepochs_"+str(model.num_epochs)+"_hiddenunits_"\
                    +str(hidden_dim)+"_hiddenlayers_2"+"_batchsize_"+str(batch_size)
    training_conf = training_conf.replace(".", "")
    path_to_model = "./data/models/regression/"
    path_to_model += training_conf + ".pt"
    path_to_losses = "./data/loss/regression"
    path_to_loss = os.path.join(path_to_losses, training_conf)
    path_to_loss += ".npz"

    train = False
    if train:
        model.train(mode=True)
        print("Training Bayesian neural network...")
        start_time = time.time()
        train_loss, val_loss = model.train_model(training_loader, validation_loader)
        end_time = time.time()
        training_time = end_time - start_time
        print("Training time: {:.2f} s".format(training_time))
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=train_loss, validation_loss=val_loss, training_time=training_time)
    else:
        model.load_state_dict(torch.load(path_to_model))
        print("Trained model loaded..\n")
        with np.load(path_to_loss) as data:
            train_loss = data["training_loss"]
            val_loss = data["validation_loss"]
            training_time = data["training_time"]

    model.train(mode=True)

    plt.figure()
    plt.plot(range(model.num_epochs), train_loss, label="training")
    plt.plot(range(model.num_epochs), val_loss, label="validation")
    plt.title("Loss curves, training time {:.2f}s".format(training_time))
    plt.ylabel("ELBO loss")
    plt.xlabel("Epoch")

    # Plot predictions and credible intervals for wells in the test set
    wells = list(set(df_test[well_variable]))
    for well in wells:
        df_test_single_well = df_test[df_test[well_variable] == well]
        test_set = create_torch_dataset(df_test_single_well, target_variable, explanatory_variables)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=len(test_set),
                                                  shuffle=False,
                                                  )
        x_test, y_test = unpack_dataset(test_loader)
        mse, mae = model.evaluate_performance(test_loader, B=100)
        print("Performance metrics for well {}".format(well))
        print("MSE: {:.3f} +/- {:.3f}".format(mse[0], mse[1]))
        print("MAE: {:.3f} +/- {:.3f}".format(mae[0], mae[1]))

        mean_predictions, var_epistemic, var_aleatoric, var_total = model.aleatoric_epistemic_variance(test_loader, B=100)
        lower_ci_e, upper_ci_e = credible_interval(mean_predictions, var_epistemic, std_multiplier=2)
        lower_ci_t, upper_ci_t = credible_interval(mean_predictions, var_total, std_multiplier=2)
        empirical_coverage = coverage_probability(test_loader, lower_ci_t, upper_ci_t)
        depths = df_test_single_well["DEPTH"]
        plt.figure(figsize=(6, 10))
        plt.title("Well: {}. Coverage probability: {:.2f}%".format(well, 100 * empirical_coverage))
        plt.ylabel("Depth")
        plt.xlabel("ACS")
        plt.plot(y_test, depths, "-", label="true")
        plt.plot(mean_predictions, depths, "-", label="predicted")
        plt.fill_betweenx(depths, lower_ci_t, upper_ci_t, color="green", alpha=0.2, label="95% CI, total")
        plt.fill_betweenx(depths, lower_ci_e, upper_ci_e, color="red", alpha=0.2, label="95% CI, epistemic")
        plt.ylim([depths.values[-1], depths.values[0]])
        plt.legend(loc="best")

    plt.show()
