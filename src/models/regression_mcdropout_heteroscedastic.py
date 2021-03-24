import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    upper_ci = [m + std_multiplier*np.sqrt(v) for m, v in zip(mean, variance)]
    lower_ci = [m - std_multiplier*np.sqrt(v) for m, v in zip(mean, variance) ]

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


class MCDropoutHeteroscedastic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, N, M):
        super(MCDropoutHeteroscedastic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.dropout_rate = 0.10
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # initialize weights and biases (He-initialization)
        self.fc1.weight.data = torch.rand((hidden_dim, input_dim))*np.sqrt(1/input_dim)
        self.fc1.bias.data = torch.zeros(hidden_dim)

        self.fc2.weight.data = torch.rand((hidden_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.fc2.bias.data = torch.zeros(hidden_dim)

        self.fc3.weight.data = torch.rand((output_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.fc3.bias.data = torch.zeros(output_dim)

        self.log_var.weight.data = torch.rand((output_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.log_var.bias.data = torch.zeros(output_dim)

        self.N = N
        self.M = M

        self.num_epochs = 10
        self.precision = 1.0 # TODO: tune this
        self.length_scale = 1  # specifying a standard normal prior for the parameters
        self.reg_dropout = (1-self.dropout_rate)*self.length_scale**2 / (2*self.N*self.precision)
        self.reg_final = self.length_scale**2 / (2*self.N*self.precision)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam([
            {'params': self.fc1.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc2.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc3.parameters(), 'weight_decay': self.reg_final},
            {'params': self.bn1.parameters()},
            {'params': self.bn2.parameters()},
            {'params': self.log_var.parameters()}
        ], lr=self.lr)

    def forward(self, x):
        x_ = self.fc1(x)
        x_ = self.bn1(x_)
        x_ = self.relu(x_)
        x_ = self.dropout(x_)

        x_ = self.fc2(x_)
        x_ = self.bn2(x_)
        x_ = self.relu(x_)
        x_ = self.dropout(x_)

        yhat = self.fc3(x_)
        log_var = self.log_var(x_)

        return yhat, log_var

    def loss(self, y, y_pred, log_var): # negative log-likelihood
        neg_loglik = 0.5*len(y)*np.log(2*np.pi) + 0.5*(log_var.sum() +
                                                       torch.div(torch.pow(y - y_pred, 2), torch.exp(log_var)).sum())

        return neg_loglik

    def train_model(self, train_loader, val_loader):
        train_loss, val_loss = [], []
        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                y_pred, log_var = self.forward(x_train)
                loss = self.loss(y_train, y_pred, log_var)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.item())

            for x_val, y_val in val_loader:
                y_pred_val, log_var_val = self.forward(x_val)
                loss_val = self.loss(y_val, y_pred_val, log_var_val)
            val_loss.append(loss_val.item())

            print("\nEpoch {}; Train loss: {}, Val loss: {}".format(epoch+1, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

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
    M = int(N/batch_size) # number of mini-batches

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N, M=M)

    training_configuration = "mcdropout_heteroscedastic_lr_"+str(model.lr)+"_numepochs_"+str(model.num_epochs)+"_hiddenunits_"\
                             +str(hidden_dim)+"_hiddenlayers_2"+"_batch_size_"+str(batch_size)
    training_configuration = training_configuration.replace(".", "")
    path_to_model = "./data/models/regression/"
    path_to_model += training_configuration + ".pt"
    path_to_losses = "./data/loss/regression"
    path_to_loss = os.path.join(path_to_losses, training_configuration)
    path_to_loss += ".npz"

    train = False
    if train:
        model.train(mode=True)  # keep this on during test time, to obtain probabilistic behaviour
        print("Training MC Dropout model (heteroscedastic)...")
        start_time = time.time()
        training_loss, validation_loss = model.train_model(training_loader, validation_loader)
        end_time = time.time()
        training_time = end_time - start_time
        print("\nTraining time: {:.2f}s".format(training_time))
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=training_loss, validation_loss=validation_loss, training_time=training_time)
    else:
        model.load_state_dict(torch.load(path_to_model))
        print("Trained model loaded..\n")
        with np.load(path_to_loss) as data:
            training_loss = data["training_loss"]
            validation_loss = data["validation_loss"]
            training_time = data["training_time"]

    model.train(mode=True)

    # plot loss curves
    plt.figure()
    plt.plot(range(model.num_epochs), training_loss, label="training")
    plt.plot(range(model.num_epochs), validation_loss, label="validation")
    plt.title("Loss curves, training time {:.2f}s".format(training_time))
    plt.ylabel("Negative log-likelihood")
    plt.xlabel("Epoch")

    # plot predictions and credible intervals for wells in the test set
    wells = list(set(df_test[well_variable]))
    for well in wells:
        df_test_single_well = df_test[df_test[well_variable] == well]
        test_set = create_torch_dataset(df_test_single_well, target_variable, explanatory_variables)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=len(test_set),
                                                  shuffle=False)
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
        plt.title("Well: {}. Coverage probability: {:.2f}".format(well, 100*empirical_coverage))
        plt.ylabel("Depth")
        plt.xlabel("ACS")
        plt.plot(y_test, depths, "-", label="true")
        plt.plot(mean_predictions, depths, "-", label="predicted")
        plt.fill_betweenx(depths, lower_ci_t, upper_ci_t, color="green", alpha=0.2, label="95% CI, total")
        plt.fill_betweenx(depths, lower_ci_e, upper_ci_e, color="red", alpha=0.2, label="95% CI, epistemic")
        plt.ylim([depths.values[-1], depths.values[0]])
        plt.legend(loc="best")

    plt.show()



