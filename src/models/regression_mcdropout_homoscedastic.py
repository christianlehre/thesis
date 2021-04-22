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


class MCDropoutHomoscedastic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, N, M, dropout_rate):
        super(MCDropoutHomoscedastic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # initialize homoscedastic variance - treated as model parameter to be optimized during training
        self.log_var = nn.Parameter(
            torch.FloatTensor(1,).normal_(mean=-2.5, std=0.001), requires_grad=True
        )

        # initialize weights and biases (He-initialization)
        self.fc1.weight.data = torch.rand((hidden_dim, input_dim)) * np.sqrt(1 / input_dim)
        self.fc1.bias.data = torch.zeros(hidden_dim)

        self.fc2.weight.data = torch.rand((hidden_dim, hidden_dim)) * np.sqrt(2 / hidden_dim)
        self.fc2.bias.data = torch.zeros(hidden_dim)

        self.fc3.weight.data = torch.rand((output_dim, hidden_dim)) * np.sqrt(2 / hidden_dim)
        self.fc3.bias.data = torch.zeros(output_dim)

        self.num_epochs = 10

        self.precision = 1.0  # TODO: tune this
        self.length_scale = 1  # standard normal prior on the parameters
        self.N = N
        self.M = M
        self.reg_dropout = (1-self.dropout_rate)*self.length_scale**2 / (2*self.N*self.precision)
        self.reg_final = self.length_scale**2 / (2*self.N*self.precision)
        self.lr = 0.0001
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.fc1.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc2.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc3.parameters(), 'weight_decay': self.reg_final},
            {'params': self.bn1.parameters()},
            {'params': self.bn2.parameters()},
            {'params': self.log_var}
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

        output = self.fc3(x_)
        return output

    def loss(self, y, y_pred): # negative log-likelihood
        neg_loglik = 0.5*len(y)*(np.log(2*np.pi) + self.log_var) + 0.5*torch.div(torch.pow(y - y_pred, 2), torch.exp(self.log_var)).sum()
        return neg_loglik/self.precision

    def train_model(self, train_loader, val_loader):
        train_loss, val_loss = [], []
        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                y_pred = self.forward(x_train)
                loss = self.loss(y_train, y_pred)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.item())

            for x_val, y_val in val_loader:
                y_pred_val = self.forward(x_val)
                loss_val = self.loss(y_val, y_pred_val)
            val_loss.append(loss_val.item())

            print("\nEpoch {}; Train loss: {}, Val loss: {}".format(epoch+1, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

    def evaluate_performance(self, test_loader, B=100):
        x_test, y_test = unpack_dataset(test_loader)
        mse, mae = [], []
        for _ in range(B):
            predictions = self.forward(x_test)
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
        log_var_B = np.zeros(B)
        for b in range(B):
            predictions = self.forward(x_test)
            predictions_B[b, :] = predictions.detach().flatten().numpy()
            log_var_B[b] = self.log_var.detach().numpy()
        mean_predictions = np.mean(predictions_B, axis=0)
        mean_log_var = np.mean(log_var_B)
        var_epistemic = np.std(predictions_B, axis=0)**2
        var_aleatoric = np.exp(mean_log_var)*np.ones_like(var_epistemic)
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
    dropout_rate = 0.50


    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    model = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N, M=M,
                                   dropout_rate=dropout_rate)

    training_configuration = "mcdropout_homoscedastic_dropout_"+str(model.dropout_rate)+"_lr_"+str(model.lr)+"_numepochs_"+str(model.num_epochs)+"_hiddenunits_"\
                             +str(hidden_dim)+"_hiddenlayers_2"+"_batch_size_"+str(batch_size)
    training_configuration = training_configuration.replace(".", "")
    path_to_model = "./data/models/regression/"
    path_to_model += training_configuration + ".pt"
    path_to_losses = "./data/loss/regression"
    path_to_loss = os.path.join(path_to_losses, training_configuration)
    path_to_loss += ".npz"

    train = True
    if train:
        model.train(mode=True)
        print("Training MC Dropout model (homoscedastic)...")
        start_time = time.time()
        training_loss, validation_loss = model.train_model(training_loader, validation_loader)
        end_time = time.time()
        training_time = end_time - start_time
        print("\nTraining time: {:.2f}s".format(training_time))
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=training_loss, validation_loss=validation_loss, training_time=training_time)
    else:
        model.load_state_dict(torch.load(path_to_model))
        print("\nTrained model loaded.")
        with np.load(path_to_loss) as data:
            training_loss = data["training_loss"]
            validation_loss = data["validation_loss"]
            training_time = data["training_time"]

    # loss curves
    plt.figure()
    plt.plot(range(model.num_epochs), training_loss, label="training")
    plt.plot(range(model.num_epochs), validation_loss, label="validation")
    plt.title("Loss curves, training time {:.2f}s".format(training_time))
    plt.ylabel("Negative log-likelihood")
    plt.xlabel("Epoch")

    """
    # Predictions and credible intervals for wells in test set
    wells = list(set(df_test[well_variable]))
    for well in wells:
        df_single_well = df_test[df_test[well_variable] == well]
        test_set = create_torch_dataset(df_single_well, target_variable, explanatory_variables)
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
        depths = df_single_well["DEPTH"]

        plt.figure(figsize=(6, 10))
        plt.title("Well: {}. Coverage probability {:.2f}".format(well, 100*empirical_coverage))
        plt.ylabel("Depth")
        plt.xlabel("ACS")
        plt.plot(y_test, depths, "-", label="true")
        plt.plot(mean_predictions, depths, label="predicted")
        plt.fill_betweenx(depths, lower_ci_t, upper_ci_t, color="green", alpha=0.2, label="95% CI total")
        plt.fill_betweenx(depths, lower_ci_e, upper_ci_e, color="red", alpha=0.2, label="95% CI epistemic")
        plt.ylim([depths.values[-1], depths.values[0]])
        plt.legend(loc="best")
    plt.show()
    """