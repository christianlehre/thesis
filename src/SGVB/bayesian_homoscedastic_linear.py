import pandas as pd
import numpy as np
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt
from src.SGVB.bayesianlinear import BayesianLinear

from src.dataloader.dataloader import Dataloader
from src.utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class KL:
    accumulated_kl_div = 0


class BayesianRegressorHomoscedastic(nn.Module):
    def __init__(self, in_size, out_size, n_batches, dropout_rate):
        super().__init__()
        self.kl_loss = KL

        # architecture - single layer for linear model, no activation
        self.bfc1 = BayesianLinear(in_size, out_size, self.kl_loss, n_batches, prior_mu=0, prior_sigma=1)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # initialize homoscedastic variance
        self.log_var = nn.Parameter(torch.FloatTensor(1,).normal_(mean=-2.5, std=0.001), requires_grad=True)

        self.num_epochs= 10
        self.n_batches = n_batches

        self.lr = 0.001
        self.optimizer=torch.optim.Adam(self.parameters(), lr=self.lr)

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def forward(self, x):
        x_ = self.dropout(x)
        output = self.bfc1(x_)

        return output

    def det_loss(self, y, y_pred):
        neg_loglik = 0.5*len(y)*(np.log(2*np.pi) + self.log_var) + torch.div(torch.pow(y-y_pred, 2), 2*torch.exp(self.log_var)).sum()

        kl = self.accumulated_kl_div
        self.reset_kl_div()
        return neg_loglik + kl

    def train_model(self, train_loader, val_loader):
        train_loss = []
        val_loss = []
        for epoch in range(self.num_epochs):
            for x_train, y_train in train_loader:
                x_train = x_train.requires_grad_()
                self.optimizer.zero_grad()
                y_pred = self.forward(x_train)
                loss = self.det_loss(y_train, y_pred)
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss.item())

            for x_val, y_val in val_loader:
                y_pred_val = self.forward(x_val)
                loss_val = self.det_loss(y_val, y_pred_val)
            val_loss.append(loss_val.item())
            print("\nEpoch {}; Train loss: {}, Val loss: {}".format(epoch+1, train_loss[epoch], val_loss[epoch]))

        return train_loss, val_loss

    def print_trainable_parameters(self):
        print("\nTrainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

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
    output_dim = 1
    batch_size = 100
    dropout_rate = 0.10
    N = len(training_set)
    M = int(N/batch_size)

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set, test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    model = BayesianRegressorHomoscedastic(in_size=input_dim, out_size=output_dim, n_batches=M, dropout_rate=dropout_rate)

    training_configuration = "sgvb_homoscedastic_linear"
    path_to_model = "./data/models/regression/"
    path_to_model += training_configuration + ".pt"
    path_to_losses = "./data/loss/regression"
    path_to_loss = os.path.join(path_to_losses, training_configuration)
    path_to_loss += ".npz"

    train = True
    if train:
        model.train(mode=True)
        print("Training Bayesian neural network...")
        start_time = time.time()
        training_loss, validation_loss = model.train_model(training_loader, validation_loader)
        end_time = time.time()
        training_time = end_time - start_time
        print("Training time: {:.2f} s".format(training_time))
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=training_loss, validation_loss=validation_loss, training_time=training_time)
    else:
        model.load_state_dict(torch.load(path_to_model))
        print("Trained model loaded..")
        with np.load(path_to_loss) as data:
            training_loss = data["training_loss"]
            validation_loss = data["validation_loss"]
            training_time = data["training_time"]

    model.train(mode=False)

    mse, mae = model.evaluate_performance(test_loader, B=100)
    print("Performance over full test set:")
    print("MSE: {:.5f} +/- {:.5f}".format(mse[0], mse[1]))
    print("MAE: {:.5f} +/- {:.5f}".format(mae[0], mae[1]))

    # Training curves
    plt.figure()
    plt.plot(range(model.num_epochs), training_loss, label="training")
    plt.plot(range(model.num_epochs), validation_loss, label="validation")
    plt.title("Loss curves - Homoscedastic SGVB", fontsize=18)
    plt.ylabel("ELBO loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

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
        print("MSE: {:.5f} +/- {:.5f}".format(mse[0], mse[1]))
        print("MAE: {:.5f} +/- {:.5f}".format(mae[0], mae[1]))

        mean_predictions, var_epistemic, var_aleatoric, var_total = model.aleatoric_epistemic_variance(test_loader,
                                                                                                       B=100)
        lower_ci_e, upper_ci_e = credible_interval(mean_predictions, var_epistemic, std_multiplier=2)
        lower_ci_t, upper_ci_t = credible_interval(mean_predictions, var_total, std_multiplier=2)
        empirical_coverage = coverage_probability(y_test, lower_ci_t, upper_ci_t)
        depths = df_single_well["DEPTH"]
        plt.figure(figsize=(8, 12))
        plt.title("Well: {}. Coverage probability {:.2f}%".format(well, 100 * empirical_coverage), fontsize=18)
        plt.ylabel("Depth", fontsize=16)
        plt.xlabel("ACS", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(y_test, depths, "-", label="True")
        plt.plot(mean_predictions, depths, label="Prediction")
        plt.fill_betweenx(depths, lower_ci_t, upper_ci_t, color="green", alpha=0.2, label="95% CI total")
        plt.fill_betweenx(depths, lower_ci_e, upper_ci_e, color="red", alpha=0.2, label="95% CI epistemic")
        plt.ylim([depths.values[-1], depths.values[0]])
        plt.legend(loc="best", fontsize=12)
        #if well == "30/8-5 T2":
        #    plt.legend(loc="lower right", fontsize=12)
        # set x-lim for different wells:
        #if well == "25/4-10 S":
        #    plt.xlim([-5, 7])
        #elif well == "25/7-6":
        #    plt.xlim([-4, 4])
        #elif well == "30/8-5 T2":
        #    plt.xlim([-5, 7])
        #elif well == "30/6-26":
        #    plt.xlim([-5, 9])
        #elif well == "30/11-10":
        #    plt.xlim([-7, 7])
        #elif well == "30/11-7":
        #    plt.xlim([-7, 9])
        #elif well == "30/11-9 ST2":
        #    plt.xlim([-4, 8])
        #else:
        #    plt.xlim([-5, 11])
    plt.show()