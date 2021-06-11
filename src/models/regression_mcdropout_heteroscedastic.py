import os
import time
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *

class MCDropoutHeteroscedastic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, N, M, dropout_rate):
        super(MCDropoutHeteroscedastic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.log_var = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # initialize weights and biases (He-initialization)
        self.fc1.weight.data = torch.rand((hidden_dim, input_dim))*np.sqrt(1/input_dim)
        self.fc1.bias.data = torch.zeros(hidden_dim)

        self.fc2.weight.data = torch.rand((hidden_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.fc2.bias.data = torch.zeros(hidden_dim)

        self.fc3.weight.data = torch.rand((output_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.fc3.bias.data = torch.zeros(output_dim)

        self.log_var.weight.data = torch.randn((output_dim, hidden_dim))*np.sqrt(2/hidden_dim)
        self.log_var.bias.data = torch.zeros(output_dim)

        self.N = N
        self.M = M

        self.num_epochs = 10
        self.precision = 1.0
        self.length_scale = 1  # specifying a standard normal prior for the parameters
        self.reg_dropout = (1-self.dropout_rate)*self.length_scale**2 / (2*self.N*self.precision)
        self.reg_final = self.length_scale**2 / (2*self.N*self.precision)
        self.lr = 0.0001
        self.optimizer = torch.optim.Adam([
            {'params': self.fc1.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc2.parameters(), 'weight_decay': self.reg_dropout},
            {'params': self.fc3.parameters(), 'weight_decay': self.reg_final},
            {'params': self.bn1.parameters()},
            {'params': self.bn2.parameters()},
            {'params': self.log_var.parameters()}
        ], lr=self.lr)

    def forward(self, x):
        """
        Forward pass of the model, calculating output of the model

        :param input: matrix of samples, dim = (num samples, num predictors)
        :return: tuple of prediction and log-variance of the multi-headed network (prediction, log_variance)
        """
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
        """
        Compute heteroscedastic NLL loss

        :param y: true value of response
        :param y_pred: predicted value of response
        :param log_var: log-variance (aleatoric)
        :return: NLL loss (scalar)
        """
        neg_loglik = 0.5*len(y)*np.log(2*np.pi) + 0.5*(log_var.sum() +
                                                       torch.div(torch.pow(y - y_pred, 2), torch.exp(log_var)).sum())

        return neg_loglik

    def train_model(self, train_loader, val_loader):
        """
        Train the model using the partial training set,i.e. excluding the validation set,
        estimate the test loss during training using the validation set

        :param train_loader: torch dataloader object for the training set, excluding the validation set
        :param val_loader: torch dataloader object for the validation set
        :return: tuple containig the training and validation loss for each epoch
        """
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
        """
        Evaluate model performance on the full test set in terms of the following performance metrics
        - MSE
        - MAE

        :param test_loader: torch dataloader object for the test set
        :param B: number of stochastic forward passes
        :return: triplet containing the performance metrics, (mse, mae, mape)
        """
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
        """
        Estimate epistemic, aleatoric and total predictive uncertainty

        :param test_loader: torch dataloader object containing the test set
        :param B: Number of stochastic forward passes
        :return: mean predictions, epistemic variance, aleatoric variance, total predictive variance
        """
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
    dropout_rate = 0.10

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    training_loader = dataloader.training_loader()
    validation_loader = dataloader.validation_loader()
    test_loader = dataloader.test_loader()

    model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N, M=M,
                                     dropout_rate=dropout_rate)
    training_configuration = "mcdropout_heteroscedastic_dropout_"+str(model.dropout_rate)+"_lr_"+str(model.lr)+"_numepochs_"+str(model.num_epochs)+"_hiddenunits_"\
                             +str(hidden_dim)+"_hiddenlayers_3"+"_batch_size_"+str(batch_size)
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

    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
        else:
            m.eval()

    # plot loss curves
    plt.figure()
    epochs = range(model.num_epochs)
    epochs = list(map(lambda x: x+1, epochs))
    plt.xticks(epochs)
    plt.plot(epochs, training_loss, label="Training")
    plt.plot(epochs, validation_loss, label="Validation")
    plt.ylabel("Negative log-likelihood", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    mse, mae = model.evaluate_performance(test_loader, B=100)
    print("Performance over full test set:")
    print("MSE: {:.5f} +/- {:.5f}".format(mse[0], mse[1]))
    print("MAE: {:.5f} +/- {:.5f}".format(mae[0], mae[1]))

    zoomed_out = True

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
        print("MSE: {:.5f} +/- {:.5f}".format(mse[0], mse[1]))
        print("MAE: {:.5f} +/- {:.5f}".format(mae[0], mae[1]))

        mean_predictions, var_epistemic, var_aleatoric, var_total = model.aleatoric_epistemic_variance(test_loader, B=100)
        lower_ci_e, upper_ci_e = credible_interval(mean_predictions, var_epistemic, std_multiplier=2)
        lower_ci_t, upper_ci_t = credible_interval(mean_predictions, var_total, std_multiplier=2)
        empirical_coverage = coverage_probability(y_test, lower_ci_t, upper_ci_t)
        depths = df_test_single_well["DEPTH"]

        # Save data for plotting prediction curves
        path_to_prediction_curves = "./data/prediction_curves/MCDropout/Heteroscedastic/"
        well_name = well.replace(" ", "")
        well_name = well_name.replace("/", "")
        path_to_prediction_curves += well_name + ".npz"

        np.savez(path_to_prediction_curves, predictions=mean_predictions,
                 epistemic_ci=(lower_ci_e, upper_ci_e),
                 total_ci=(lower_ci_t, upper_ci_t),
                 depths=depths,
                 y_test=y_test,
                 empirical_coverage=empirical_coverage,
                 well=well)
    plt.show()



