import os
import time
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *
from src.models.regression_mcdropout_homoscedastic import MCDropoutHomoscedastic
from src.models.regression_mcdropout_heteroscedastic import MCDropoutHeteroscedastic
from src.SGVB.bayesian_regression_homoscedastic import BayesianRegressorHomoscedastic as SGVBHomoscedastic
from src.SGVB.bayesian_regression_heteroscedastic import BayesianRegressor as SGVBHeteroscedastic

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("..")
    print(os.getcwd())

    ######################### Load dataset and models ##################################################################
    df_train = pd.read_csv("./data/train_regression.csv", sep=";")
    df_val = pd.read_csv("./data/val_regression.csv", sep=";")
    df_test = pd.read_csv("./data/test_regression.csv", sep=";")

    variables = df_train.columns.values
    target_variable = "ACS"
    well_variable = "well_name"
    explanatory_variables = [var for var in variables if var not in [target_variable, well_variable]]

    validation_set = create_torch_dataset(df_val, target_variable, explanatory_variables)
    test_set = create_torch_dataset(df_test, target_variable, explanatory_variables)


    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    batch_size = 100
    dropout_rate = 0.10
    # Choose model type to train
    heteroscedastic = False
    mcdropout = True

    # iterate over fractions of full training dataset
    fractions = np.linspace(0.10, 0.90, num=9)
    for f in fractions:
        # create training dataset from df_train
        training_df = df_train.sample(frac=f)

        training_set = create_torch_dataset(training_df, target_variable, explanatory_variables)
        N = len(training_set)
        M = int(N / batch_size)  # number of mini-batches

        if mcdropout:
            training_configuration = "mcdropout_"
            model_type = "MC Dropout"
        else:
            training_configuration = "sgvb_"
            model_type = "SGVB"

        # initialize model objects
        if heteroscedastic:
            title = "Heteroscedastic"
            training_configuration += title.lower() + "_"
            if model_type == "MC Dropout":
                model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim,
                                                 output_dim=output_dim, N=N, M=M, dropout_rate=dropout_rate)
            else:
                model = SGVBHeteroscedastic(in_size=input_dim, hidden_size=hidden_dim,
                                            out_size=output_dim, n_batches=M)
        else:
            title = "Homoscedastic"
            training_configuration += title.lower() + "_"
            if model_type == "MC Dropout":
                model = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim,
                                               output_dim=output_dim, N=N, M=M, dropout_rate=dropout_rate)
            else:
                model = SGVBHomoscedastic(in_size=input_dim, hidden_size=hidden_dim,
                                          out_size=output_dim, n_batches=M)

        training_configuration += "dropout_" + str(model.dropout_rate) + "_lr_" + str(model.lr) + "_numepochs_" + str(
            model.num_epochs) + "_hiddenunits_" \
                                  + str(hidden_dim) + "_hiddenlayers_2" + "_batch_size_" + str(batch_size)
        training_configuration = training_configuration.replace(".", "")
        path_to_model = "./data/models/regression/varying_training_set_size/"
        path_to_loss = "./data/loss/regression/varying_training_set_size/"
        path_to_model += "dropout"+str(dropout_rate).replace(".", "")+"/"
        path_to_model += "size"+str(int(100*f))+"/"
        path_to_loss += "size"+str(int(100*f))+"/"

        # make directories is not present
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
        if not os.path.exists(path_to_loss):
            os.makedirs(path_to_loss)

        path_to_model += training_configuration + ".pt"
        path_to_loss += training_configuration + ".npz"

        dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                                test_set=test_set, batch_size=batch_size)

        training_loader = dataloader.training_loader()
        validation_loader = dataloader.validation_loader()
        test_loader = dataloader.test_loader()

        model.train(mode=True)
        print("Training model: {}, with fraction {} of training set...".format(training_configuration, f))
        start_time = time.time()
        training_loss, validation_loss = model.train_model(training_loader, validation_loader)
        end_time = time.time()
        training_time = end_time - start_time

        # Save model, loss and training time to file
        torch.save(model.state_dict(), path_to_model)
        np.savez(path_to_loss, training_loss=training_loss, validation_loss=validation_loss, training_time=training_time)

        # plot loss curves
        plt.figure()
        plt.plot(range(model.num_epochs), training_loss, label="training")
        plt.plot(range(model.num_epochs), validation_loss, label="validation")
        plt.title("Loss curves, training time {:.2f}s".format(training_time))
        plt.ylabel("Negative log-likelihood")
        plt.xlabel("Epoch")
    plt.show()