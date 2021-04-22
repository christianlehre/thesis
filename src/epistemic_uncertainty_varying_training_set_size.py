import os
import ast
import time
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *
from src.models.regression_mcdropout_homoscedastic import MCDropoutHomoscedastic
from src.models.regression_mcdropout_heteroscedastic import MCDropoutHeteroscedastic
from src.SGVB.bayesian_regression_homoscedastic import BayesianRegressorHomoscedastic as SGVBHomoscedastic
from src.SGVB.bayesian_regression_heteroscedastic import BayesianRegressor as SGVBHeteroscedastic


def nested_dictionary(test_loader,input_dim, hidden_dim, output_dim, N, M, path_to_models="./data/models/regression/varying_training_set_size"):
    uncertainty_over_all_training_size = {}
    # Iterate over training set size
    for dir in os.listdir(path_to_models):
        if dir.startswith("."):
            continue

        path_to_models_with_same_training_sets = os.path.join(path_to_models, dir)
        uncertainty_per_training_size = {}
        # Iterate over models
        for model_path in os.listdir(path_to_models_with_same_training_sets):
            # Initialize model
            if model_path.startswith("mcdropout_homoscedastic"):
                model = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N, M=M)
                model.dropout_rate = 0.50
                model_type = "mcdropout_homoscedastic"
            elif model_path.startswith("mcdropout_heteroscedastic"):
                model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N, M=M)
                model.dropout_rate = 0.50
                model_type = "mcdropout_heteroscedastic"
            elif model_path.startswith("sgvb_homoscedastic"):
                model = SGVBHomoscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)
                model_type = "sgvb_homoscedastic"
            else:
                model = SGVBHeteroscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)
                model_type = "sgvb_heteroscedastic"

            path_to_model = os.path.join(path_to_models_with_same_training_sets, model_path)

            # Load models and extract epistemic uncertainty
            model.load_state_dict(torch.load(path_to_model))

            model.eval()
            for m in model.modules():
                if m.__class__.__name__.startswith("Dropout") and model_type.startswith("mcdropout"):
                    m.train()
                else:
                    m.eval()

            _, var_epistemic, _, _ = model.aleatoric_epistemic_variance(test_loader, B=100)
            std_epistemic = np.sqrt(var_epistemic)

            uncertainty_per_training_size[model_type] = np.mean(std_epistemic)

        uncertainty_over_all_training_size[dir] = uncertainty_per_training_size

    return uncertainty_over_all_training_size


def save_dictionary(dictionary, path_to_dictionary):
    try:
        file = open(path_to_dictionary, 'w')
        file.write(str(dictionary))
        file.close()
    except:
        print("Unable to write to file")
    return


def load_dictionary(path_to_dictionary):
    file = open(path_to_dictionary, "r")
    content = file.read()
    dictionary = ast.literal_eval(content)
    file.close()
    return dictionary

def plot_nested_dict(dictionary):
    plt.figure(figsize=(12, 8))

    inner_keys = list(dictionary.values())[0].keys()
    x_axis_values = list(map(str, dictionary.keys()))
    for model in inner_keys:
        y_axis_values = [v[model] for v in dictionary.values()]

        plt.plot(x_axis_values, y_axis_values, label=model.replace("_", " ").title(), linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Fraction of training set", fontsize=20)
    plt.ylabel("Epistemic uncertainty", fontsize=20)
    plt.title("Epistemic uncertainty for varying size of training set", fontsize=24)
    plt.legend(fontsize=16)

if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("..")
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
    M = int(N / batch_size)  # number of mini-batches

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    test_loader = dataloader.test_loader()

    path_to_dictionary = "./data/epistemic_uncertainty/dropout_010_epistemic_uncertainty_varying_training_set_size.txt"

    create_dict = False

    if create_dict:
        uncertainty_dict = nested_dictionary(test_loader, input_dim, hidden_dim, output_dim, N, M)
        save_dictionary(uncertainty_dict, path_to_dictionary)
    else:
        uncertainty_dict = load_dictionary(path_to_dictionary)


    # sort dict in ascending fractino of full training set
    dict_to_plot = {}
    dict_to_plot["10%"] = uncertainty_dict['size10']
    dict_to_plot["20%"] = uncertainty_dict["size20"]
    dict_to_plot["30%"] = uncertainty_dict["size30"]
    dict_to_plot["40%"] = uncertainty_dict["size40"]
    dict_to_plot["50%"] = uncertainty_dict["size50"]
    dict_to_plot["60%"] = uncertainty_dict["size60"]
    dict_to_plot["70%"] = uncertainty_dict["size70"]
    dict_to_plot["80%"] = uncertainty_dict["size80"]
    dict_to_plot["90%"] = uncertainty_dict["size90"]
    dict_to_plot["100%"] = uncertainty_dict["size100"]

    plot_nested_dict(dict_to_plot)
    plt.tight_layout()
    plt.savefig("../../Figures/epistemic_uncertainty_varying_training_set_size.pdf")
    plt.show()
