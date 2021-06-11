import os
import ast
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *

# import linear models
from src.SGVB.bayesian_homoscedastic_linear import BayesianRegressorHomoscedastic as Homoscedastic_SGVB_Linear
from src.SGVB.bayesian_heteroscedastic_linear import BayesianRegressor as Heteroscedastic_SGVB_Linear

# import single layer models
from src.SGVB.bayesian_homoscedastic_single import BayesianRegressorHomoscedastic as Homoscedastic_SGVB_Single
from src.SGVB.bayesian_heteroscedastic_single import BayesianRegressor as Heteroscedastic_SGVB_Single

# import intermediate models
from src.SGVB.bayesian_regression_homoscedastic import BayesianRegressorHomoscedastic as Homoscedastic_SGVB_Intermediate
from src.SGVB.bayesian_regression_heteroscedastic import BayesianRegressor as Heteroscedastic_SGVB_Intermediate

# import complex models
from src.SGVB.bayesian_homoscedastic_complex import BayesianRegressorHomoscedastic as Homoscedastic_SGVB_Complex
from src.SGVB.bayesian_heteroscedastic_complex import BayesianRegressor as Heteroscedastic_SGVB_Complex


def nested_dictionary(test_loader, input_dim, hidden_dim, output_dim, layers, M, dropout_rate, path_to_models):
    """
    Creates a nested dictionary of the epistemic uncertainty for a range of model complexities for both SGVB models.
    The outer keys are the complexities, while the inner keys are the different models.

    :param test_loader: torch dataloader object
    :param input_dim: input dimension
    :param hidden_dim: dimension of hidden layers
    :param output_dim: output dimension
    :param M: number of mini-batches/iterations in a single epoch
    :param dropout_rate: dropout rate
    :param path_to_models: path to models (str)
    :return: nested dictionary containing the epistemic uncertainty for different model complexities for the SGVB models
    """
    uncertainty_over_all_complexities = {}
    # Iterate over all model complexities
    for dir in os.listdir(path_to_models): # linear, intermediate, complex
        if dir.startswith("."):
            continue
        path_to_models_with_same_complexity = os.path.join(path_to_models, dir)
        uncertainty_per_complexity = {}
        for model_path in os.listdir(path_to_models_with_same_complexity):
            if model_path.startswith("."):
                continue

            # Initialize models
            if model_path.startswith("sgvb"):
                if "homoscedastic" in model_path.split("_"):
                    model_type = "SGVB Homoscedastic"
                    if dir == "linear":
                        model = Homoscedastic_SGVB_Linear(in_size=input_dim, out_size=output_dim, n_batches=M,
                                                          dropout_rate=dropout_rate)

                    elif dir == "single_layer":
                        model = Homoscedastic_SGVB_Single(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M,
                                                          dropout_rate=dropout_rate)

                    elif dir == "intermediate":
                        model = Homoscedastic_SGVB_Intermediate(in_size=input_dim, hidden_size=hidden_dim,
                                                                out_size=output_dim, n_batches=M,
                                                                dropout_rate=dropout_rate)
                    elif dir == "complex":
                        model = Homoscedastic_SGVB_Complex(layers=layers, n_batches=M, dropout_rate=dropout_rate)
                    else:
                        print("Neither linear, single layer, intermediate nor complex homoscedastic sgvb model found")

                elif "heteroscedastic" in model_path.split("_"):
                    model_type = "SGVB Heteroscedastic"
                    if dir == "linear":
                        model = Heteroscedastic_SGVB_Linear(in_size=input_dim, out_size=output_dim, n_batches=M,
                                                            dropout_rate=dropout_rate)

                    elif dir == "single_layer":
                        model = Heteroscedastic_SGVB_Single(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M,
                                                          dropout_rate=dropout_rate)

                    elif dir == "intermediate":
                        model = Heteroscedastic_SGVB_Intermediate(in_size=input_dim, hidden_size=hidden_dim,
                                                                  out_size=output_dim, n_batches=M,
                                                                  dropout_rate=dropout_rate)
                    elif dir == "complex":
                        model = Heteroscedastic_SGVB_Complex(layers=layers, n_batches=M, dropout_rate=dropout_rate)
                    else:
                        print("Neither linear, single layer, intermediate nor complex heteroscedastic sgvb model found")
                else:
                    print("Neither homoscedastic nor heteroscedastic SGVB model found")

            else:
                print("No SGVB models found")

            path_to_model = os.path.join(path_to_models_with_same_complexity, model_path)

            # Load model and extract epistemic uncertainty
            model.load_state_dict(torch.load(path_to_model))

            model.eval()

            _, var_epistemic, _, _ = model.aleatoric_epistemic_variance(test_loader, B=100)
            std_epistemic = np.sqrt(var_epistemic)
            print("complexity {}, model {}".format(dir, model_type))
            uncertainty_per_complexity[model_type] = np.mean(std_epistemic)

        uncertainty_over_all_complexities[dir] = uncertainty_per_complexity

    return uncertainty_over_all_complexities


def save_dictionary(dictionary, path_to_dictionary):
    """
    Save dictionary to file

    :param dictionary: dict
    :param path_to_dictionary: str
    :return:
    """
    try:
        file = open(path_to_dictionary, 'w')
        file.write(str(dictionary))
        file.close()
    except:
        print("Unable to write to file")
    return


def load_dictionary(path_to_dictionary):
    """
    Load dictionary from file

    :param path_to_dictionary: str
    :return: dict
    """
    file = open(path_to_dictionary, "r")
    content = file.read()
    dictionary = ast.literal_eval(content)
    file.close()
    return dictionary

def plot_nested_dict(dictionary):
    """
    Plot nested dictionary

    :param dictionary: dict
    :return: None
    """
    plt.figure(figsize=(12, 8))

    inner_keys = list(dictionary.values())[0].keys()
    x_axis_values = list(map(str, dictionary.keys()))
    for model in inner_keys:
        y_axis_values = [v[model] for v in dictionary.values()]
        if model == "SGVB Homoscedastic":
            color = "#d62728"
        else:
            color = "#2ca02c"
        plt.plot(x_axis_values, y_axis_values, color, label=model, linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Model complexity", fontsize=20)
    plt.ylabel("Epistemic uncertainty", fontsize=20)
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
    dropout_rate = 0.10
    layers = [input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim,
              hidden_dim, hidden_dim, output_dim]

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    test_loader = dataloader.test_loader()
    path_to_models = "./data/models/regression/varying_complexity"
    path_to_dictionary = "./data/epistemic_uncertainty/varying_complexity_including_single_layer.txt"
    create_dict = False

    if create_dict:
        uncertainty_dict = nested_dictionary(test_loader, input_dim, hidden_dim, output_dim, layers, N, M, dropout_rate, path_to_models)
        save_dictionary(uncertainty_dict, path_to_dictionary)
    else:
        uncertainty_dict = load_dictionary(path_to_dictionary)


    dict_to_plot = {"Linear": uncertainty_dict["linear"],
                    "Single layer": uncertainty_dict["single_layer"],
                    "Intermediate": uncertainty_dict["intermediate"],
                    "Complex": uncertainty_dict["complex"]}

    dict_to_plot = {k : dict(OrderedDict(sorted(v.items()))) for k, v in dict_to_plot.items()}


    plot_nested_dict(dict_to_plot)

    plot_zoomed = False

    if plot_zoomed:
        plt.ylim([0.05, 0.3])
        plt.tight_layout()
        plt.savefig("../../Figures/zoomed_epistemic_uncertainty_varying_complexity.pdf")
    else:
        plt.tight_layout()
        plt.savefig("../../Figures/epistemic_uncertainty_varying_complexity.pdf")
    plt.show()