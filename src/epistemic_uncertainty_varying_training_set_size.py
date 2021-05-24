import os
import ast
from collections import OrderedDict
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *
from src.models.regression_mcdropout_homoscedastic import MCDropoutHomoscedastic
from src.models.regression_mcdropout_heteroscedastic import MCDropoutHeteroscedastic
from src.SGVB.bayesian_regression_homoscedastic import BayesianRegressorHomoscedastic as SGVBHomoscedastic
from src.SGVB.bayesian_regression_heteroscedastic import BayesianRegressor as SGVBHeteroscedastic
from tqdm import tqdm

def nested_dictionary(test_loader,input_dim, hidden_dim, output_dim, N, M, dropout_rate, path_to_models):
    uncertainty_over_all_training_size = {}
    # Iterate over training set size
    for dir in tqdm(os.listdir(path_to_models)):
        if dir.startswith("."):
            continue

        path_to_models_with_same_training_sets = os.path.join(path_to_models, dir)
        uncertainty_per_training_size = {}
        # Iterate over models
        for model_path in os.listdir(path_to_models_with_same_training_sets):
            if model_path.startswith("."):
                continue

            # Initialize model
            if model_path.startswith("mcdropout_homoscedastic"):
                model = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N,
                                               M=M, dropout_rate=dropout_rate)
                model_type = "MC Dropout Homoscedastic"
            elif model_path.startswith("mcdropout_heteroscedastic"):
                model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N,
                                                 M=M, dropout_rate=dropout_rate)
                model_type = "MC Dropout Heteroscedastic"
            elif model_path.startswith("sgvb_homoscedastic"):
                model = SGVBHomoscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M,
                                          dropout_rate=dropout_rate)
                model_type = "SGVB Homoscedastic"
            else:
                model = SGVBHeteroscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M,
                                            dropout_rate=dropout_rate)
                model_type = "SGVB Heteroscedastic"

            path_to_model = os.path.join(path_to_models_with_same_training_sets, model_path)

            # Load models and extract epistemic uncertainty
            model.load_state_dict(torch.load(path_to_model))

            model.eval()
            for m in model.modules():
                if m.__class__.__name__.startswith("Dropout") and model_type.startswith("MC"):
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

        plt.plot(x_axis_values, y_axis_values, label=model, linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Fraction of training set", fontsize=20)
    plt.ylabel("Epistemic uncertainty", fontsize=20)
    plt.ylim([0, 0.6])
    plt.title("Epistemic uncertainty for varying size of training set", fontsize=24)
    plt.legend(fontsize=16)

def plot_nested_dicts_same_figure(uncertainty_dictionary, performance_dictionary):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title("Epistemic uncertainty and MSE for varying size of training set", fontsize=24)
    ax1.set_ylabel("Epistemic uncertainty", fontsize=20)
    ax1.set_xlabel("Fraction of training set", fontsize=20)
    ax2 = ax1.twinx()
    ax2.set_ylabel("test MSE", fontsize=20)

    inner_keys = list(uncertainty_dictionary.values())[0].keys()
    x_axis_values = list(map(str, uncertainty_dictionary.keys()))
    for model in inner_keys:
        y_values_uncertainty = [v[model] for v in uncertainty_dictionary.values()]
        y_values_performance = [v[model] for v in performance_dictionary.values()]

        ax1.plot(x_axis_values, y_values_uncertainty, "-", label=model, linewidth=3)
        ax2.plot(x_axis_values, y_values_performance, "--", linewidth=3)
    plt.xticks(fontsize=16)
    ax1.set_ylim([0, 0.6])
    ax1.legend(fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)
    plt.setp(ax2.get_yticklabels(), fontsize=16)

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

    dataloader = Dataloader(training_set=training_set, validation_set=validation_set,
                            test_set=test_set, batch_size=batch_size)

    test_loader = dataloader.test_loader()
    path_to_models = "./data/models/regression/varying_training_set_size/dropout"+str(dropout_rate).replace(".", "")
    path_to_uncertainty_dictionary = "./data/epistemic_uncertainty/dropout_"+str(dropout_rate).replace(".","")+"0_epistemic_uncertainty_varying_training_set_size.txt"
    path_to_performance_dictionary = "./data/performance/dropout_" + str(dropout_rate).replace(".",
                                                                                   "") + "0_performance_varying_training_set_size.txt"

    create_dict = False

    if create_dict:
        uncertainty_dict = nested_dictionary(test_loader, input_dim, hidden_dim, output_dim, N, M, dropout_rate, path_to_models)
        save_dictionary(uncertainty_dict, path_to_uncertainty_dictionary)
    else:
        uncertainty_dict = load_dictionary(path_to_uncertainty_dictionary)
        performance_dict = load_dictionary(path_to_performance_dictionary)

    # sort dict in ascending fraction of full training set
    uncertainty_dict_to_plot = {"10%": uncertainty_dict['size10'], "20%": uncertainty_dict["size20"],
                                "30%": uncertainty_dict["size30"], "40%": uncertainty_dict["size40"],
                                "50%": uncertainty_dict["size50"], "60%": uncertainty_dict["size60"],
                                "70%": uncertainty_dict["size70"], "80%": uncertainty_dict["size80"],
                                "90%": uncertainty_dict["size90"], "100%": uncertainty_dict["size100"]}

    performance_dict_to_plot = {"10%": performance_dict['size10'], "20%": performance_dict["size20"],
                                "30%": performance_dict["size30"], "40%": performance_dict["size40"],
                                "50%": performance_dict["size50"], "60%": performance_dict["size60"],
                                "70%": performance_dict["size70"], "80%": performance_dict["size80"],
                                "90%": performance_dict["size90"], "100%": performance_dict["size100"]}

    uncertainty_dict_to_plot = {k: dict(OrderedDict(sorted(v.items()))) for k, v in uncertainty_dict_to_plot.items()}
    performance_dict_to_plot = {k: dict(OrderedDict(sorted(v.items()))) for k, v in performance_dict_to_plot.items()}

    plot_uncertainty_and_performance = True

    if plot_uncertainty_and_performance:
        plot_nested_dicts_same_figure(uncertainty_dict_to_plot, performance_dict_to_plot)
        plt.tight_layout()
        plt.savefig("../../Figures/epistemic_uncertainty_test_MSE_varying_training_set_size.pdf")
    else:
        plot_nested_dict(dict_to_plot)
        plt.tight_layout()
        plt.savefig("../../Figures/dropout_"+str(dropout_rate).replace(".", "")+"0_epistemic_uncertainty_varying_training_set_size.pdf")
    plt.show()
