import os
import ast
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
from src.dataloader.dataloader import Dataloader
from src.utils import *
from src.models.regression_mcdropout_homoscedastic import MCDropoutHomoscedastic
from src.models.regression_mcdropout_heteroscedastic import MCDropoutHeteroscedastic
from src.SGVB.bayesian_regression_homoscedastic import BayesianRegressorHomoscedastic as SGVBHomoscedastic
from src.SGVB.bayesian_regression_heteroscedastic import BayesianRegressor as SGVBHeteroscedastic


def nested_dictionary_performance(test_loader, input_dim, hidden_dim, output_dim, N, M, dropout_rate, path_to_models):
    performance_over_all_training_size = {}
    # Iterate over all training sets
    for dir in tqdm(os.listdir(path_to_models)):
        if dir.startswith("."):
            continue

        path_to_models_with_same_training_sets = os.path.join(path_to_models, dir)
        performance_per_training_size = {}
        # Iterate over models
        for model_path in os.listdir(path_to_models_with_same_training_sets):
            if model_path.startswith("."):
                continue

            # Initialize models
            if model_path.startswith("mcdropout_homoscedastic"):
                model = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N,
                                               M=M, dropout_rate=dropout_rate)
                model_type = "mcdropout_homoscedastic"
            elif model_path.startswith("mcdropout_heteroscedastic"):
                model = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, N=N,
                                                 M=M, dropout_rate=dropout_rate)
                model_type = "mcdropout_heteroscedastic"
            elif model_path.startswith("sgvb_homoscedastic"):
                model = SGVBHomoscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)
                model_type = "sgvb_homoscedastic"
            else:
                model = SGVBHeteroscedastic(in_size=input_dim, hidden_size=hidden_dim, out_size=output_dim, n_batches=M)
                model_type = "sgvb_heteroscedastic"

            path_to_model = os.path.join(path_to_models_with_same_training_sets, model_path)
            print(path_to_model)
            # Load models and extract performance, mean and standard deviation
            model.load_state_dict(torch.load(path_to_model))

            model.eval()
            for m in model.modules():
                if m.__class__.__name__.startswith("Dropout") and model_type.startswith("mcdropout"):
                    m.train()
                else:
                    m.eval()

            mse_tuple, _ = model.evaluate_performance(test_loader)
            mean_mse = mse_tuple[0]
            std_mse = mse_tuple[1]

            performance_per_training_size[model_type] = mean_mse
        performance_over_all_training_size[dir] = performance_per_training_size

    return performance_over_all_training_size

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
    plt.ylabel("MSE", fontsize=20)
    plt.title("Test MSE for varying size of training set", fontsize=24)
    plt.legend(fontsize=16)


if __name__ == "__main__":
    os.chdir("..")

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
    path_to_models = "./data/models/regression/varying_training_set_size/dropout" + str(dropout_rate).replace(".", "")
    path_to_dictionary = "./data/performance/dropout_" + str(dropout_rate).replace(".",
                                                                                             "") + "0_performance_varying_training_set_size.txt"

    create_dict = False

    if create_dict:
        performance_dict = nested_dictionary_performance(test_loader, input_dim, hidden_dim, output_dim, N, M, dropout_rate, path_to_models)
        save_dictionary(performance_dict, path_to_dictionary)
    else:
        performance_dict = load_dictionary(path_to_dictionary)

    dict_to_plot = {"10%": performance_dict['size10'],
                    "20%": performance_dict["size20"],
                    "30%": performance_dict["size30"],
                    "40%": performance_dict["size40"],
                    "50%": performance_dict["size50"],
                    "60%": performance_dict["size60"],
                    "70%": performance_dict["size70"],
                    "80%": performance_dict["size80"],
                    "90%": performance_dict["size90"],
                    "100%": performance_dict["size100"]}

    dict_to_plot = {k: dict(OrderedDict(sorted(v.items()))) for k, v in dict_to_plot.items()}

    plot_nested_dict(dict_to_plot)
    plt.tight_layout()
    plt.savefig("../../Figures/performance_varying_training_set_size.pdf")
    plt.show()