import os
from scipy.stats import norm
from matplotlib import pyplot as plt

from src.utils import *
from src.models.regression_mcdropout_homoscedastic import MCDropoutHomoscedastic
from src.models.regression_mcdropout_heteroscedastic import MCDropoutHeteroscedastic

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

    training_set = create_torch_dataset(df_train, target_variable, explanatory_variables)

    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    batch_size = 100
    N = len(training_set)
    M = int(N / batch_size)  # number of mini-batches

    # Initialize and load models
    model_heteroscedastic = MCDropoutHeteroscedastic(input_dim=input_dim, hidden_dim=hidden_dim,
                                                     output_dim=output_dim, N=N, M=M)
    model_homoscedastic = MCDropoutHomoscedastic(input_dim=input_dim, hidden_dim=hidden_dim,
                                                 output_dim=output_dim, N=N, M=M)
    heteroscedastic = True
    if heteroscedastic:
        model = model_heteroscedastic
        title = "Heteroscedastic"
        training_configuration = "mcdropout_" + title.lower() + "_"
    else:
        model = model_homoscedastic
        title = "Homoscedastic"
        training_configuration = "mcdropout_" + title.lower() + "_"

    training_configuration += "lr_" + str(model.lr) + "_numepochs_" + str(
        model.num_epochs) + "_hiddenunits_" \
                              + str(hidden_dim) + "_hiddenlayers_2" + "_batch_size_" + str(batch_size)
    training_configuration = training_configuration.replace(".", "")
    path_to_model = "./data/models/regression/"
    path_to_model += training_configuration + ".pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
        else:
            m.eval()

    ####################################################################################################################
    # calculate critical values of the standard distirbution for a range of quantiles
    significance_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    def significance_to_quantiles(alpha):
        return 1 - (1 - alpha) / 2

    qs = [significance_to_quantiles(alpha) for alpha in significance_levels]
    critical_values = norm.ppf(qs)

    # iterate over wells
    wells = list(set(df_test[well_variable]))
    coverages = np.zeros((len(wells), len(critical_values)))
    for i, well in enumerate(wells):
        df_test_single_well = df_test[df_test[well_variable] == well]
        test_set = create_torch_dataset(df_test_single_well, target_variable, explanatory_variables)
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=len(test_set),
                                                  shuffle=False)
        x_test, y_test = unpack_dataset(test_loader)
        mean_predictions, var_epistemic, var_aleatoric, var_total = model.aleatoric_epistemic_variance(
            test_loader, B=100)

        coverage = []
        for z in critical_values:
            lower_ci, upper_ci = credible_interval(mean_predictions, var_total, std_multiplier=z)
            empirical_coverage = coverage_probability(y_test, lower_ci, upper_ci)
            coverage.append(empirical_coverage)
        coverages[i, :] = coverage
        plt.figure()
        plt.title("Coverage for well {}, {} MC Dropout model".format(well, title))
        plt.plot(significance_levels, coverage, "r*", label="Empirical")
        plt.plot(significance_levels, significance_levels, "k--", label="Theoretical")
        plt.ylabel("Significance level")
        plt.xlabel("Coverage probability")
        plt.xticks(significance_levels, rotation=45)
        plt.yticks(significance_levels, rotation=25)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("./../../Figures/MCDropout/{}/well{}.pdf".format(title, well.replace("/", "")))
    mean_coverage = np.mean(coverages, axis=0)
    var_coverage = np.std(coverages, axis=0)**2
    lower_ci, upper_ci = credible_interval(mean_coverage, var_coverage, std_multiplier=1.96)  # --> 95% CI for coverage

    plt.figure(figsize=(12, 8))
    plt.title("Coverage across wells, {} MC Dropout model".format(title), fontsize=24)
    plt.plot(significance_levels, mean_coverage, "r*", label="Average empirical")
    plt.fill_between(significance_levels, lower_ci, upper_ci, color="grey", alpha=0.5, label="95% CI")
    plt.plot(significance_levels, significance_levels, "k--", label="Theoretical")
    plt.ylabel("Significance level", fontsize=20)
    plt.xlabel("Coverage probability", fontsize=20)
    plt.xticks(significance_levels,fontsize=16, rotation=45)
    plt.yticks(significance_levels, fontsize=16, rotation=25)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig("./../../Figures/MCDropout/{}/average_coverage.pdf".format(title))

    plt.show()
