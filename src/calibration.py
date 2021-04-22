import os
from scipy.stats import norm
from matplotlib import pyplot as plt

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

    training_set = create_torch_dataset(df_train, target_variable, explanatory_variables)

    input_dim = len(explanatory_variables)
    hidden_dim = 100
    output_dim = 1
    batch_size = 100
    N = len(training_set)
    M = int(N / batch_size)  # number of mini-batches
    dropout_rate = 0.10
    # Choose model to extract calibration curves from
    heteroscedastic = True
    mcdropout = False
    epistemic = True

    if mcdropout:
        training_configuration = "mcdropout_"
        model_type = "MC Dropout"
    else:
        training_configuration = "sgvb_"
        model_type = "SGVB"

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

    #TODO: include dropoutrate in path to mc dropout models
    training_configuration +="dropout_"+str(model.dropout_rate)+"_lr_" + str(model.lr) + "_numepochs_" + str(
        model.num_epochs) + "_hiddenunits_" \
                              + str(hidden_dim) + "_hiddenlayers_2" + "_batch_size_" + str(batch_size)
    training_configuration = training_configuration.replace(".", "")
    path_to_model = "./data/models/regression/"
    path_to_model += training_configuration + ".pt"
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout") and mcdropout:
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
            if epistemic:
                lower_ci, upper_ci = credible_interval(mean_predictions, var_epistemic, std_multiplier=z)
            else:
                lower_ci, upper_ci = credible_interval(mean_predictions, var_total, std_multiplier=z)
            empirical_coverage = coverage_probability(y_test, lower_ci, upper_ci)
            coverage.append(empirical_coverage)
        coverages[i, :] = coverage
        plt.figure()
        if epistemic:
            plt.title("Epistemic Calibration well {}, {} {}".format(well, title, model_type))
        else:
            plt.title("Calibration for well {}, {} {} model".format(well, title, model_type))
        plt.plot(significance_levels, coverage, "r*", label="Empirical")
        plt.plot(significance_levels, significance_levels, "k--", label="Theoretical")
        plt.ylabel("Significance level")
        plt.xlabel("Coverage probability")
        plt.xticks(significance_levels, rotation=45)
        plt.yticks(significance_levels, rotation=25)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if epistemic:
            plt.savefig(
                "./../../Figures/{}/{}/Calibration/Epistemic/well{}.pdf".format(model_type.replace(" ", ""), title,
                                                                      well.replace("/", "")))
        else:
            plt.savefig(
                "./../../Figures/{}/{}/Calibration/well{}.pdf".format(model_type.replace(" ", ""), title, well.replace("/", "")))

    mean_coverage = np.mean(coverages, axis=0)
    var_coverage = np.std(coverages, axis=0) ** 2
    lower_ci, upper_ci = credible_interval(mean_coverage, var_coverage, std_multiplier=1.96)  # --> 95% CI for coverage

    plt.figure(figsize=(12, 8))

    if epistemic:
        plt.title("Epistemic Calibration across wells, {} {}".format(title, model_type), fontsize=24)
    else:
        plt.title("Calibration across wells, {} {} model".format(title, model_type), fontsize=24)

    plt.plot(significance_levels, mean_coverage, "r*", label="Empirical mean")
    plt.fill_between(significance_levels, lower_ci, upper_ci, color="grey", alpha=0.5, label="95% CI")
    plt.plot(significance_levels, significance_levels, "k--", label="Theoretical")
    plt.ylabel("Significance level", fontsize=20)
    plt.xlabel("Coverage probability", fontsize=20)
    plt.xticks(significance_levels, fontsize=16, rotation=45)
    plt.yticks(significance_levels, fontsize=16, rotation=25)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()

    if epistemic:
        plt.savefig("./../../Figures/{}/{}/Calibration/Epistemic/average_coverage.pdf".format(model_type.replace(" ", ""), title))
    else:
        plt.savefig("./../../Figures/{}/{}/Calibration/average_coverage.pdf".format(model_type.replace(" ", ""), title))

    plt.show()
