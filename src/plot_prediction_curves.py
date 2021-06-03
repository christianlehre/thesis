import os
import numpy as np
from matplotlib import pyplot as plt


def load_data(path_to_data):
    with np.load(path_to_data) as data:
        predictions = data["predictions"]
        epistemic_ci = data["epistemic_ci"]
        total_ci = data["total_ci"]
        depths = data["depths"]
        y_test = data["y_test"]
        empirical_coverage = data["empirical_coverage"]
        well = data["well"]
    return predictions, epistemic_ci, total_ci, depths, y_test, empirical_coverage, well


def plot_prediction_curve(predictions, y_test, depths, epistemic_ci, total_ci, empirical_coverage, well):
    lower_ci_e = epistemic_ci[0]
    upper_ci_e = epistemic_ci[1]
    lower_ci_t = total_ci[0]
    upper_ci_t = total_ci[1]

    plt.figure(figsize=(8, 12))
    plt.title("Well: {}. Coverage probability {:.2f}%".format(well, 100*empirical_coverage), fontsize=18)
    plt.ylabel("Depth", fontsize=16)
    plt.xlabel("Standardized ACS", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot(y_test, depths, "-", label="True")
    plt.plot(predictions, depths, label="Prediction")
    plt.fill_betweenx(depths, lower_ci_t, upper_ci_t, color="green", alpha=0.2, label="95% CI total")
    plt.fill_betweenx(depths, lower_ci_e, upper_ci_e, color="red", alpha=0.2, label="95% CI epistemic")
    plt.ylim([depths[-1], depths[0]])
    if well == "30/8-5 T2":
        plt.legend(loc="lower right", fontsize=12)
    else:
        plt.legend(loc="best", fontsize=12)


if __name__ == "__main__":
    print(os.getcwd())
    os.chdir("../../..")
    print(os.getcwd())

    path_to_data = "./code/thesis/data/prediction_curves/"
    path_to_figures = "./Figures/"
    MCDropout = True
    if MCDropout:
        path_to_data += "MCDropout/"
        path_to_figures += "MCDropout/"
    else:
        path_to_data += "SGVB/"
        path_to_figures += "SGVB/"

    Homoscedastic = False
    if Homoscedastic:
        path_to_data += "Homoscedastic"
        path_to_figures += "Homoscedastic/"
    else:
        path_to_data += "Heteroscedastic"
        path_to_figures += "Heteroscedastic/"

    path_to_figures += "prediction_curves"
    if not os.path.exists(path_to_figures):
        os.makedirs(path_to_figures)

    zoomed_out = True

    # iterate through all files in the folder, each corresponding to a separate well
    list_of_data = os.listdir(path_to_data)
    for data in list_of_data:
        path_to_well_data = os.path.join(path_to_data, data)
        predictions, epistemic_ci, total_ci, depths, y_test, empirical_coverage, well = load_data(path_to_well_data)
        plot_prediction_curve(predictions, y_test, depths, epistemic_ci, total_ci, empirical_coverage, well)

        well_name = str(well).replace(" ", "_")
        well_name = well_name.replace("/", "")
        fig_name = well_name

        if well == "30/8-5 T2":  # zoomed out
            if zoomed_out:
                plt.xlim([-8, 13])
                fig_name += "_zoomed_out"
            else:
                plt.xlim([-6, 6])
            plt.legend(loc="lower right", fontsize=12)
        elif well == "25/4-10 S":
            plt.xlim([-5, 7])
        elif well == "25/7-6":
            plt.xlim([-4, 4])
            plt.legend(loc="lower right", fontsize=12)
        elif well == "30/6-26":
            plt.xlim([-5, 8])
        elif well == "30/11-10":
            plt.xlim([-7, 7])
        elif well == "30/11-7":
            plt.xlim([-7, 8])
        elif well == "30/11-9 ST2":
            plt.xlim([-4, 8])
        else:  # 30/11-11 S
            plt.xlim([-5, 11])

        fig_name += ".pdf"
        path_to_figure = os.path.join(path_to_figures, fig_name)
        plt.savefig(path_to_figure)
    plt.show()

