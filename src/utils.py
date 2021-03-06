import torch
import numpy as np

def create_torch_dataset(df, target, predictors):
    """
    Create a torch dataset based on a dataframe, target variable and predictors

    :param df: pandas dataframe containing the dataset
    :param target: target variable (string)
    :param predictors: explanatory variables/predictors (list of strings)
    :return: torch dataset
    """
    y = df[target]
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    x = df[predictors]
    x = torch.tensor(x.values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


def unpack_dataset(dataloader_object):
    """
    Unpacks a dataset from a dataloader object

    :param dataloader_object: torch dataloader object
    :return: dataset as a tuple (X, y)
    """
    for x, y in dataloader_object:
        x, y = x, y
    return x, y


def credible_interval(mean, variance, std_multiplier):
    """
    Calcualtes a credible interval around the mean for a given significance level

    :param mean: (list) mean values
    :param variance: (list) variances
    :param std_multiplier: (float) specifying the confidence level
            as a critical value of the standard normal distribution, e.g. 1.960 --> 95% C.I
    :return: lower and upper credible interval as a tuple (lower, upper)
    """
    upper_ci = [m + std_multiplier*np.sqrt(v) for m, v in zip(mean, variance)]
    lower_ci = [m - std_multiplier*np.sqrt(v) for m, v in zip(mean, variance) ]

    return lower_ci, upper_ci


def coverage_probability(y_test, lower_ci, upper_ci):
    """
    Calculates the coverage probability of a credible interval, i.e. the fraction of samples that
    falls inside the credible interval

    :param test_loader: torch dataloader object representing the test data
    :param lower_ci: (list) lower bound of the confidence interval for all samples in the test set
    :param upper_ci: (list) upper cound of the credible interval for all samples in the test set
    :return: (float) coverage probability, in (0,1)
    """
    num_samples = len(y_test)
    num_samples_inside_ci = 0

    for i in range(num_samples):
        if upper_ci[i] > y_test[i] > lower_ci[i]:
            num_samples_inside_ci += 1
    coverage = num_samples_inside_ci / num_samples

    return coverage


def set_size(width, prediction_curves, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if prediction_curves:
        fig_height_in = fig_width_in*1.5
    else:
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
