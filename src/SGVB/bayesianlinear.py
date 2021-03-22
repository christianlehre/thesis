import torch
import torch.nn as nn
from torch.nn import functional as F

class BayesianLinear(nn.Module):
    """
    Mean-field approximation of nn.Linear
    """
    def __init__(self, in_features, out_features, parent, n_batches, prior_mu, prior_sigma, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parent = parent
        self.n_batches = n_batches # for KL re-weighting
        self.include_bias = bias
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        if getattr(parent, "accumulated_kl_div", "None") is None:
            parent.accumulated_kl_div = 0

        # initialize mean-field variational parameters (mu, rho) for weights and biases of the layer
        self.weight_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=0, std=0.001)
        )
        self.weight_rho = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(mean=-2.5, std=0.001)
        )

        if self.include_bias:
            self.bias_mu = nn.Parameter(
                torch.zeros(out_features)
            )
            self.bias_rho = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        # prior distribution of the parameters
        self.prior_weights = torch.distributions.normal.Normal(self.prior_mu, self.prior_sigma)
        self.prior_bias = torch.distributions.normal.Normal(self.prior_mu, self.prior_sigma)

        self.log_prior_w = self.prior_weights.log_prob
        self.log_prior_b = self.prior_weights.log_prob

    def reparametrize(self, mu, rho):  # sample weights/biases of the linear layer
        sigma = torch.log1p(torch.exp(rho))
        eps = torch.randn_like(sigma)
        w = mu + torch.mul(sigma, eps)
        return w

    def kl_divergence(self, z, mu_theta, rho_theta, log_prior):  # KL div between variational dist and prior
        log_prior = log_prior(z)
        log_prob_q = torch.distributions.normal.Normal(mu_theta, torch.log1p(torch.exp(rho_theta))).log_prob(z)
        return (log_prob_q - log_prior).sum() / self.n_batches

    def kl_div_q_prior(self, mu_theta, rho_theta):  # as derived analytically.
        sigma_theta = torch.log1p(torch.exp(rho_theta))

        term1 = 2*torch.log(torch.div(self.prior_sigma, sigma_theta)) - 1
        term2 = torch.pow(torch.div(sigma_theta, self.prior_sigma), 2)
        term3 = torch.pow(torch.div(mu_theta - self.prior_mu, self.prior_sigma), 2)
        res = 0.5*(term1 + term2 + term3).sum() / self.n_batches
        return res

    def forward(self, x):  # sample from the distribution of parameters every forward pass
        w = self.reparametrize(self.weight_mu, self.weight_rho)
        if self.include_bias:
            b = self.reparametrize(self.bias_mu, self.bias_rho)
        else:
            b = 0

        self.parent.accumulated_kl_div += self.kl_div_q_prior(self.weight_mu, self.weight_rho)
        if self.include_bias:
            self.parent.accumulated_kl_div += self.kl_div_q_prior(self.bias_mu, self.bias_rho)
        z = x @ w + b

        return z
