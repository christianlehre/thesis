# Introduction 
This repo contains the source code for implementing and performing the analysis presented in my master's thesis 
in Industrial Mathematics at the Norwegian University of Science and Technology (NTNU). 

The thesis aims to construct Bayesian neural networks (BNNs) that are able to quantify their predictive uncertainty, and
furthermore to decompose the predictive uncertainty into the epistemic and aleatoric components.
Two methods for constructing BNNs are implemented and extended, namely MC Dropout and SGVB. The methods are extended 
from their original formulation by allowing the aleatoric uncertainty to be estimated from data, rather than being set as an hyper-parameter.
Furthermore, the provided uncertainty estimates are properly evaluated by using an anlogy between Bayesian credible 
intervals and the frequentist coverage probability. Additionally, an analysis of the epistemic uncertainty is carried 
out in terms of the training set size and the model complexity