Introduction
============

The Likelihood Function is the fundamental ingredient of any statistical inference.
It encodes the full information on experimental measurements and allows for their interpretation from frequentist and Bayesian perspectives.

Current experimental and phenomenological results in fundamental physics and astrophysics typically involve complicated fits with several parameters of interest and hundreds of nuisance parameters.
Unfortunately, it is generically considered a hard task to provide all the information encoded in the Likelihood Function in a practical and reusable way.
Therefore, experimental analyses usually deliver only a small fraction of the full information contained in the Likelihood Function, typically in the form of confidence intervals obtained by profiling on the nuisance parameters, or in terms of probability intervals obtained by marginalising over nuisance parameters.
This way of presenting results is very practical, since it can be encoded graphically into simple plots or tables.

The DNNLikelihood project aims in encoding the full experimental likelihood function in a trained Deep Neural Network, which is able to reproduce the original Likelihood Function as a function of physical and nuisance parameters with the accuracy required to allow for the four aforementioned tasks.

