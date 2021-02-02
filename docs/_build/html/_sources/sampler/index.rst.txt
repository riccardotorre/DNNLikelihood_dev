.. _sampler_object:

.. module:: sampler

The Sampler object
------------------

The :class:`Sampler <DNNLikelihood.Sampler>` class is an API to the |emcee_link| Python package that can be used to sample
:class:`Lik <DNNLikelihood.Lik>` objects (more precisely the corresponding logpdf function) and export data
as a :mod:`Data <data>` objects. The :class:`Sampler <DNNLikelihood.Sampler>` object offers several methods to
perform MCMC sampling, analyze convergence, and produce different kind of plots. 
The API uses |emcee_link| to perform the sampling and manage the backend, which ensures that samples are safely stored to file 
at any time of the sampling process. See also :ref:`the Likelihood object <likelihood_object>` and 
:ref:`the Data object <data_object>`, which are respectively used to initialize the 
:class:`Sampler <DNNLikelihood.Sampler>` class and to export the :mod:`Data <data>` object.

.. toctree::
   :maxdepth: 8

   usage
   class
   references

.. include:: ../external_links.rst