.. _histfactory_object:

The Histfactory object
----------------------

The :class:`Histfactory <DNNLikelihood.Histfactory>` class is an API to the |pyhf_link| Python package that can be used to import 
likelihoods in the ATLAS histfactory format into the DNNLikelihood module. 
The API uses |pyhf_link| (with default |numpy_link| backend) to parse all relevant information contained in the histfactory workspace
and to create a :class:`Lik <DNNLikelihood.Lik>` class object (see :ref:`the Likelihood object <likelihood_object>`).
The :class:`Histfactory <DNNLikelihood.Histfactory>` object is stored in an HDF5 file. A log of the relevant calls to the class methods
is also saved in json format into a log file.

Code examples shown below refer to the following ATLAS histfactory likelihood:

   - |histfactory_sbottom_link|.

.. toctree::
   :maxdepth: 8

   usage
   class

.. include:: ../external_links.rst