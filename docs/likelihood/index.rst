.. _likelihood_object:

.. module:: likelihood

Lik object
-----------------

The :class:`Lik <DNNLikelihood.Lik>` class acts as a container for the likelihood function. 
It contains information on parameters initializations, positions, bounds, and labels, the logpdf function and its arguments, and methods that allow 
one to plot the logpdf funcion and to compute its (profiled and global) maximiza. 
In case in which the likelihood is obtained using the interface to the ATLAS histfactory workspaces given by the
:obj:`Histfactory <histfactory>` object, the logpdf is constructed from the |pyhf_model_logpdf_link| method.
The :class:`Lik <DNNLikelihood.Lik>` object is stored in an HDF5 file. A log of the relevant calls to the class methods
is also saved in json format into a log file.

.. toctree::
    :maxdepth: -1
    
    usage
    class

.. include:: ../external_links.rst