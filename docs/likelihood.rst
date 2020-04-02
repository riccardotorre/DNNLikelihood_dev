likelihood.py
-------------

Summary
^^^^^^^

The likelihood class acts as a container for the original likelihood function. It contains information on parameters,
parameters initializations, information on likelihood maxima and the logpdf function. In case in which the likelihood
is obtained using the histfactory interface, the logpdf is constructed from the pyhf.Workspace.model.logpdf method.

Usage
^^^^^

Class
^^^^^

The file histfactoy.py contains a single class.

.. autoclass:: source.likelihood.Likelihood
   :undoc-members:

Arguments
"""""""""


Additional attributes
"""""""""""""""""""""


Methods
"""""""