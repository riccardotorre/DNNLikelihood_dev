.. _histfactory_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Histfactory <DNNLikelihood.Histfactory>` class. Refer to the 
full class documentation for more details.

The first time a :class:`Histfactory <DNNLikelihood.Histfactory>` object is created, the 
:argument:`workspace_folder <Histfactory.workspace_folder>` argument, 
corresponding to the folder containing the histfactory workspace, needs to be specified. All other input arguments have a default value
that may need to be changed (see :ref:`Arguments documentation <histfactory_arguments>`). Optionally, the user may specify the argument
:argument:`output_folder <Histfactory.output_folder>` containing the path (either relative or absolute) to a folder where output files will be saved and the argument
:argument:`name <Histfactory.name>` with the name of the object (which is otherwise automatically generated).

A basic initialization code is

.. code-block:: python
    
   import DNNLikelihood

   histfactory = DNNLikelihood.Histfactory(workspace_folder="HEPData_workspaces",
                                           name = "ATLAS_sbottom_search",
                                           output_folder = "<my_output_folder>")

When the object is created, it is automatically saved and two files are created:

   - <my_output_folder>/ATLAS_sbottom_search_histfactory.h5 
   - <my_output_folder>/ATLAS_sbottom_search_histfactory.log

See the documentation of the :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>` and 
:meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>` methods.

The object can also be initialized importing it from saved files. In this case only the :argument:`input_file <Histfactory.input_file>` 
argument needs to be specified, while all other arguments are ignored.
One could also optionally specify a new :argument:`output_folder <Histfactory.output_folder>`. In case this is not specified, the 
:attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute from the imported object is used.
For instance we could import the object created above with

.. code-block:: python
    
   import DNNLikelihood

   histfactory = DNNLikelihood.Histfactory(input_file="<my_output_folder>/ATLAS_sbottom_search_histfactory")

The object we created above has the attribute :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`, whichs contains
a dictionary with items corresponsing to likelihoods and their properties. For instance, for the first
likelihood, the dictionary looks like this

.. code-block:: python
    
   histfactory.likelihoods_dict[0]

   >>> {'signal_region': 'A',
        'bg_only_file': '<full path to HEPData_workspaces>/RegionA/BkgOnly.json',
        'patch_file': '<full path to HEPData_workspaces>/RegionA/patch.sbottom_1000_131_1.json',
        'name': 'ATLAS_sbottom_search_histfactory_0_region_A_patch_sbottom_1000_131_1_likelihood',
        'model_loaded': False}

The ``'model_loaded': False`` flag indicates that the corresponding likelihood has not yet been fully imported, i.e. that the corresponding
parameters information and logpdf are not available. Liks can be imported using the
:meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` method. One can choose to import only one likelihood,
only some of them, or all of them, specifying the ``lik_numbers_list`` argument. For instance the first likelihood
can be imported through:

.. code-block:: python

   histfactory.import_likelihoods(lik_numbers_list=[0],verbose=True)

When importing more than one likelihood, the option ``progressbar``, which by default is ``True``, allows to monitor the import with
a progress bar. The imported likelihood are stored in the dictionary so that the corresponding item now contains full likelihood information, 
and the ``model_loaded`` flag is set to ``True``. When the object is imported, the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` 
attribute is updated, as well as the corresponding file <my_output_folder>/ATLAS_sbottom_search_histfactory.log.

Looking at the dictionary for the fist likelihood, now one gets

.. code-block:: python

   histfactory.likelihoods_dict[0]

   >>> {'signal_region': 'A',
        'bg_only_file': '<full path to HEPData_workspaces>/RegionA/BkgOnly.json',
        'patch_file': '<full path to HEPData_workspaces>/RegionA/patch.sbottom_1000_131_1.json',
        'name': 'ATLAS_sbottom_search_histfactory_0_region_A_patch_sbottom_1000_131_1_likelihood',
        'model_loaded': True,
        'model': <pyhf.pdf.Model at 0x22c40b29d48>,
        'obs_data': array([153.,  52.,  19.,  12.,   3.,   2.,   0.,   0.,   0.,   0.,   0.,
                 ...]),
        'pars_central': array([1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               ...]),
        'pars_bounds': array([[ 9.150e-01,  1.085e+00],
               ...]),
        'pars_labels': ['lumi_0',
         ...],
        'pars_pos_poi': array([5]),
        'pars_pos_nuis': array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
               ...])}

Once likelihoods are imported the object can be saved using the :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>`
method. This saves the whole object, unless the optional argument ``lik_numbers_list`` is specified. It it is specified, then only the listed likelihoods
are saved in ``'model_loaded': True`` mode (if they have been previously imported), so with full likelihood information, while all other likelihoods 
are saved in ``'model_loaded': False`` mode, that means without full likelihood information. This may allow to save disk space. If only some likelihoods 
have been imported and no ``lik_numbers_list`` argument is specified in the 
:meth:`Histfactory.save <DNNLikelihood.Histfactory.save>` method, then the object is saved in its current state. For instance,
we can save our object (which only contains the first likelihood in ``'model_loaded': True`` mode), simply by writing

.. code-block:: python

   histfactory.save(overwrite=True)

The ``overwrite=True`` ensure that the output files (generated when initializing the object) are updated.

Finally, from any of the imported likelihoods, one can obtain a :class:`Lik <DNNLikelihood.Lik>` object through the 
:meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>`. When created, the 
:class:`Lik <DNNLikelihood.Lik>` object is automatically saved (see the documentation of the
:meth:`Lik.save <DNNLikelihood.Lik.save>` method). For instance, for our
first likelihood we can do

.. code-block:: python

   likelihood_0 = histfactory.get_likelihood_object(lik_number=0)

For additional information see the :mod:`Likelihood <likelihood>` object documentation.

.. include:: ../external_links.rst