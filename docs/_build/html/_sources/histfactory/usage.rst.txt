.. _histfactory_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Histfactory <DNNLikelihood.Histfactory>` class. 
Refer to the full class documentation for more details.

The first time a :class:`Histfactory <DNNLikelihood.Histfactory>` object is created, the 
:argument:`workspace_folder <Histfactory.workspace_folder>` argument, 
corresponding to the folder containing the histfactory workspace, needs to be specified. All other input 
arguments have a default value that may need to be changed (see 
:ref:`Arguments documentation <histfactory_arguments>`). Optionally, the user may specify the argument
:argument:`output_folder <Histfactory.output_folder>` containing the path (either relative to the code 
execution path or absolute) to a folder where output files will be saved and the argument 
:argument:`name <Histfactory.name>` with the name of the object (which is otherwise automatically 
generated).

A basic initialization code is

.. code-block:: python
    
    import DNNLikelihood

    histfactory = DNNLikelihood.Histfactory(workspace_folder=<path_to_HEPData_workspaces>",
                                            name = "ATLAS_sbottom_search",
                                            output_folder = <path_to_output_folder>",
                                            verbose = 2)
    
    >>> ============================== 
        Initialize Histfactory object.
        
        ============================== 
        No Histfactory input files and folders specified.
        
        ============================== 
        HistFactory output folder set to
        	 <abs_path_to_output_folder> .
        
        ============================== 
        Histfactory Workspace folder
        	 <abs_path_to_HEPData_workspaces> 
        copied into the folder
        	 <abs_path_to_output_folder>/histfactory_workspace .
        
        ============================== 
        Successfully imported 649 likelihoods from 3 regions.
        
        ============================== 
        Histfactory json file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.json 
        saved in 0.03017405600985512 s.
        
        ============================== 
        Histfactory h5 file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.h5 
        saved in 0.03426715300884098 s.
        
        ============================== 
        Histfactory log file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log 
        saved in 0.0014114639780018479 s.

where we used the placeholders <path_to_HEPData_workspaces>, <abs_path_to_HEPData_workspaces>, 
<path_to_output_folder>, and <abs_path_to_output_folder> to indicate the corresponding paths 
(relative paths can be given as input and are automatically converted into absolute paths).

When the object is created, it is automatically saved and three files are created:

    - <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.h5
    - <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.json 
    - <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log

See the documentation of the 

    - :meth:`Histfactory.save<DNNLikelihood.Histfactory.save>`
    - :meth:`Histfactory.save_h5<DNNLikelihood.Histfactory.save_h5>` 
    - :meth:`Histfactory.save_json<DNNLikelihood.Histfactory.save_json>` 
    - :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>` 

methods for more details.

Moreover, when the object is created, the ``workspace_folder`` given as input is copied 
(if it is not already there) into the output folder "<abs_path_to_output_folder>" and
renamed "histfactory_workspace", and the corresponding path 
``<abs_path_to_output_folder>/histfactory_workspace`` is saved in the 
:attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>` attribute.

The object can also be initialized importing it from saved files. In this case only the 
:argument:`input_file <Histfactory.input_file>`
argument needs to be specified (with path relative to the code execution folder
or absolute and with or without extension), while all other arguments are ignored.
One could also optionally specify a new :argument:`output_folder <Histfactory.output_folder>`. 
In case this is not specified, the 
:attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute is set equal to
the :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>` one, so that the 
object continues to be saved in the same path from which it has been imported.
For instance we could import the object created above with

.. code-block:: python

    histfactory_loaded = DNNLikelihood.Histfactory(input_file="<path_to_output_folder>/ATLAS_sbottom_search_histfactory")

    >>> ============================== 
        Initialize Histfactory object.
        
        ============================== 
        Histfactory input folder set to
            <abs_path_to_output_folder> .
        
        ============================== 
        Histfactory object loaded in 0.016981199999918317 .
        
        ============================== 
        HistFactory output folder set to
            <abs_path_to_output_folder> .
        
        ============================== 
        Histfactory Workspace folder
            <abs_path_to_output_folder>/histfactory_workspace 
        already present in the output folder.
        
        ============================== 
        Histfactory log file
            <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log 
       updated (or saved if it did not exist) in 0.0009314999999787688 s.

The object we created above has the attribute 
:attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`, 
whichs contains a dictionary with items corresponsing to likelihoods and their properties. 
For instance, for the first likelihood, the dictionary looks like this

.. code-block:: python
    
    histfactory.likelihoods_dict[0]

    >>> {'signal_region': 'A',
         'bg_only_file': 'BkgOnly.json',
         'patch_file': 'patch.sbottom_1000_131_1.json',
         'name': 'ATLAS_sbottom_search_histfactory_0_region_A_patch_sbottom_1000_131_1_likelihood',
         'model_loaded': False}

The ``'model_loaded': False`` flag indicates that the corresponding likelihood has not yet been 
fully imported, i.e. that the corresponding parameters information and logpdf are not available. 
Likelihoods can be imported using the 
:meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` method. 
One can choose to import only one likelihood, only some of them, or all of them, specifying the 
``lik_list`` argument. For instance the first likelihood can be imported through:

.. code-block:: python

    histfactory.import_likelihoods(lik_list=[0],verbose=1)

    >>> ============================== 
        Loading
            <abs_path_to_output_folder>/histfactory_workspace/RegionA/patch.sbottom_1000_131_1.json 
        patch file.
        
        ============================== 
        File
            <abs_path_to_output_folder>/histfactory_workspace/RegionA/patch.sbottom_1000_131_1.json 
        processed in 1.2005613000000004 s.
        
        ============================== 
        Histfactory log file
            <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log 
        updated (or saved if it did not exist) in 0.0015189000000006558 s.

        ============================== 
        Imported 1 likelihoods in  1.2017501000000017 s.

When importing more than one likelihood, the option ``progressbar``, which by default is ``True``, 
allows to monitor the import with a progress bar. The imported likelihood are stored in the dictionary 
so that the corresponding item now contains full likelihood information, and the ``model_loaded`` flag 
is set to ``True``. When the object is imported, the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` 
attribute is updated, as well as the corresponding file 
<abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log.

Looking at the dictionary for the fist likelihood, now one gets

.. code-block:: python

    histfactory.likelihoods_dict[0]

    >>> {'signal_region': 'A',
         'bg_only_file': 'BkgOnly.json',
         'patch_file': 'patch.sbottom_1000_131_1.json',
         'name': 'ATLAS_sbottom_search_histfactory_0_region_A_patch_sbottom_1000_131_1_likelihood',
         'model_loaded': True,
         'model': <pyhf.pdf.Model at 0x1dac0074ee0>,
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

Once likelihoods are imported the object can be saved using the 
:meth:`Histfactory.save<DNNLikelihood.Histfactory.save>`
method. This saves the whole object, unless the optional argument ``lik_list`` is specified. 
If it is specified, then only the listed likelihoods are saved in ``'model_loaded' = True`` mode 
(if they have been previously imported), so with full likelihood information, while all other likelihoods 
are saved in ``'model_loaded' = False`` mode, that means without full likelihood information. 
This may allow to save disk space. If only some likelihoods have been imported and no ``lik_list`` 
argument is specified, then the object is saved in its current state. For instance, we can save our object 
(which only contains the first likelihood in ``'model_loaded' = True`` mode), simply by writing

.. code-block:: python

    histfactory.save(overwrite=True)

    >>> ============================== 
        Histfactory json file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.json
        updated (or saved if it did not exist) in 0.049732799999901545 s.
        
        ============================== 
        Histfactory h5 file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.h5 
        updated (or saved if it did not exist) in 0.024474800000007235 s.
        
        ============================== 
        Histfactory log file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log 
        updated (or saved if it did not exist) in 0.0014777999999751046 s.

The ``overwrite=True`` argument ensures that the output files (generated when initializing the object) 
are updated. If ``overwrite=False``, then the old files are renamed and new files are saved.

Finally, from any of the imported likelihoods, one can obtain a :class:`Lik <DNNLikelihood.Lik>` object 
through the :meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>`
method. When created, the :class:`Lik <DNNLikelihood.Lik>` object is automatically saved 
(see the documentation of the :meth:`Lik.save <DNNLikelihood.Lik.save>` method). For instance, for our
first likelihood we can do

.. code-block:: python

    likelihood_0 = histfahistfactoryct.get_likelihood_object(lik_number=0, output_folder="ATLAS_sbottom_search/likelihood_0",verbose=2)
    
    >>> ============================== 
        Creating 'Lik' object
        
        ============================== 
        Initialize Likelihood object.
        
        ============================== 
        Lik output folder set to
        	 <abs_path_to_likelihood_output_folder> .
        
        ============================== 
        Predictions json file
        	 <abs_path_to_likelihood_output_folder>/ATLAS_sbottom_search_0_region_A_patch_sbottom_1000_131_1_likelihood_predictions.json 
        saved in 0.001089200000002677 s.
        
        ============================== 
        Likelihood log file
        	 <abs_path_to_likelihood_output_folder>/ATLAS_sbottom_search_0_region_A_patch_sbottom_1000_131_1_likelihood.log 
        saved in 0.0015916000000046893 s.
        
        ============================== 
        Likelihood h5 file
        	 <abs_path_to_likelihood_output_folder>/ATLAS_sbottom_search_0_region_A_patch_sbottom_1000_131_1_likelihood.h5 
        saved in 0.0571498000000048 s.
        
        ============================== 
        Lik object for likelihood 0 created and saved in 0.0637558000000027 s.
        
        ============================== 
        Histfactory log file
        	 <abs_path_to_output_folder>/ATLAS_sbottom_search_histfactory.log 
        updated (or saved if it did not exist) in 0.0008726000000010004 s.

where the placeholder <abs_path_to_likelihood_output_folder> represent the absolute paths to the likelihood 
object output folder.

For additional information see the :mod:`Likelihood <likelihood>` object documentation.

.. include:: ../external_links.rst