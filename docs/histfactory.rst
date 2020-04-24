.. _histfactory_object:

The Histfactory object
----------------------

Summary
^^^^^^^

The :class:`Histfactory <DNNLikelihood.Histfactory>` class is an API to the |pyhf_link| Python package that can be used to import 
likelihoods in the ATLAS histfactory format into the DNNLikelihood module. 
The API uses |pyhf_link| (with default |numpy_link| backend) to parse all relevant information contained in the histfactory workspace
and to create a :class:`Likelihood <DNNLikelihood.Likelihood>` class object (see :ref:`the Likelihood object <likelihood_object>`).

Code examples shown below refer to the following ATLAS histfactory likelihood:

   - |histfactory_sbottom_link|.

.. _histfactory_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Histfactory <DNNLikelihood.Histfactory>` class. Refer to the 
full class documentation for more details.

The first time a :class:`Histfactory <DNNLikelihood.Histfactory>` object is created, the :option:`workspace_folder` argument, 
corresponding to the folder containing the histfactory workspace, needs to be specified. All other input arguments have a default value
that may need to be changed (see :ref:`Arguments documentation <histfactory_arguments>`). Optionally, the user may specify the argument
:option:`output_folder` containing the path (either relative or absolute) to a folder where output files will be saved and the argument
:option:`name` with the name of the object (which is otherwise automatically generated).

A basic initialization code is

.. code-block:: python
    
   import DNNLikelihood

   histfactory = DNNLikelihood.Histfactory(workspace_folder="HEPData_workspaces",
                                        name = "ATLAS_sbottom_search",
                                        output_folder = "<my_output_folder>")

When the object is created, it is automatically saved and three files are created:

   - <my_output_folder>/ATLAS_sbottom_search_histfactory.pickle
   - <my_output_folder>/ATLAS_sbottom_search_histfactory.json 
   - <my_output_folder>/ATLAS_sbottom_search_histfactory.log

See the documentation of the :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>` and of the corresponding methods
with a :meth:`_json <DNNLikelihood.Histfactory.save_json>`,
:meth:`_log <DNNLikelihood.Histfactory.save_log>`,
and :meth:`_pickle <DNNLikelihood.Histfactory.save_pickle>` suffix.

The object can also be initialized importing it from saved files. In this case only the :option:`input_file` argument needs to be specified,
while all other arguments are ignored.
One could also optionally specify a new ``output_folder``. In case this is not specified, the 
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
parameters information and logpdf are not available. Likelihoods can be imported using the
:meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` method. One can choose to import only one likelihood,
only some of them, or all of them, specifying the ``lik_numbers_list`` argument. For instance the first likelihood
can be imported through:

.. code-block:: python

   histfactory.import_likelihoods(lik_numbers_list=[0])

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
        'pars_init': array([1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
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

   histfactory.save()

Finally, from any of the imported likelihoods, one can obtain a :class:`Likelihood <DNNLikelihood.Likelihood>` object through the 
:meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>`, which, by default, aslo saves it. For instance, for our
first likelihood we can do

.. code-block:: python

   likelihood_0 = histfactory.get_likelihood_object(lik_number=0)

For additional information on the :class:`Likelihood <DNNLikelihood.Likelihood>` object see :ref:`the Likelihood object <likelihood_object>` 
documentation.

Class
^^^^^

.. autoclass:: DNNLikelihood.Histfactory
   :undoc-members:

.. _histfactory_arguments:

Arguments
"""""""""

   .. option:: workspace_folder

      Path (either relative to the code execution folder or absolute)
      containing the ATLAS histfactory workspace (containing the "Regions" subfolders).
      It is saved in the :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>` attribute.

         - **type**: ``str`` or ``None``
         - **default**: ``None``   
    
   .. option:: name
   
      Name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object.
      It is used to build the :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` attribute.
         
         - **type**: ``str`` or ``None``
         - **default**: ``None``   

   .. option:: regions_folders_base_name

      Common folder name of the "Region" folders contained in the 
      :option:`workspace_folder` (these folders are usually named "RegionA", "RegionB", etc.).
      It is used to set the 
      :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>` 
      attribute.
            
         - **type**: ``str``
         - **default**: ``Region``  

   .. option:: bkg_files_base_name

      Name (with or without the .json extension) of the "background" json files 
      in the "Region" folders (e.g. "BkgOnly").
      It is used to set the 
      :attr:`Histfactory.bkg_files_base_name <DNNLikelihood.Histfactory.bkg_files_base_name>` 
      attribute.
            
         - **type**: ``str``
         - **default**: ``BkgOnly``

   .. option:: patch_files_base_name
      
      Base name (without the .json extension) of the "signal" patch
      json files in the "Region" folders (e.g. "patch").
      It is used to set the 
      :attr:`Histfactory.patch_files_base_name <DNNLikelihood.Histfactory.patch_files_base_name>` 
      attribute.
            
         - **type**: ``str``
         - **default**: ``patch`` 

   .. option:: output_folder
         
      Path (either relative to the code execution folder or absolute) where output files are saved.
      It is used to set the 
      :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute.
            
         - **type**: ``str`` or ``None``
         - **default**: ``None`` 

   .. option:: input_file
         
      File name (either relative to the code execution folder or absolute, with or without any of the
      .json or .pickle extensions) of a saved :class:`Histfactory <DNNLikelihood.Histfactory>` object. 
      It is used to set the 
      :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>` 
      attribute.
            
         - **type**: ``str`` or ``None``
         - **default**: ``None``

   .. option:: verbose

      Argument used to set the verbosity mode of the 
      :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` 
      method and the default verbosity mode of all class methods that accept a
      ``verbose`` argument.
      See :ref:`Verbosity mode <verbosity_mode>`.

         - **type**: ``bool`` or ``int``
         - **default**: ``True``

Attributess
"""""""""""

   .. py:attribute:: DNNLikelihood.Histfactory.bkg_files_base_name

      Attribute corresponding to the input argument :option:`bkg_files_base_name`.
      Background files are extracted taking all files in the region subfolders 
      including the string ``bkg_files_base_name``.
            
         - **type**: ``str``

   .. py:attribute:: DNNLikelihood.Histfactory.input_file

      Attribute corresponding to the input argument :option:`input_file`.
      Whenever this parameter is not ``None`` the :class:`Histfactory <DNNLikelihood.Histfactory>` object
      is reconstructed from input files (see the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>`
      method for details).
            
         - **type**: ``str`` or ``None``

   .. py:attribute:: DNNLikelihood.Histfactory.input_json_file    

      Absolute path to the .json file containing saved :class:`Histfactory <DNNLikelihood.Histfactory>` json (see
      the :meth:`Histfactory.save_json <DNNLikelihood.Histfactory.save_json>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`.
      When the latter is ``None``, the attribute is set to ``None``.
            
         - **type**: ``str`` or ``None``

   .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_log_file    

      Absolute path to the .log file containing saved :class:`Histfactory <DNNLikelihood.Histfactory>` log (see
      the :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`.
      When the latter is ``None``, the attribute is set to ``None``.
            
         - **type**: ``str`` or ``None``

   .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_pickle_file    

      Absolute path to the .pickle file containing saved :class:`Histfactory <DNNLikelihood.Histfactory>` pickle (see
      the :meth:`Histfactory.save_pickle <DNNLikelihood.Histfactory.save_pickle>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`.
      When the latter is ``None``, the attribute is set to ``None``.
            
         - **type**: ``str`` or ``None``

   .. py:attribute:: DNNLikelihood.Histfactory.histfactory_output_json_file

      Absolute path to the .json file where part of the :class:`Histfactory <DNNLikelihood.Histfactory>` 
      object is saved (see the :meth:`Histfactory.save_json <DNNLikelihood.Histfactory.save_json>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
            
         - **type**: ``str`` 

   .. py:attribute:: DNNLikelihood.Histfactory.histfactory_output_log_file

      Absolute path to the .log file where the :class:`Histfactory <DNNLikelihood.Histfactory>` 
      object log is saved (see the :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
            
         - **type**: ``str`` 

   .. py:attribute:: DNNLikelihood.Histfactory.output_pickle_file

      Absolute path to the .pickle file where part of the :class:`Histfactory <DNNLikelihood.Histfactory>` 
      object is saved (see the :meth:`Histfactory.save_pickle <DNNLikelihood.Histfactory.save_pickle>`
      method for details).
      This is automatically generated from the attribute
      :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
            
         - **type**: ``str`` 

   .. py:attribute:: DNNLikelihood.Histfactory.likelihoods_dict

      Dictionary containing likelihoods parameters and properties for all regions/signal 
      hypotheses. All available likelihoods from the workspace are enumerated. Dictionary keys are integers running from ``0``
      to the number of likelihoods in the workspace (minus one), while values are dictionaries containing likelihood information.
            
         - **type**: ``dict``
         - **keys**: ``int``
         - **values**: ``dict`` with the following structure:

            - *"signal_region"* (value type: ``str``)
               Name of the signal region to which the member belongs 
            - *"bg_only_file"* (value type: ``str``)
               Absolute path to the background file corresponding to the "signal_region".
            - *"patch_file"* (value type: ``str``)
               Absolute path to the patch file for the given likelihood.
            - *"name"* (value type: ``str``)
               Name of the given likelihood. It is set to "\ *hf_name*\ _region_\ *rn*\ _patch_\ *pn*\ _\ *lik_n*\ _likelihood"
               where *hf_name* is the :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` attribute, *rn* is the region name determined from 
               the region folder name excluded the string contained in the 
               :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>` attribute, *pn* is the patch name 
               determined from the patch file name excluded the string contained in the 
               :attr:`Histfactory.patch_files_base_name <DNNLikelihood.Histfactory.patch_files_base_name>` attribute (and a dot), and *lik_n* is 
               the likelihood number (the corresponding key in the :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>` dictionary).
            - *"model_loaded"* (value type: ``str``)
               Flag that returns ``False`` is the model is not loaded, i.e. only the items *"signal_region"*, *"bg_only_file"*, 
               *"patch_file"*, *"name"*, and *"model_loaded"* are available in the dictionary, and ``True`` if all dictionary items,
               i.e. full model information and |pyhf_model_logpdf_link| object, are available in the dictionary.
            - *"model"* (value type: |pyhf_model_logpdf_link| object)
               Object containing the given likelihood parameters and logpdf.
               See the |pyhf_link| documentation.
            - *"obs_data"* (value type: ``numpy.ndarray``, value shape: ``(n_bins,)``)
               Numpy array containing the number of observed events in each of the ``n_bins`` bins for the given signal
               region.
            - *"pars_init"* (value type: ``numpy.ndarray``, value shape ``(n_pars,)``)
               Numpy array with a length equal to the number of parameters ``n_pars``
               entering in the likelihood (logpdf) function and containing their initial values.
            - *"pars_bounds"* (value type: ``numpy.ndarray``, value shape ``(n_pars,2)``)
               Numpy array with lower and upper limit on each of the ``n_pars`` parameters.
            - *"pars_labels"* (value type: ``list``)
               List of strings containing the name of each parameter. Parameters labels are always parsed as "raw" strings (like, for instance,
               ``r"%s"%pars_labels[0]``) and can contain latex expressions that are properly compiled when making plots.
            - *"pars_pos_poi"* (value type: ``numpy.ndarray``, value shape: ``(n_poi)``)
               Numpy array with the list of positions, in the array of parameters, of the ``n_poi`` parameters of interest.
            - *"pars_pos_nuis"* (value type: ``numpy.ndarray``, value shape: ``(n_nuis)``)
               Numpy array with the list of positions, in the array of parameters, of the ``n_nuis`` nuisance parameters.
            
            Available items in the dictionary depend on whether the corresponding likelihood has been imported (``dic["model_loaded"]=True``)
            or not (``dic["model_loaded"]=False``). If ``dic["model_loaded"]=False`` only the items corresponding to the keys
            *"signal_region"*, *"bg_only_file"*, *"patch_file"*, *"name"*, and *"model_loaded"* are availanble.

   .. py:attribute:: DNNLikelihood.Histfactory.log    

      Dictionary containing a log of the :class:`Histfactory <DNNLikelihood.Histfactory>` object calls. The dictionary has datetime 
      strings as keys and actions as values. Actions are also dictionaries, containing details of the methods calls.
            
         - **type**: ``dict``
         - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
         - **values**: ``dict`` with the following structure:

            - *"action"* (value type: ``str``)
               Short description of the action.
               
               **possible values**: ``"created"``, ``"created likelihood object"``, ``"imported histfactory"``, ``"imported likelihoods"``, ``"loaded"``, ``"saved"``, and ``"saved likelihood object"``
            - *"likelihoods numbers"* (value type: ``list`` of ``int``)
               When an operation involving several likelihoods has been performed this value is the list of involved likelihoods .
            - *"likelihood number"* (value type: ``int``)
               When an operation involving a single likelihood has been performed this value is the likelihood number.
            - *"file name"* (value type: ``str``)
               File name of file involved in the action.
            - *"file path"* (value type: ``str``)
               Path of file involved in the action.
            - *"files names"* (value type: ``list`` of ``str``)
               List of file names of files involved in the action.
            - *"files paths"* (value type: ``list`` of ``str``)
               List of paths of files involved in the action.


   .. py:attribute:: DNNLikelihood.Histfactory.name

      Name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object generated from
      the :attr:`name` input argument. If ``None`` is passed, then ``name`` is assigned the value 
      ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_histfactory"``, 
      while if a string is passed, the ``"_histfactory"`` suffix is appended 
      (preventing duplication if it is already present).
      It is used to generate output files names and is passed to the generated 
      :class:`Likelihood <DNNLikelihood.Likelihood>` objects.
            
         - **type**: ``str`` 

   .. py:attribute:: DNNLikelihood.Histfactory.output_folder

      Absolute path corresponding to the input argument
      :option:`output_folder`. If the latter is ``None``, then 
      :attr:`output_folder <DNNLikelihood.Histfactory.output_folder>`
      is set to the code execution folder. If the folder does not exist it is created
      by the :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
      function.

         - **type**: ``str``

   .. py:attribute:: DNNLikelihood.Histfactory.patch_files_base_name

      Attribute corresponding to the input argument :option:`patch_files_base_name`.
      Patch files are extracted taking all files in the region subfolders including 
      the string :attr:`Histfactory.patch_files_base_name <DNNLikelihood.Histfactory.patch_files_base_name>`
      attribute. 
            
         - **type**: ``str``

   .. py:attribute:: DNNLikelihood.Histfactory.regions
         
      Dictionary containing "Region" names (str) as keys 
      and "Region" folders full path (str) as values.
            
         - **type**: ``str`` or ``None``
         - **default**: ``None`` 

   .. py:attribute:: DNNLikelihood.Histfactory.regions_folders_base_name

      Attribute corresponding to the input argument :option:`regions_folders_base_name`.
      When determining the regions, the :meth:`Histfactory.__import <DNNLikelihood.Histfactory._Histfactory__import>` 
      method looks at all subfolders of :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`
      containing the string :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>`, 
      then deletes this latter string (and a dot) to obtain the region names and build the :attr:`Histfactory.regions <DNNLikelihood.Histfactory.regions>` 
      dictionary.
            
         - **type**: ``str``

   .. py:attribute:: DNNLikelihood.Histfactory.verbose

      Attribute corresponding to the input argument :option:`verbose`.
      It represents the verbosity mode of the 
      :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` 
      method and the default verbosity mode of all class methods that accept a
      ``verbose`` argument.
      See :ref:`Verbosity mode <verbosity_mode>`.
   
         - **type**: ``bool`` or ``int``

   .. py:attribute:: DNNLikelihood.Histfactory.workspace_folder

      Absolute path of the ATLAS histfactory workspace folder
      corresponding to the input argument :option:`workspace_folder`.

         - **type**: ``str``

Methods
"""""""

   .. automethod:: DNNLikelihood.Histfactory.__init__

   .. automethod:: DNNLikelihood.Histfactory._Histfactory__check_define_input_files

   .. automethod:: DNNLikelihood.Histfactory._Histfactory__check_define_output_files

   .. automethod:: DNNLikelihood.Histfactory._Histfactory__check_define_name

   .. automethod:: DNNLikelihood.Histfactory._Histfactory__import

   .. automethod:: DNNLikelihood.Histfactory._Histfactory__load

   .. automethod:: DNNLikelihood.Histfactory.import_likelihoods

   .. automethod:: DNNLikelihood.Histfactory.save_log

   .. automethod:: DNNLikelihood.Histfactory.save_json

   .. automethod:: DNNLikelihood.Histfactory.save_pickle

   .. automethod:: DNNLikelihood.Histfactory.save

   .. automethod:: DNNLikelihood.Histfactory.get_likelihood_object

   .. automethod:: DNNLikelihood.Histfactory.set_verbosity

.. |histfactory_sbottom_link| raw:: html
    
    <a href="https://www.hepdata.net/record/ins1748602"  target="_blank"> Search for bottom-squark pair production with the ATLAS detector 
    in final states containing Higgs bosons, b-jets and missing transverse momentum</a>

.. |numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> numpy</a>