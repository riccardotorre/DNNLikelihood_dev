The Histfactory object
----------------------

Summary
^^^^^^^

The histfactory class is an API to |pyhf_link| that can be used to import likelihoods in the ATLAS histfactory format into
the DNNLikelihood module. The API uses |pyhf_link| to parse all relevant information contained in the histfactory workspace
and to create a ``likelihood`` object (see :class:`The Likelihood object <DNNLikelihood.likelihood.Likelihood>`).

Usage
^^^^^

Class
^^^^^

.. autoclass:: DNNLikelihood.Histfactory
   :undoc-members:

Arguments
"""""""""

    .. _histfactory_workspace_folders:
    workspace_folders
      Path (either relative to the code execution folder or absolute)
      containing the ATLAS histfactory workspace (usually containing the Regions subfolders).

         - **type**: ``str`` or ``None``
         - **default**: ``None``   
    
    .. _histfactory_name:
    name
         Name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object.
         If ``None`` is passed ``name`` is assigned the value 
         ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_histfactory"``, 
         while if a string is passed, the ``"_histfactory"`` suffix is appended 
         (preventing duplication if it is already present).
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``   

    .. _histfactory_regions_folders_base_name:
    regions_folders_base_name
         Common folder name of the Region folders contained in the 
         workspace_folder (these folders are usually named "RegionA", "RegionB", etc.) 
            
            - **type**: ``str``
            - **default**: ``Region``  

    .. _histfactory_bkg_files_base_name:
    bkg_files_base_name
         Name (without .json extension) of the "background" json files 
         in the region folders (e.g. "BkgOnly")
            
            - **type**: ``str``
            - **default**: ``BkgOnly`` 

    .. _histfactory_patch_files_base_name:
    patch_files_base_name
         Base name (without .json extension) of the "signal" patch
         json files in the region folders (e.g. "patch"). 
            
            - **type**: ``str``
            - **default**: ``patch`` 

    .. _histfactory_output_folder:
    output_folder
         Path (either relative to the code execution folder or absolute) where output files are saved.
         If no output folder is specified, ``output_folder`` is set to the code execution folder.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 

    .. _histfactory_histfactory_input_file:
    histfactory_input_file
         File name (either relative to the code execution folder or absolute, with or without any of the
         .json or .pickle extensions) of a saved ``Histfactory`` object. 
         Whenever this parameter is not None all other inputs but :attr:`output_folder` and :attr:`verbose`
         are ignored and the object is reconstructed from the imported files. 
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``

    .. _histfactory_verbosee:
    verbose
         Set verbosity in the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` method. 
         See :ref:`Verbosity mode <verbosity_mode>`.

            - **type**: ``bool``
            - **default**: ``True``

Attributes
""""""""""

    .. py:attribute:: DNNLikelihood.Histfactory.bkg_files_base_name

         Attribute corresponding to the input argument :attr:`bkg_files_base_name`.
         Background files are extracted taking all files in the region subfolders 
         including the string ``bkg_files_base_name``.
            
            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_file

         Absolute path corresponding to the input argument :attr:`histfactory_input_file`
         If the input argument is `None`, the attribute is set to ``None``.
            
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_json_file    

         Absolute path to the .json file containing saved ``Histfactory`` attributes.
         This is automatically generated from the attribute
         :attr:`Histfactory.histfactory_input_file <DNNLikelihood.Histfactory.histfactory_input_file>`.
         When the latter is ``None``, the attribute is set to ``None``.
            
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_log_file    

         Absolute path to the .log file containing saved 
         :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` attribute.
         This is automatically generated from the attribute
         :attr:`Histfactory.histfactory_input_file <DNNLikelihood.Histfactory.histfactory_input_file>`.
         When the latter is ``None``, the attribute is set to ``None``.
            
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_pickle_file    

         Absolute path to the .pickle file containing saved 
         :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>` attribute.
         This is automatically generated from the attribute
         :attr:`Histfactory.histfactory_input_file <DNNLikelihood.Histfactory.histfactory_input_file>`.
         When the latter is ``None``, the attribute is set to ``None``.
            
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_output_json_file

         Absolute path to the .json file where the ``Histfactory`` object is saved.
         This is automatically generated from the attribute
         :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
         See the :meth:`Histfactory.save_histfactory <DNNLikelihood.Histfactory.save_histfactory>`
         method for details about the file contet.
            
            - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_output_pickle_file

         Absolute path to the .pickle file where the ``Histfactory`` object is saved.
         This is automatically generated from the attribute
         :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
         See the :meth:`Histfactory.save_histfactory <DNNLikelihood.Histfactory.save_histfactory>`
         method for details about the file contet.
            
            - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Histfactory.likelihoods_dict

         Main dictionary containing likelihoods parameters and properties for all regions/signal 
         hypotheses. All available likelihoods from the workspace are enumerated so that the dictionary integer keys corresponding
         to each likelihood object: ``_histfactory_additional_attrs = {1: value1, 2: value2, ...}``. 
            
            - **type**: ``dict`` or ``None``
            - **keys**:

               - *"signal_region"* (type: ``str``): name of the signal region to which the member belongs 
               - "bg_only_file"* (type: ``str``): absolute path to the background file corresponding to the signal_region.
               - *"patch_file"* (type: ``str``): absolute path to the patch file for the given likelihood.
               - *"name"* (type: ``str``): name of the given likelihood. It is set to "\ *hf_name*\ _region_\ *rn*\ _patch_\ *pn*\ _\ *lik_n*\ _likelihood"
                 where *hf_name* is the ``Histfactory.name`` attribute, *rn* is the region name determined from the region folder
                 name excluded the string ``regions_folders_base_name``, *pn* is the patch name determined from the patch file name
                 excluded the string ``patch_files_base_name+"."``, and *lik_n* is the likelihood number (the corresponding key
                 in the ``likelihoods_dict``).
               - *"model_loaded"* (type: ``str``): flag that returns ``False`` is the model is not loaded, i.e. only the items *"signal_region"*, *"bg_only_file"*, 
                 *"patch_file"*, *"name"*, and *"model_loaded"* are available in the dictionary, and ``True`` if all dictionary items,
                 i.e. full model information and |pyhf_model_logpdf_link| object, are available in the dictionary.
               - *"model"* (type: |pyhf_model_logpdf_link| object): object containing the given likelihood parameters and logpdf.
                 See the |pyhf_link| documentation.
               - *"obs_data"* (type: ``numpy.ndarray``, shape: ``(n_bins,)``): numpy array containing the number of observed events in each of the n_bins bins for the given signal
                 region.
               - *"pars_init"* (type: ``numpy.ndarray``, shape ``(n_pars,)``): array with a length equal to the number of parameters n_pars
                 entering in the likelihood (logpdf) function and containing their initial values.
               - *"pars_bounds"* (type: ``numpy.ndarray``, shape ``(n_pars,2)``): array with lower and upper limit on each parameter of the n_pars parameters.
                 The logpdf function is constructed such that if any of the parameter has a value outside these bounds, it evaluates
                 to ``-np.inf``.
               - *"pars_labels"* (type: ``list``): list of strings containing the name of each parameter.
               - *"pars_pos_poi"* (type: ``numpy.ndarray``, shape: ``(n_pois)``): array with the list of positions, in the array of parameters, of the n_pois parameters of interest.
               - *"pars_pos_nuis"* (type: ``numpy.ndarray``, shape: ``(n_nuis)``): array with the list of positions, in the array of parameters, of the n_nuis nuisance parameters.

    .. py:attribute:: DNNLikelihood.Histfactory.log    

         Dictionary containing a log of the ``Histfactory`` object calls. The dictionary has datetime strings as keys
         and actions as values. Actions are also dictionaries, containing details of the methods calls.
            
            - **type**: ``dict``
            - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S")``
            - **values**: ``dict``
            - **values keys**:

               **key**: ``"action"``

                  - **possible values**: ``"created"``, ``"loaded"``, ``"imported likelihoods"``, ``"saved"``, ``"created likelihood object"``, and ``"saved likelihood object"``

               **key**: ``"likelihoods numbers"``

                  - **value type**: ``list`` of ``int``

               **key**: ``"likelihood number"``

                  - **value type**: ``int``

               **key**: ``"files"``

                  - **value type**: ``list`` of ``str``

               **key**: ``"file"``

                  - **value type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.name

         Name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object generated from
         the :attr:`name` input argument. It is used to generate 
         output files and is passed to the generated :class:`Likelihood <DNNLikelihood.Likelihood>` objects.
            
            - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Histfactory.output_folder

         Absolute path corresponding to the input argument
         :attr:`output_folder`

            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.patch_files_base_name

         Attribute corresponding to the input argument :attr:`patch_files_base_name`.
         Patch files are extracted taking all files in the region subfolders including 
         the string ``patch_files_base_name``. 
            
            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.regions
         
         Dictionary containing region names (str) as keys 
         and region folders full path (str) as values.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Histfactory.regions_folders_base_name

         Attribute corresponding to the input argument :attr:`regions_folders_base_name`.
         When determining the regions, the code looks at all subfolders of ``workspace_folder`` 
         containing the string ``regions_folders_base_name``, then deletes this latter string to obtain
         the region names and build the :attr:`Histfactory.regions <DNNLikelihood.Histfactory.regions>` 
         dictionary class attribute.   
            
            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.timestamp

         String containing datetime used to rename files. It is initialized with current datetime each
         time the :class:`Histfactory.timestamp <DNNLikelihood.Histfactory.timestamp>` object is created.

    .. py:attribute:: DNNLikelihood.Histfactory.verbose

         Attribute corresponding to the input argument :attr:`verbose`.

    .. py:attribute:: DNNLikelihood.Histfactory.workspace_folders

         Absolute path corresponding to the input argument
         :attr:`workspace_folders`

            - **type**: ``str``

Methods
"""""""

    .. automethod:: DNNLikelihood.Histfactory.__init__

    .. automethod:: DNNLikelihood.Histfactory._Histfactory__check_define_name

    .. automethod:: DNNLikelihood.Histfactory._Histfactory__import_histfactory

    .. automethod:: DNNLikelihood.Histfactory._Histfactory__load_histfactory

    .. automethod:: DNNLikelihood.Histfactory.set_verbosity

    .. automethod:: DNNLikelihood.Histfactory.import_histfactory

    .. automethod:: DNNLikelihood.Histfactory.save_histfactory_log

    .. automethod:: DNNLikelihood.Histfactory.save_histfactory_json

    .. automethod:: DNNLikelihood.Histfactory.save_histfactory_pickle

    .. automethod:: DNNLikelihood.Histfactory.save_histfactory

    .. automethod:: DNNLikelihood.Histfactory.get_likelihood_object