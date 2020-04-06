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

    .. py:attribute:: DNNLikelihood.Histfactory.workspace_folders

         Path (either relative to the code execution folder or absolute)
         containing the ATLAS histfactory workspace (usually containing the Regions subfolders).
         The ``__init__`` method automatically converts the path into an absolute path.

            - **type**: ``str`` or ``None``
            - **default**: ``None``   

    .. py:attribute:: DNNLikelihood.Histfactory.name  

         Name of the histfactory object. It is used to generate output files and is passed
         to the generated likelihood objects.
         If no ``name`` is specified (default), name is assigned the value ``histfactory_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")``  
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``   

    .. py:attribute:: DNNLikelihood.Histfactory.regions_folders_base_name

         Common folder name of the Region folders contained in the 
         workspace_folder (these folders are usually named 'RegionA', 'RegionB', etc.)
         When determining the regions, the code looks at all subfolders of ``workspace_folder`` 
         containing the string ``regions_folders_base_name``, then deletes this latter string to obtain
         the region names and build the ``regions`` dictionary class attribute (see `Additional attributes`_).   
            
            - **type**: ``str``
            - **default**: ``Region``  

    .. py:attribute:: DNNLikelihood.Histfactory.bkg_files_base_name   

         Name (without .json extension) of the 'background' json files 
         in the region folders (e.g. 'BkgOnly')
         Background files are extracted taking all files in the region subfolders including the string ``bkg_files_base_name``.  
            
            - **type**: ``str``
            - **default**: ``BkgOnly`` 

    .. py:attribute:: DNNLikelihood.Histfactory.patch_files_base_name 

         Base name (without .json extension) of the 'signal' patchx
         json files in the region folders (e.g. 'patch'). Patch files are extracted taking 
         all files in the region subfolders including the string ``patch_files_base_name``. 
            
            - **type**: ``str``
            - **default**: ``patch`` 

    .. py:attribute:: DNNLikelihood.Histfactory.output_folder    

         Path (either relative to the code execution folder or absolute) where output files are saved.
         The __init__ method automatically converts the path into an absolute path.
         If no output folder is specified, ``output_folder`` is set to the code execution folder.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_input_file    

         File name (either relative to the code execution folder or
         absolute) of a saved ``Histfactory`` object.
         Whenever this parameter is not None all other parameters are ignored and the object is
         reconstructed from the imported file. 
         The ``Histfactory`` objects is saved using pickle wit the file containing the .pickle extension. 
         ``histfactory_input_file`` can contain or not the extension. In case it does not, the extension 
         is added by the ``__init__`` method.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 

Additional attributes
"""""""""""""""""""""

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

    .. py:attribute:: DNNLikelihood.Histfactory.histfactory_output_file

         Base name of the output file of the ``Histfactory.save_likelihoods`` method. It is set to 
         ``path.join(Histfactory.output_folder, utils.check_add_suffix(name, "_histfactory")+".pickle")``. 
            
            - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Histfactory.regions
         
         Dictionary containing region names (str) as keys 
         and region folders full path (str) as values.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 

Methods
"""""""

    .. automethod:: DNNLikelihood.Histfactory.__init__

    .. automethod:: DNNLikelihood.Histfactory._Histfactory__import_histfactory

    .. automethod:: DNNLikelihood.Histfactory._Histfactory__load_histfactory

    .. automethod:: DNNLikelihood.Histfactory.import_histfactory

    .. automethod:: DNNLikelihood.Histfactory.save_histfactory

    .. automethod:: DNNLikelihood.Histfactory.get_likelihood_object