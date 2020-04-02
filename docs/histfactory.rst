histfactory.py
--------------

Summary
^^^^^^^

The histfactory class is an API to pyhf that can be used to import likelihoods in the ATLAS histfactory format into
the DNNLikelihood module. The API uses pyhf to parse all relevant information contained in the histfactory workspace
and to create a ``likelihood`` object (see :ref:`likelihood_class`).

Usage
^^^^^

Class
^^^^^

The file histfactoy.py contains a single class.

.. autoclass:: source.histfactory.Histfactory
   :undoc-members:

Arguments
"""""""""

    .. py:attribute:: workspace_folders

        Path (either relative to the code execution folder or absolute)
        containing the ATLAS histfactory workspace (usually containing the Regions subfolders).
        The ``__init__`` method automatically converts the path into an absolute path.  

        :type: ``str`` or ``None``
        :return: ``None``   

    .. py:attribute:: name  

        Name of the histfactory object. It is used to generate output files and is passed
        to the generated likelihood objects.
        If no ``name`` is specified (default), name is assigned the value ``histfactory_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")``  

        :type: ``str`` or ``None``
        :return: ``None``   

    .. py:attribute:: regions_folders_base_name

        Common folder name of the Region folders contained in the 
        workspace_folder (these folders are usually named 'RegionA', 'RegionB', etc.)
        When determining the regions, the code looks at all subfolders of ``workspace_folder`` 
        containing the string ``regions_folders_base_name``, then deletes this latter string to obtain
        the region names and build the ``regions`` dictionary class attribute (see :ref:`Additional attributes <_histfactory_additional_attrs>`).   

        :type: ``str``
        :return: ``"Region"``   

    .. py:attribute:: bkg_files_base_name   

        Name (without .json extension) of the 'background' json files 
        in the region folders (e.g. 'BkgOnly')
        Background files are extracted taking all files in the region subfolders including the string ``bkg_files_base_name``.  

        :type: ``str``
        :return: ``"BkgOnly"``  

    .. py:attribute:: patch_files_base_name 

        Base name (without .json extension) of the 'signal' patchx
        json files in the region folders (e.g. 'patch'). Patch files are extracted taking 
        all files in the region subfolders including the string ``patch_files_base_name``.  

        :type: ``str``
        :return: ``"patch"``    

    .. py:attribute:: out_folder    

        Path (either relative to the code execution folder or absolute)
        where output files are saved.
        The __init__ method automatically converts the path into an absolute path.
        If no output folder is specified, ``out_folder`` is set to the code execution folder.   

        :type: ``str`` or ``None``
        :return: ``None``   

    .. py:attribute:: histfactory_input_file    

        File name (either relative to the code execution folder or
        with full path) of a saved histfactory object.
        Whenever this parameter is not None all other parameters are ignored and the object is
        reconstructed from the imported file. 
        Histfactory objects are saved using pickle and contain a .pickle extension. ``histfactory_input_file``
        can contain or not the extension. In case it does not, extension is added by the ``__init__`` method.   

        :type: ``str`` or ``None``
        :return: ``None``   

Additional attributes
"""""""""""""""""""""

    .. py:attribute:: likelihoods_dict

        Main dictionary containing likelihoods parameters and properties for all regions/signal 
        hypotheses. All available likelihoods from the workspace are enumerated so that the dictionary integer keys corresponding
        to each likelihood object: ``_histfactory_additional_attrs = {1: value1, 2: value2, ...}``. 
        
        :type: ``dict``
        
        Keys are:

            + *"signal_region"* (type: ``str``)
                Name of the signal region to which the member belongs 
            + *"bg_only_file"* (type: ``str``)
                Absolute path to the background file corresponding to the signal_region.
            + *"patch_file"* (type: ``str``)
                Absolute path to the patch file for the given likelihood.
            + *"name"* (type: ``str``)
                Name of the given likelihood. It is set to "\ *hf_name*\ _region_\ *rn*\ _patch_\ *pn*\ _\ *lik_n*\ _likelihood"
                where *hf_name* is the ``histfactory.name`` attribute, *rn* is the region name determined from the region folder
                name excluded the string ``regions_folders_base_name``, *pn* is the patch name determined from the patch file name
                excluded the string ``patch_files_base_name+"."``, and *lik_n* is the likelihood number (the corresponding key
                in the ``likelihoods_dict``).
            + *"model_loaded"* (type: ``str``)
                Flag that returns ``False`` is the model is not loaded, i.e. only the items *"signal_region"*, *"bg_only_file"*, 
                *"patch_file"*, *"name"*, and *"model_loaded"* are available in the dictionary, and ``True`` if all dictionary items,
                i.e. full model information and ``pyhf.Workspace.model`` object, are available in the dictionary.
            + *"model"* (type: ``pyhf.Workspace.model`` object)
                ``pyhf.Workspace.model()`` object containing the given likelihood parameters and logpdf.
                See the `pyhf documentation <https://scikit-hep.org/pyhf/>`_.
            + *"obs_data"* (type: ``numpy.ndarray``, shape: ``(n_bins,)``)
                Numpy array containing the number of observed events in each of the n_bins bins for the given signal
                region.
            + *"pars_init"* (type: ``numpy.ndarray``, shape ``(n_pars,)``)
                Array with a length equal to the number of parameters n_pars
                entering in the likelihood (logpdf) function and containing their initial values.
            + *"pars_bounds"* (type: ``numpy.ndarray``, shape ``(n_pars,2)``)
                Array with lower and upper limit on each parameter of the n_pars parameters.
                The logpdf function is constructed such that if any of the parameter has a value outside these bounds, it evaluates
                to ``-np.inf``.
            + *"pars_labels"* (type: ``list``)
                List of strings containing the name of each parameter.
            + *"pars_pos_poi"* (type: ``numpy.ndarray``, shape: ``(n_pois)``)
                Array with the list of positions, in the array of parameters, of the n_pois parameters of interest.
            + *"pars_pos_nuis"* (type: ``numpy.ndarray``, shape: ``(n_nuis)``)
                Array with the list of positions, in the array of parameters, of the n_nuis nuisance parameters.

    .. py:attribute:: output_file_base_name

        base name of the output file of the ``histfactory.save_likelihoods()`` method. It is set to 
        ``histfactory.name+"_histfactory"``. The extension .pickle is not included and is added to 
        the output file when saving.

        :type: ``str```

    .. py:attribute:: regions

        dictionary containing region names (str) as keys and region folders full path (str) as values. 

        :type: ``dict``` 

.. _histfactory_methods
Methods
"""""""

    .. automethod:: source.histfactory.Histfactory.__init__

    .. automethod:: source.histfactory.Histfactory._Histfactory__import_histfactory

    .. automethod:: source.histfactory.Histfactory._Histfactory__load_histfactory

    .. automethod:: source.histfactory.Histfactory.import_histfactory

    .. automethod:: source.histfactory.Histfactory.save_histfactory

    .. automethod:: source.histfactory.Histfactory.get_lik_object