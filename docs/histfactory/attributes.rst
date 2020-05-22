Attributess
"""""""""""

.. currentmodule:: DNNLikelihood

.. py:attribute:: Histfactory.bkg_files_base_name

   Attribute corresponding to the input argument :argument:`bkg_files_base_name`.
   Background files are extracted taking all files in the region subfolders 
   including the string ``bkg_files_base_name``.
         
      - **type**: ``str``

.. py:attribute:: Histfactory.input_file

   Absolute path corresponding to the input argument :argument:`input_file`.
   Whenever this parameter is not ``None`` the :class:`Histfactory <DNNLikelihood.Histfactory>` object
   is reconstructed from input files (see the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>`
   method for details).
         
      - **type**: ``str`` or ``None``

.. py:attribute:: Histfactory.input_h5_file    

   Absolute path to the .h5 file containing a saved :class:`Histfactory <DNNLikelihood.Histfactory>` object (see
   the :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>` method for details).
   It is automatically generated from the attribute
   :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`.
   When the latter is ``None``, the attribute is set to ``None``.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: Histfactory.input_log_file    

   Absolute path to the .log file containing a saved :class:`Histfactory <DNNLikelihood.Histfactory>` object log (see
   the :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>` method for details).
   It is automatically generated from the attribute
   :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`.
   When the latter is ``None``, the attribute is set to ``None``.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: Histfactory.likelihoods_dict

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
         - *"pars_central"* (value type: ``numpy.ndarray``, value shape ``(ndims,)``)
            Numpy array with a length equal to the number of parameters ``ndims``
            entering in the likelihood (logpdf) function and containing their initial values.
         - *"pars_bounds"* (value type: ``numpy.ndarray``, value shape ``(ndims,2)``)
            Numpy array with lower and upper limit on each of the ``ndims`` parameters.
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

.. py:attribute:: Histfactory.log    

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

.. py:attribute:: Histfactory.name

   Attribute corresponding to the input argument :argument:`name` and containing the
   name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object. 
   If ``None`` is passed, then ``name`` is assigned the value 
   ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_histfactory"``, 
   while if a string is passed, the ``"_histfactory"`` suffix is appended 
   (preventing duplication if it is already present).
   It is used to generate output files names and is passed to the generated 
   :class:`Lik <DNNLikelihood.Lik>` objects.
         
      - **type**: ``str`` 

.. py:attribute:: Histfactory.output_folder

   Absolute path corresponding to the input argument
   :argument:`output_folder`. If the latter is ``None``, then 
   :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
   is set to the corresponding attribute of the
   :class:`Lik <DNNLikelihood.Lik>` object.
   If the folder does not exist it is created
   by the :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
   function.

      - **type**: ``str``

.. py:attribute:: Histfactory.output_h5_file

   Absolute path to the .h5 file where the :class:`Histfactory <DNNLikelihood.Histfactory>` 
   object is saved (see the :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>`
   method for details).
   It is automatically generated from the attribute
   :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: Histfactory.output_log_file

   Absolute path to the .log file where the :class:`Histfactory <DNNLikelihood.Histfactory>` 
   object log is saved (see the :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>`
   method for details).
   It is automatically generated from the attribute
   :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: Histfactory.patch_files_base_name

   Attribute corresponding to the input argument :argument:`patch_files_base_name`.
   Patch files are extracted taking all files in the region subfolders including 
   the string :attr:`Histfactory.patch_files_base_name <DNNLikelihood.Histfactory.patch_files_base_name>`
   attribute. 
         
      - **type**: ``str``

.. py:attribute:: Histfactory.regions
      
   Dictionary containing "Region" names (str) as keys 
   and "Region" folders full path (str) as values.
         
      - **type**: ``str`` or ``None``
      - **default**: ``None`` 

.. py:attribute:: Histfactory.regions_folders_base_name

   Attribute corresponding to the input argument :argument:`regions_folders_base_name`.
   When determining the regions, the :meth:`Histfactory.__import <DNNLikelihood.Histfactory._Histfactory__import>` 
   method looks at all subfolders of :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`
   containing the string :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>`, 
   then deletes this latter string (and a dot) to obtain the region names and build the :attr:`Histfactory.regions <DNNLikelihood.Histfactory.regions>` 
   dictionary.
         
      - **type**: ``str``

.. py:attribute:: Histfactory.verbose

   Attribute corresponding to the input argument :argument:`verbose`.
   It represents the verbosity mode of the 
   :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` 
   method and the default verbosity mode of all class methods that accept a
   ``verbose`` argument.
   See :ref:`Verbosity mode <verbosity_mode>`.
 
      - **type**: ``bool`` or ``int``

.. py:attribute:: Histfactory.workspace_folder

   Absolute path of the ATLAS histfactory workspace folder
   corresponding to the input argument :argument:`workspace_folder`.

      - **type**: ``str``

.. include:: ../external_links.rst