Attributess
"""""""""""

.. currentmodule:: DNNLikelihood

.. py:attribute:: Histfactory.bkg_files_base_name

    Attribute corresponding to the input argument :argument:`bkg_files_base_name`.
    Background files are extracted taking all files in the region subfolders 
    including the string ``bkg_files_base_name``.
            
        - **type**: ``str``

.. py:attribute:: Histfactory.input_file

    See :attr:`input_file <common_classes_attributes.input_file>`.

.. py:attribute:: Histfactory.input_folder

    See :attr:`input_folder <common_classes_attributes.input_folder>`.

.. py:attribute:: Histfactory.input_h5_file

    See :attr:`input_h5_file <common_classes_attributes.input_h5_file>`.

.. py:attribute:: Histfactory.input_log_file

    See :attr:`input_log_file <common_classes_attributes.input_log_file>`.

.. py:attribute:: Histfactory.likelihoods_dict

    Dictionary containing likelihoods parameters and properties for all background/signal regions hypotheses. 
    All available likelihoods from the workspace are enumerated. Dictionary keys are integers running from ``0``
    to the total number of likelihoods in the workspace minus one, while values are dictionaries containing likelihoods information.
            
        - **type**: ``dict``
        - **keys**: ``int``
        - **values**: ``dict`` with the following structure:

            - *"signal_region"* (value type: ``str``)
                Name of the signal region to which the member belongs
            - *"bg_only_file"* (value type: ``str``)
                Name of the background only file corresponding to the "signal_region".
            - *"patch_file"* (value type: ``str``)
                Name of the patch file for the given likelihood.
            - *"name"* (value type: ``str``)
                Name of the given likelihood. It is set to "\ *hf_name*\ _region_\ *rn*\ _patch_\ *pn*\ _\ *lik_n*\ _likelihood"
                where *hf_name* is the :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` attribute, *rn* is the region name, *pn* is the patch name 
                determined from the patch file name, and *lik_n* is the likelihood number 
                (the corresponding key in the :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>` dictionary).
            - *"model_loaded"* (value type: ``str``)
                Flag that is set to ``False`` if the model is not loaded, i.e. only the items *"signal_region"*, *"bg_only_file"*, 
                *"patch_file"*, *"name"*, and *"model_loaded"* are available in the dictionary, and to ``True`` if all dictionary items,
                i.e. full model information and |pyhf_model_logpdf_link| object, are available in the dictionary.
            - *"model"* (value type: |pyhf_model_logpdf_link| object)
                Object containing the given likelihood parameters and logpdf.
                See the |pyhf_link| documentation.
            - *"obs_data"* (value type: ``numpy.ndarray``, value shape: ``(n_bins,)``)
                |Numpy_link| array containing the number of observed events in each of the ``n_bins`` bins for the given signal
                region.
            - *"pars_central"* (value type: ``numpy.ndarray``, value shape ``(ndims,)``)
                |Numpy_link| array with a length equal to the number of parameters ``ndims``
                entering in the likelihood (logpdf) function and containing their initial values.
            - *"pars_bounds"* (value type: ``numpy.ndarray``, value shape ``(ndims,2)``)
                |Numpy_link| array with lower and upper limit on each of the ``ndims`` parameters.
            - *"pars_labels"* (value type: ``list``)
                List of strings containing the name of each parameter. Parameters labels are always parsed as "raw" strings (like, for instance,
                ``r"%s"%pars_labels[0]``) and can contain latex expressions that are properly compiled when making plots.
            - *"pars_pos_poi"* (value type: ``numpy.ndarray``, value shape: ``(n_poi)``)
                |Numpy_link| array with the list of positions, in the array of parameters, of the ``n_poi`` parameters of interest.
            - *"pars_pos_nuis"* (value type: ``numpy.ndarray``, value shape: ``(n_nuis)``)
                |Numpy_link| array with the list of positions, in the array of parameters, of the ``n_nuis`` nuisance parameters.
            
            Available items in the dictionary depend the value of the "model_loaded" dictionary item. 
            See the :meth:`Histfactory.__import <DNNLikelihood.Histfactory._Histfactory__import>` and
            :meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` methods documentation for 
            more details.

.. py:attribute:: Histfactory.log

    See :attr:`log <common_classes_attributes.log>`.

.. py:attribute:: Histfactory.name

    See :attr:`name <common_classes_attributes.name>`.

.. py:attribute:: Histfactory.output_folder

    See :attr:`output_folder <common_classes_attributes.output_folder>`.

.. py:attribute:: Histfactory.output_h5_file

    See :attr:`output_h5_file <common_classes_attributes.output_h5_file>`.

.. py:attribute:: Histfactory.output_json_file

    See :attr:`output_json_file <common_classes_attributes.output_json_file>`.

.. py:attribute:: Histfactory.output_log_file

    See :attr:`output_log_file <common_classes_attributes.output_log_file>`.

.. py:attribute:: Histfactory.patch_files_base_name

    Attribute corresponding to the input argument :argument:`patch_files_base_name`.
    Patch files are extracted taking all files in the region subfolders including 
    the string stored in this attribute. 
            
        - **type**: ``str``

.. py:attribute:: Histfactory.regions
        
    Dictionary containing "Region" names (str) as keys 
    and "Region" folders full path (str) as values.
            
        - **type**: ``str`` or ``None``
        - **default**: ``None`` 

.. py:attribute:: Histfactory.regions_folders_base_name

    Attribute corresponding to the input argument :argument:`regions_folders_base_name`.
    When determining the regions, the 
    :meth:`Histfactory.__import <DNNLikelihood.Histfactory._Histfactory__import>` 
    method looks at all subfolders of 
    :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`
    containing the string 
    :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>`, 
    then deletes this latter string (and a dot) to obtain the region names and build the 
    :attr:`Histfactory.regions <DNNLikelihood.Histfactory.regions>` dictionary.
            
        - **type**: ``str``

.. py:attribute:: Histfactory.verbose

    See :attr:`verbose <common_classes_attributes.verbose>`.

.. py:attribute:: Histfactory.workspace_folder

    Absolute path of the folder "histfactory_workspace" located into
    the :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`.
    If it does not exist, such folder, containing the Histfactory Workspace, is created
    by copying the folder corresponding to the input argument :argument:`workspace_folder`.

        - **type**: ``str``

.. include:: ../external_links.rst