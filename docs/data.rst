.. _data_object:

The Data object
----------------------

Summary
^^^^^^^

Bla bla bla

Usage
^^^^^

Class
^^^^^

.. autoclass:: DNNLikelihood.Data
   :undoc-members:

.. _data_arguments:

Arguments
"""""""""

    .. option:: name

        Name of the :class:`Data <DNNLikelihood.Data>` object.
        It is used to build the :attr:`Data.name <DNNLikelihood.Data.name>` attribute.
         
            - **type**: ``str`` or ``None``
            - **default**: ``None``   

    .. option:: data_X

        |Numpy_link| array containing the x data points.
        It is used to build the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **default**: ``(npoints,ndim)``

    .. option:: data_Y

        |Numpy_link| array containing the x data points.
        It is used to build the :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **default**: ``(npoints,)``

    .. option:: dtype

        Data type. All data are converted, after import either from argument or from files, to the diven data type.
        It is used to build the :attr:`Data.dtype <DNNLikelihood.Data.dtype>` attribute.

            - **type**: ``str`` or ``None``
            - **default**: ``float64`` 

    .. option:: pars_pos_poi   

        List or |numpy_link| array containing the positions in the parameters list of the
        parameters of interest.
        It is used to build the :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>` attribute.

            - **type**: ``list`` or ``numpy.ndarray``
            - **shape**: ``(n_poi,)``
            - **default**: ``None`` 

    .. option:: pars_pos_nuis   

        List or |numpy_link| array containing the positions in the parameters list of the
        nuisance parameters.
        It is used to build the :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>` attribute.

            - **type**: ``list`` or ``numpy.ndarray``
            - **shape**: ``(n_nuis,)``
            - **default**: ``None`` 

    .. option:: pars_labels   

        List containing the parameters names as strings.
        Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.
        It is used to build the :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>` attribute.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``
            - **default**: ``None`` 

    .. option:: pars_bounds   

        List or |numpy_link| array containing containing bounds for the parameters.
        It is used to build the :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``(n_pars,2)``
            - **default**: ``None`` 

    .. option:: test_fraction

        Specifies the fraction of data that is used as test set. The first ``(1-test_fraction)`` fraction
        of points will be used to generate train and validation data while the remaining ``test_fraction``
        to generate test data.
        It is used to build the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute.

    .. option:: load_on_RAM

        If ``True`` all data available in the :attr:`Dara.input_h5_file <DNNLikelihood.Data.input_h5_file>`
        are loaded on RAM for faster generation of train/validation/test data. If ``False`` the HDF5 file is open in read
        mode and train/validation/test data are generated on demand.
        It is used to build the :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>` attribute.

    .. option:: output_folder
         
        Path (either relative to the code execution folder or absolute) where output files are saved.
        It is used to set the :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` attribute.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``

    .. option:: input_file

        File name (either relative to the code execution folder or absolute, with or without any of the
        .json or .h5 extensions) of a saved :class:`Data <DNNLikelihood.Data>` object. 
        It is used to set the 
        :attr:`Data.input_file <Data.Likelihood.input_file>` attribute.

           - **type**: ``str`` or ``None``
           - **default**: ``None``

    .. option:: verbose

        Argument used to set the verbosity mode of the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` 
        method and the default verbosity mode of all class methods that accept a ``verbose`` argument.
        See :ref:`Verbosity mode <verbosity_mode>`.

           - **type**: ``bool``
           - **default**: ``True``

Attributes
""""""""""


Methods
"""""""

    .. automethod:: DNNLikelihood.Data.__init__

    .. automethod:: DNNLikelihood.Data._Data__check_define_input_files

    .. automethod:: DNNLikelihood.Data._Data__check_define_output_files

    .. automethod:: DNNLikelihood.Data._Data__check_define_name

    .. automethod:: DNNLikelihood.Data._Data__check_define_pars

    .. automethod:: DNNLikelihood.Data._Data__check_define_mode

    .. automethod:: DNNLikelihood.Data._Data__init_mode

    .. automethod:: DNNLikelihood.Data._Data__create_data

    .. automethod:: DNNLikelihood.Data._Data__load

    .. automethod:: DNNLikelihood.Data._Data__check_data

    .. automethod:: DNNLikelihood.Data.define_test_fraction

    .. automethod:: DNNLikelihood.Data.close_samples

    .. automethod:: DNNLikelihood.Data.save

    .. automethod:: DNNLikelihood.Data.generate_train_indices

    .. automethod:: DNNLikelihood.Data.generate_train_data

    .. automethod:: DNNLikelihood.Data.update_train_indices

    .. automethod:: DNNLikelihood.Data.update_train_data

    .. automethod:: DNNLikelihood.Data.generate_test_indices

    .. automethod:: DNNLikelihood.Data.generate_test_data

    .. automethod:: DNNLikelihood.Data.compute_sample_weights

    .. automethod:: DNNLikelihood.Data.define_scalers