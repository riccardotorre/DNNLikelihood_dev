.. _data_arguments:

Arguments
"""""""""

.. currentmodule:: Data

.. argument:: name

    Name of the :mod:`Data <data>` object.
    It is used to build the :attr:`Data.name <DNNLikelihood.Data.name>` attribute.
     
        - **type**: ``str`` or ``None``
        - **default**: ``None``   

.. argument:: data_X

    |Numpy_link| array containing the X data points.
    It is used to build the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` attribute.

        - **type**: ``numpy.ndarray`` or ``None``
        - **default**: ``(npoints,ndim)``

.. argument:: data_Y

    |Numpy_link| array containing the Y data points.
    It is used to build the :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attribute.

        - **type**: ``numpy.ndarray`` or ``None``
        - **default**: ``(npoints,)``

.. argument:: dtype

    The argument has a different meaning when creating the object (:argument:`input_file` is ``None``)
    and when importing the object from files (:argument:`input_file` is not ``None``) and is used to set 
    the :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and
    :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attributes.
    
    - :argument:`input_file` is ``None``
    
        It represents the data type of the data that will be saved in 
        the :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>` dataset.

    - :argument:`input_file` is ``None``

        It represents the data type required for the generation of train/validation/test datasets 
        (stored in the temporary attribute :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`).

    In the case in which, when the object is created, one needs a different dtype for stored data and for the 
    generation of train/validation/test datasets, one can pass a list of two strings corresponding to two dtypes
    that are assigned to the :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and
    :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attributes, respectively.
    When importing a saved :mod:`Data <data>` object, if a list of two dtypes is passed, then the first
    is ignored (is always fixed from the saved data) and only the second is assigned to the 
    :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attribute.

        - **type**: ``str`` or ``list`` or ``None``
        - **shape of list**: ``(2,)`` (first entry for :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and second entry for :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
        - **default**: ``float64``

.. argument:: pars_central   

    List or |numpy_link| array containing central values of the parameters of the original lilekelihood.
    It is used to build the :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>` attribute.
        
        - **type**: ``list`` or ```numpy.ndarray``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_pos_poi   

    List or |numpy_link| array containing the positions in the parameters list of the
    parameters of interest.
    It is used to build the :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>` attribute.

        - **type**: ``list`` or ``numpy.ndarray``
        - **shape**: ``(n_poi,)``
        - **default**: ``None`` 

.. argument:: pars_pos_nuis   

    List or |numpy_link| array containing the positions in the parameters list of the
    nuisance parameters.
    It is used to build the :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>` attribute.

        - **type**: ``list`` or ``numpy.ndarray``
        - **shape**: ``(n_nuis,)``
        - **default**: ``None`` 

.. argument:: pars_labels   

    List containing the parameters names as strings.
    Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
    and can contain latex expressions that are properly compiled when making plots.
    It is used to build the :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>` attribute.

        - **type**: ``list``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_bounds   

    List or |numpy_link| array containing containing bounds for the parameters.
    It is used to build the :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` attribute.

        - **type**: ``numpy.ndarray`` or ``None``
        - **shape**: ``(ndims,2)``
        - **default**: ``None`` 

.. argument:: test_fraction

    It specifies the fraction of data that is used as test set. 
    It is used to build the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute.

.. argument:: load_on_RAM

    If ``True`` all data available in the HDF5 dataset :attr:`Dara.input_h5_file <DNNLikelihood.Data.input_h5_file>`
    are loaded into the RAM for faster generation of train/validation/test data. If ``False`` the HDF5 file is open in read
    mode and train/validation/test data are generated on demand.
    It is used to build the :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>` attribute.

.. argument:: output_folder
     
    Path (either relative to the code execution folder or absolute) where output files are saved.
    It is used to set the :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` attribute.
        
        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: input_file

    File name (either relative to the code execution folder or absolute, with or without extension
    and with or without the "_object" suffix) 
    of a saved :mod:`Data <data>` object. 
    It is used to set the 
    :attr:`Data.input_file <DNNLikelihood.Data.input_file>` attribute.

       - **type**: ``str`` or ``None``
       - **default**: ``None``

.. argument:: verbose

    Argument used to set the verbosity mode of the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` 
    method and the default verbosity mode of all class methods that accept a ``verbose`` argument.
    See :ref:`Verbosity mode <verbosity_mode>`.

       - **type**: ``bool``
       - **default**: ``True``

.. include:: ../external_links.rst