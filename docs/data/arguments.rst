.. _data_arguments:

Arguments
"""""""""

.. currentmodule:: Data

.. argument:: name

    See :argument:`name <common_classes_arguments.name>`.

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

    See :argument:`pars_central <common_classes_arguments.pars_central>`.

.. argument:: pars_pos_poi   

    See :argument:`pars_pos_poi <common_classes_arguments.pars_pos_poi>`.

.. argument:: pars_pos_nuis   

    See :argument:`pars_pos_nuis <common_classes_arguments.pars_pos_nuis>`.

.. argument:: pars_labels   

    See :argument:`pars_labels <common_classes_arguments.pars_labels>`.

.. argument:: pars_bounds   

    See :argument:`pars_bounds <common_classes_arguments.pars_bounds>`.

.. argument:: test_fraction

    It specifies the fraction of data that is used as test set. 
    It is used to build the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute.

.. argument:: load_on_RAM

    If ``True`` all data available in the HDF5 dataset :attr:`Dara.input_h5_file <DNNLikelihood.Data.input_h5_file>`
    are loaded into the RAM for faster generation of train/validation/test data. If ``False`` the HDF5 file is open in read
    mode and train/validation/test data are generated on demand.
    It is used to build the :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>` attribute.

.. argument:: output_folder
     
    See :argument:`output_folder <common_classes_arguments.output_folder>`.

.. argument:: input_file

    See :argument:`input_file <common_classes_arguments.input_file>`.

.. argument:: verbose

   See :argument:`verbose <common_classes_arguments.verbose>`.

.. include:: ../external_links.rst