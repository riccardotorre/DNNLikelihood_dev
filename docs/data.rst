.. _data_object:

The Data object
----------------------

Summary
^^^^^^^

The :class:`Data <DNNLikelihood.Data>` class acts as a data container. It is used by the 
:class:`DNN_likelihood <DNNLikelihood.DNN_likelihood>` and 
:class:`DNN_likelihood_ensemble <DNNLikelihood.DNN_likelihood_ensemble>` objects to manage training/validation/test data
through the :class:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
See the documentation of :ref:`the DNN_likelihood object <DNN_likelihood_object>` and 
:ref:`the DNN_likelihood_ensemble object <DNN_likelihood_ensemble_object>` for more details on how the
:class:`Data <DNNLikelihood.Data>` object is used when building the DNNLikelihood.


.. _data_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Data <DNNLikelihood.Data>` class. Once created, 
the object is not supposed to be used directly by the used, but it is automatically used by the :class:`DNN_likelihood <DNNLikelihood.DNN_likelihood>` and 
:class:`DNN_likelihood_ensemble <DNNLikelihood.DNN_likelihood_ensemble>` objects to manage training/validation/test data.
For more details refer to the full class documentation and to the documentation of :ref:`the DNN_likelihood object <DNN_likelihood_object>` and 
:ref:`the DNN_likelihood_ensemble object <DNN_likelihood_ensemble_object>`.

The :class:`Data <DNNLikelihood.Data>` object can be directly created from a :class:`Sampler <DNNLikelihood.Sampler>` object or from
input arguments. An example of the first method is given in :ref:`the Sampler object Usage <sampler_usage>` section of the documentation
and consists of the following code:

.. code-block:: python

    import DNNLikelihood

    sampler = DNNLikelihood.Sampler(new_sampler=False, input_file=<my_output_folder>/toy_sampler)

    data = sampler.get_data_object(nsamples=200000, burnin=5000, thin=10, dtype="float64", test_fraction=0)

This gives a :class:`Data <DNNLikelihood.Data>` object with ``200000`` points which gets automatically stored in the files

    - <my_output_folder>/toy_data.h5
    - <my_output_folder>/toy_data.json 
    - <my_output_folder>/toy_data.log

To give an example of the creation of the object from input arguments we first define data arrays (for simplicity we use here
100000 points from the 100000 available in ``data``, the user can use any other data), and then create the object as follows:

.. code-block:: python

    data_X = data.data_X[:100000]
    data_Y = data.data_Y[:100000]
    dtype = ["float64","float32"]

    data_new = DNNLikelihood.Data(name = "new_data",
                                  data_X = data_X,
                                  data_Y = data_Y,
                                  dtype = ["float64","float32"],
                                  pars_pos_poi = [0],
                                  pars_pos_nuis = None,
                                  pars_labels = None,
                                  pars_bounds = None,
                                  test_fraction = 0.2,
                                  output_folder = <my_output_folder>,
                                  input_file = None,
                                  verbose = True)

This created the new object ``data_new`` and saves it to the files

    - <my_output_folder>/new_data.h5
    - <my_output_folder>/new_data.json
    - <my_output_folder>/new_data.log

See the documentation of the :meth:`Data.save <DNNLikelihood.Data.save>` and of the corresponding methods
with a :meth:`_h5 <DNNLikelihood.Data.save_h5>`,
:meth:`_json <DNNLikelihood.Data.save_json>`,
and :meth:`_log <DNNLikelihood.Data.save_log>` suffix.

If the :option:`name` argument is not passed, then one is automatically generated. In the example above we passed a list for the
:option:`dtype` argument, where the first entry represents the data type of the stored data, while the second is the required data
type of the train/validation/test data created from the :class:`Data <DNNLikelihood.Data>` object (see below). Concerning the arguments
related to parameters, if neither :option:`pars_pos_poi` nor :option:`pars_pos_nuis` are specified, then all parameters are assumed to
be parameters of interest, while if only one is specified, the other is automatically set. The arguments :option:`pars_labels` and
:option:`pars_bounds` can be optionally speficied, otherwise they are automatically set (see the full documentation for details).
The :option:`output_folder` is also optional and, if not specified, the output folder will automatically be set to the code execution
folder.

The fraction of train/validation vs test sets is specified through the :option:`test_fraction`, which, in our example, has been set to ``0.2``.
If :option:`test_fraction` is not specified (default is ``None``), it is automatically set to ``0`` and no test data will be available.
The fraction of train/validation vs test sets can be changed at any time by changing the 
:attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute and calling the 
:meth:`Data.define_test_fraction <DNNLikelihood.Data.define_test_fraction>` method as follows:

.. code-block:: python

    data.test_fraction = 0.3
    data.define_test_fraction()

This updates the attributes :attr:`Data.train_range <DNNLikelihood.Data.train_range>` and 
:attr:`Data.test_range <DNNLikelihood.Data.test_range>`, used for train/validation/test data generation.

The :class:`Data <DNNLikelihood.Data>` object can also be initialized importing it from saved files. 
In this case only the :option:`input_file` argument needs to be specified,
while all other arguments are ignored. One could also optionally specify a new :option:`output_folder`. 
In case this is not specified, the 
:attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>` attribute from the imported object is used and the object is
saved by updating existing files. If a new :option:`output_folder` is specified, then the whole object is saved to the new location.
One could decide, when importing the object, to load all data stored in the 
:attr:`Data.input_h5_file <DNNLikelihood.Data.input_h5_file>` into the RAM for faster generation of train/validation/test data
by setting to ``True`` the argument :option:`load_on_RAM`.
One could specify the option :option:`dtype`. This will affect neither saved data, nor the 
:attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>`, which is read from input files, but will change the value of the
:attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` that will affect the dtype of the generated 
train/validation/test data. If :option:`dtype` is not specified, then the attribute
:attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` is set equal to the 
:attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` attribute. Finally, also the argument :option:`test_fraction` can be
specified, which will update the corresponding attribute available from input files.
An example of code to create the object from input files is given by:

.. code-block:: python
    
   import DNNLikelihood

   data = DNNLikelihood.Data(input_file="<my_output_folder>/new_data",
                             dtype = "float32",
                             load_on_RAM = True
                            )

When the object is imported, the :attr:`Data.log <DNNLikelihood.Data.log>` 
attribute is updated and saved in the corresponding file :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`.

Data stored in the object can be accessed from the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` 
and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attributes as follows:

.. code-block:: python

    data.data_X

    >>> array([[-0.27304407, -0.46952336,  0.02573462, ..., -0.50863902, -1.12434653,  0.03133135],
               ...,
               [-0.1232975 ,  0.44791999, -0.21759863, ..., -1.04695986, -1.5426192 , -0.21464772]])

    data.data_Y

    >>> array([-58.52285607, -51.57067745, -52.23295711, ..., -52.55021968, -53.44788316, -54.33490431])

Upon creation of the :class:`Data <DNNLikelihood.Data>` object, the :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`
dictionary is initialized as an empyt dictionary containing empty |numpy_link| arrays with the correct ``dtype``:

.. code-block:: python

    data.data_dictionary

    >>> {'X_train': array([], shape=(1, 0), dtype=float32),
         'Y_train': array([], dtype=float32),
         'X_val': array([], shape=(1, 0), dtype=float32),
         'Y_val': array([], dtype=float32),
         'X_test': array([], shape=(1, 0), dtype=float32),
         'Y_test': array([], dtype=float32),
         'idx_train': array([], dtype=int32),
         'idx_val': array([], dtype=int32),
         'idx_test': array([], dtype=int32)}

Such dictionary is updated any time data are generated or updated. For instance, we could generate 10000 train, 5000 validation,
and 5000 test points as follows:

.. code-block:: python

    data.generate_train_data(npoints_train=10000, npoints_val=5000, seed=1)
    data.generate_test_data(npoints_test=5000)

    data.data_dictionary

    >>> {'X_train': array([[-2.8694883e-01,  4.3361101e-01, -4.6597528e-01, ..., -9.9797267e-01, -9.8734242e-01, -9.5092475e-01],
                            ...,
                           [-4.1338271e-01, -2.6408151e-01,  4.8945403e-01, ..., -7.6410377e-01, -1.5141882e+00, -4.5764816e-01]], dtype=float32),
         'Y_train': array([-52.884396, -55.569386, -54.26178 , ..., -52.6744  , -57.04926 , -55.466824], dtype=float32),
         'X_val':   array([[ 0.19161688, -0.45214146, -0.10997196, ..., -0.86869633, -1.1265236 ,  0.6219092 ],
                            ...,
                           [-0.46064648,  0.5269843 ,  0.19721822, ..., -0.47361913, -0.27832463, -0.06421166]], dtype=float32),
         'Y_val':   array([-53.06865 , -53.42976 , -51.243603, ..., -55.466824, -54.91352 , -50.538002], dtype=float32),
         'X_test':  array([[-0.75995547,  0.49505967,  0.58410156, ..., -1.2593005 , -0.32470152,  0.8180257 ],
                            ...,
                           [-0.79108167,  0.7511644 ,  0.51859653, ..., -0.10589346, 1.1247568 ,  0.6004382 ]], dtype=float32),
         'Y_test':  array([-52.339706, -53.3896  , -55.370182, ..., -53.374393, -53.940056, -53.93778 ], dtype=float32),
         'idx_train': array([   10,    12,    19, ..., 69991, 69993, 69996]),
         'idx_val': array([   14,    67,    74, ..., 69946, 69987, 69998]),
         'idx_test': array([70000, 70001, 70002, ..., 74997, 74998, 74999])}

Where we have also print the new value of the :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` attribute.
In the generation of train/validation data one can profit of the ``seed`` option to make the process reproducible. Indeed this
initializes the |numpy_link| random state to ``seed`` every time before selecting and splitting data. Concerning test data, 
these are always generated deterministically by taking the first ``npoints_test`` points in 
:attr:`Data.test_range <DNNLikelihood.Data.test_range>` range of data.

If scaling of ``X`` or ``Y`` data is necessary to train the neural network, one can define |standard_scalers_link| with the 
:meth:`Data.define_scalers <DNNLikelihood.Data.define_scalers>` method. For instance, scalers can be defined from the training 
data generated above with the following code:

.. code-block:: python

    data.define_scalers(data.data_dictionary["X_train"], data.data_dictionary["Y_train"], scalerX_bool=True, scalerY_bool=True)

    >>> [StandardScaler(copy=True, with_mean=True, with_std=True),
         StandardScaler(copy=True, with_mean=True, with_std=True)]

One could also define weights for data points, that allow to train on weighted data. This is done using the 
:meth:`Data.compute_sample_weights <DNNLikelihood.Data.compute_sample_weights>` method. This method bins data and weights data
with the inverse frequency of the bins allowing for an arbitrary exponent. For instance we can compute weights making an histogram
with ``50`` bins and weights equal to the inverse square root of frequencies as follows:

.. code-block:: python

    data.compute_sample_weights(data.data_dictionary["Y_train"], nbins=50, power=0.5)

    >>> array([0.70231453, 0.89124505, 0.72738372, ..., 0.71332963, 1.16068553, 0.89124505])

Notice that the :meth:`Data.define_scalers <DNNLikelihood.Data.define_scalers>` and 
:meth:`Data.compute_sample_weights <DNNLikelihood.Data.compute_sample_weights>` methods return a result and do not set any 
attribute.

Data in the :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` as well as |standard_scalers_link| and sample weights
are not stored when saving the :class:`Data <DNNLikelihood.Data>` object, and will only be saved as part of trained models and 
models ensemble by the :class:`DNN_likelihood <DNNLikelihood.DNN_likelihood>` and 
:class:`DNN_likelihood_ensemble <DNNLikelihood.DNN_likelihood_ensemble>` objects.
For more details refer to the full class documentation and to the documentation of :ref:`the DNN_likelihood object <DNN_likelihood_object>` and 
:ref:`the DNN_likelihood_ensemble object <DNN_likelihood_ensemble_object>`.

Even though the files corresponding to the saved object are usually kept sync with the object state, manual change of some attributes
does not update them. Nevertheless, the full object can be saved at any time through

.. code-block:: python 

    data.save(overwrite=True)

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

        |Numpy_link| array containing the X data points.
        It is used to build the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **default**: ``(npoints,ndim)``

    .. option:: data_Y

        |Numpy_link| array containing the Y data points.
        It is used to build the :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **default**: ``(npoints,)``

    .. option:: dtype

        The argument has a different meaning when creating the object (:option:`input_file` is ``None``)
        and when importing the object from files (:option:`input_file` is not ``None``) and is used to set 
        the :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and
        :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attributes.
        
        - :option:`input_file` is ``None``
        
            It represents the data type of the data that will be saved in 
            the :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>` dataset.

        - :option:`input_file` is ``None``

            It represents the data type required for the generation of train/validation/test datasets 
            (stored in the temporary attribute :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`).

        In the case in which, when the object is created, one needs a different dtype for stored data and for the 
        generation of train/validation/test datasets, one can pass a list of two strings corresponding to two dtypes
        that are assigned to the :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and
        :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attributes, respectively.
        When importing a saved :class:`Data <DNNLikelihood.Data>` object, if a list of two dtypes is passed, then the first
        is ignored (is always fixed from the saved data) and only the second is assigned to the 
        :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` attribute.

            - **type**: ``str`` or ``list`` or ``None``
            - **shape of list**: ``[,]`` (first entry for :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` and second entry for :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
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

        It specifies the fraction of data that is used as test set. 
        It is used to build the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute.

    .. option:: load_on_RAM

        If ``True`` all data available in the HDF5 dataset :attr:`Dara.input_h5_file <DNNLikelihood.Data.input_h5_file>`
        are loaded into the RAM for faster generation of train/validation/test data. If ``False`` the HDF5 file is open in read
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
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>` attribute.

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

    .. py:attribute:: DNNLikelihood.Data.data_X

        Attribute corresponding to the input argument :option:`data_X` and
        containing a |numpy_link| array with the X data points.

            - **type**: ``numpy.ndarray``
            - **default**: ``(npoints,ndim)``

    .. py:attribute:: DNNLikelihood.Data.data_Y

        Attribute corresponding to the input argument :option:`data_X` and
        containing a |numpy_link| array with the x data points.

            - **type**: ``numpy.ndarray``
            - **default**: ``(npoints,ndim)``

    .. py:attribute:: DNNLikelihood.Data.data_dictionary

    Dictionary used by the :class:`DNN_likelihood <DNNLikelihood.DNN_likelihood>` and 
    :class:`DNN_likelihood_ensemble <DNNLikelihood.DNN_likelihood_ensemble>` objects to keep track and store on the RAM
    the generated train/validation/test data (used, for instance, when training the DNNLikelihood and in making predictions).
    The dictionary is updated each time data are generated or updated and is not stored when saving the object (data
    indices only are stored with trained models by the :class:`DNN_likelihood <DNNLikelihood.DNN_likelihood>` and 
    :class:`DNN_likelihood_ensemble <DNNLikelihood.DNN_likelihood_ensemble>` objects).
    See :ref:`the DNN_likelihood object <DNN_likelihood_object>` and 
    :ref:`the DNN_likelihood_ensemble object <DNN_likelihood_ensemble_object>` objects for more
    information.

        - **type**: ``dict`` with the following structure:

            - *"X_train"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current training dataset X points.
            - *"Y_train"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current training dataset Y points.
            - *"X_val"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current validation dataset X points.
            - *"Y_val"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current validation dataset Y points.
            - *"X_test"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current test dataset X points.
            - *"Y_test"* (value type: ``numpy.ndarray`` with dtype equal to :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`)
               |Numpy_link| array with current test dataset Y points.
            - *"idx_train"* (value type: ``numpy.ndarray`` with dtype ``int32``)
               |Numpy_link| array with the indices of the training points inside :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`.
            - *"idx_val"* (value type: ``numpy.ndarray`` with dtype ``int32``)
               |Numpy_link| array with the indices of the validation points inside :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`.
            - *"idx_test"* (value type: ``numpy.ndarray`` with dtype ``int32``)
               |Numpy_link| array with the indices of the test points inside :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`.

    .. py:attribute:: DNNLikelihood.Data.dtype_stored

        Data type of the dataset stored in the :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>`.
        It is set to the value of the :option:`dtype` input argument if not ``None`` and to ``"float64"`` if ``None`` 
        the first time the object is created and remains unchanged when saving/loading the 
        :class:`Data <DNNLikelihood.Data>` object.

            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Data.dtype_required

        Required data type for the generation of the train/validation/test datasets. The attribute is always set to
        the value of the :option:`dtype` input argument if not ``None`` and to ``"float64"`` if ``None``.
        It represents the data type of the data stored in the attribute 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`.

            - **type**: ``str``
    
    .. py:attribute:: DNNLikelihood.Data.generic_pars_labels

        List containing parameters names automatically generated by the function
        :func:`utils.define_generic_pars_labels <DNNLikelihood.utils.define_generic_pars_labels>`.
        All parameters of interest are named ``r"$\theta_{i}$"`` with ``i`` ranging between
        one to the number of parameters of interest and all nuisance parameters are named
        ``r"$\nu_{j}$"`` with ``j`` ranging between one to the number of nuisance parameters.
        Parameters labels are always used as "raw" strings (like, for instance, ``r"%s"%generic_pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``

    .. py:attribute:: DNNLikelihood.Data.input_file

        Absolute path corresponding to the input argument :option:`input_file`.
        Whenever this attribute is not ``None``, it is used to reconstructed the object from input files 
        (see the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>`
        method for details).
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Data.input_h5_file

        Absolute path to the .h5 file containing saved :class:`Data <DNNLikelihood.Data>` HDF5 (see
        the :meth:`Data.save_h5 <DNNLikelihood.Data.save_h5>` method for details).
        It is automatically generated from the attribute
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Data.input_json_file

        Absolute path to the .json file containing saved :class:`Data <DNNLikelihood.Data>` json (see
        the :meth:`Data.save_json <DNNLikelihood.Data.save_json>`
        method for details).
        It is automatically generated from the attribute :attr:`Data.input_file <DNNLikelihood.Data.input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
             
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Data.input_log_file

        Absolute path to the .log file containing saved :class:`Data <DNNLikelihood.Data>` log (see
        the :meth:`Data.save_log <DNNLikelihood.Data.save_log>` method for details).
        It is automatically generated from the attribute
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Data.load_on_RAM

        Attribute corresponding to the input argument :option:`load_on_RAM`.
        If ``True`` all data available in the HDF5 dataset :attr:`Dara.input_h5_file <DNNLikelihood.Data.input_h5_file>`
        are loaded into the RAM for faster generation of train/validation/test data. 
        If ``False`` the HDF5 file is open in read mode and train/validation/test data are generated on demand.
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Data.log

        Dictionary containing a log of the :class:`Data <DNNLikelihood.Data>` object calls. The dictionary has datetime 
        strings as keys and actions as values. Actions are also dictionaries, containing details of the methods calls.
              
            - **type**: ``dict``
            - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
            - **values**: ``dict`` with the following structure:

                - *"action"* (value type: ``str``)
                   Short description of the action.
                   **possible values**: ``"created"``, ``"loaded"``, ``"saved"``, 
                   ``"updated data dictionary"``, ``"computed sample weights"``, ``"defines scalers"``.
                - *"data"* (value type: ``list`` of ``str``)
                   List of keys of the :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`
                   dictionary corresponding to values (data) that have been updated.
                - *"npoints train"* (value type: ``int``)
                   Number of training points available in the updated 
                   :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
                - *"npoints val"* (value type: ``int``)
                   Number of validation points available in the updated 
                   :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
                - *"npoints test"* (value type: ``int``)
                   Number of test points available in the updated 
                   :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
                - *"scaler X"* (value type: ``bool``)
                   When defining scalars it indicates if X points have been scaled.
                - *"scaler Y"* (value type: ``bool``)
                   When defining scalars it indicates if Y points have been scaled.
                - *"file name"* (value type: ``str``)
                   File name of file involved in the action.
                - *"file path"* (value type: ``str``)
                   Path of file involved in the action.
                - *"files names"* (value type: ``list`` of ``str``)
                   List of file names of files involved in the action.
                - *"files paths"* (value type: ``list`` of ``str``)
                   List of paths of files involved in the action.

    .. py:attribute:: DNNLikelihood.Data.name

        Attribute corresponding to the input argument :option:`name` and containing the
        name of the :class:`Data <DNNLikelihood.Data>` object. 
        If ``None`` is passed, then ``name`` is assigned the value 
        ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_data"``, 
        while if a string is passed, the ``"_data"`` suffix is appended 
        (preventing duplication if it is already present).
        It is used to generate output files names.

           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Data.ndims

        Number of dimensions of the X data (i.e. number of 
        parameters entering in the logpdf). It is automatically set to the length of
        the first vector in the input argument :option:`data_X`.

            - **type**: ``int``

    .. py:attribute:: DNNLikelihood.Data.npoints

        Number of X/Y data. It is automatically set to the length of
        the the input argument :option:`data_X`.

            - **type**: ``int``

    .. py:attribute:: DNNLikelihood.Data.opened_dataset

        Opened HDF5 dataset. When importing data with the :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>` 
        attribute set to ``False`` the HDF5 dataset is kept open until it is manually close with the
        :meth:`Data.close_opened_dataset <DNNLikelihood.Data.close_opened_dataset>` method.

            - **type**: ``HDF5 file`` object

    .. py:attribute:: DNNLikelihood.Data.output_folder

        Absolute path corresponding to the input argument
        :option:`output_folder`. If the latter is ``None``, then 
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>`
        is set to the code execution folder. If the folder does not exist it is created
        by the :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
        function.

           - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Data.output_h5_file

        Absolute path to the .h5 dataset file where the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and
        :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attributes are saved
        (see the :meth:`Data.save_log <DNNLikelihood.Data.save_log>`
        method for details).
        It is automatically generated from the
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` and 
        :attr:`Data.name <DNNLikelihood.Data.name>` attributes.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Data.output_json_file

        Absolute path to the .json file where part of the :class:`Data <DNNLikelihood.Data>` 
        object is saved (see the :meth:`Sampler.save_json <DNNLikelihood.Data.save_json>`
        method for details).
        It is automatically generated from the
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` and 
        :attr:`Data.name <DNNLikelihood.Data.name>` attributes.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Data.output_log_file

        Absolute path to the .log file where the :attr:`Data.log <DNNLikelihood.Data.log>` attribute
        is saved (see the :meth:`Data.save_log <DNNLikelihood.Data.save_log>`
        method for details).
        It is automatically generated from the
        :attr:`Sampler.output_folder <DNNLikelihood.Data.output_folder>` and 
        :attr:`Sampler.name <DNNLikelihood.Data.name>` attributes.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Data.pars_bounds

        Attribute corresponding to the input argument :option:`pars_bounds` and
        containing a |numpy_link| array with the parameters bounds. If the input argument is ``None``
        then bounds for all parameters are set to ``[-np.inf,np.inf]``.

            - **type**: ``numpy.ndarray``
            - **shape**: ``(n_pars,2)``

    .. py:attribute:: DNNLikelihood.Data.pars_labels

        List corresponding to the input argument :option:`pars_labels`. If the input argument is ``None`` then
        :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>` is set equal to the automatically
        generated :attr:`Data.generic_pars_labels <DNNLikelihood.Data.generic_pars_labels>`.
        Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``

    .. py:attribute:: DNNLikelihood.Data.pars_pos_nuis   

        |Numpy_link| array corresponding to the input argument :option:`pars_pos_nuis`.

            - **type**: ``list`` or ``numpy.ndarray``
            - **shape**: ``(n_nuis,)``

    .. py:attribute:: DNNLikelihood.Data.pars_pos_poi   

        |Numpy_link| array corresponding to the input argument :option:`pars_pos_poi`.

            - **type**: ``list`` or ``numpy.ndarray``
            - **shape**: ``(n_poi,)``

    .. py:attribute:: DNNLikelihood.Data.test_fraction

        Attribute corresponding to the input argument :option:`test_fraction`.
        It specifies the fraction of data that is used as test set. The first ``(1-test_fraction)`` fraction
        of points will be used to generate train and validation data while the remaining ``test_fraction``
        to generate test data.

    .. py:attribute:: DNNLikelihood.Data.test_range

        Attribute set from the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>`
        attribute and containing the range of test data indices in the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`
        |numpy_link| arrays.

    .. py:attribute:: DNNLikelihood.Data.train_range

        Attribute set from the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>`
        attribute and containing the range of train/validation data indices in the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`
        |numpy_link| arrays.

    .. py:attribute:: DNNLikelihood.Data.verbose

        Attribute corresponding to the input argument :option:`verbose`.
        It represents the verbosity mode of the 
        :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` 
        method and the default verbosity mode of all class methods that accept a
        ``verbose`` argument.
        See :ref:`Verbosity mode <verbosity_mode>`.

            - **type**: ``bool`` or ``int``

Methods
"""""""

    .. automethod:: DNNLikelihood.Data.__init__

    .. automethod:: DNNLikelihood.Data._Data__check_define_input_files

    .. automethod:: DNNLikelihood.Data._Data__check_define_output_files

    .. automethod:: DNNLikelihood.Data._Data__check_define_name

    .. automethod:: DNNLikelihood.Data._Data__check_define_data

    .. automethod:: DNNLikelihood.Data._Data__check_define_pars

    .. automethod:: DNNLikelihood.Data._Data__check_sync_data_dictionary

    .. automethod:: DNNLikelihood.Data._Data__load

    .. automethod:: DNNLikelihood.Data._Data__define_test_fraction

    .. automethod:: DNNLikelihood.Data.define_test_fraction

    .. automethod:: DNNLikelihood.Data.close_opened_dataset

    .. automethod:: DNNLikelihood.Data.save_log

    .. automethod:: DNNLikelihood.Data.save_json

    .. automethod:: DNNLikelihood.Data.save_h5

    .. automethod:: DNNLikelihood.Data.save

    .. automethod:: DNNLikelihood.Data.generate_train_indices

    .. automethod:: DNNLikelihood.Data.generate_train_data

    .. automethod:: DNNLikelihood.Data.update_train_indices

    .. automethod:: DNNLikelihood.Data.update_train_data

    .. automethod:: DNNLikelihood.Data.generate_test_indices

    .. automethod:: DNNLikelihood.Data.generate_test_data

    .. automethod:: DNNLikelihood.Data.compute_sample_weights

    .. automethod:: DNNLikelihood.Data.define_scalers

    .. py:method:: DNNLikelihood.Data.set_verbosity

      Method inherited from the :class:`Verbosity <DNNLikelihood.Verbosity>` object.
      See the documentation of :meth:`Verbosity.set_verbosity <DNNLikelihood.Verbosity.set_verbosity>`.

.. |numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> numpy</a>

.. |Numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> Numpy</a>