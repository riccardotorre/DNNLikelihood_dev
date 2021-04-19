Attributes
""""""""""

.. currentmodule:: DNNLikelihood

.. py:attribute:: DnnLik._DnnLik__model_callbacks_inputs

   Private attribute corresponding to the input argument :argument:`model_callbacks_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_callbacks_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.

      - **type**: ``list``
      - **list items type**: ``str`` and/or ``dict`` (see :argument:`model_callbacks_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__model_compile_inputs

   Private attribute corresponding to the input argument :argument:`model_compile_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_compile_inputs` if present,
   otherwise it is constructed setting the loss to ``"mse"`` and the metrics to ``["mse","mae","mape","msle"]``. If the
   object is imported from files the attribute is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.

      - **type**: ``dict`` (see :argument:`model_compile_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__model_data_inputs

   Private attribute corresponding to the input argument :argument:`model_data_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_data_inputs`
   otherwise is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.
   If the number of validation and test data points are smaller than one, then they are treated as fractions of the
   corresponding number of training points and are replaced by the absolute numbers. If ``"scaleX"`, ``"scaleY"``,
   and/or ``"weighted"`` are not specified than they are set to ``False``.

      - **type**: ``dict`` (see :argument:`model_data_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__model_define_inputs

   Private attribute corresponding to the input argument :argument:`model_define_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_define_inputs`
   otherwise is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.
   If ``"act_func_out_layer"`, ``"dropout_rate"``,
   and/or ``"batch_norm"`` are not specified than they are set to ``"linear"``, ``0``, and ``False``, respectively.

      - **type**: ``dict`` (see :argument:`model_define_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__model_optimizer_inputs

   Private attribute corresponding to the input argument :argument:`model_optimizer_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_optimizer_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.
   For all arguments of the |tf_keras_optimizers_link_2| that are not specified the default value is used.

      - **type**: ``str`` or ``dict`` (see :argument:`model_optimizer_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__model_train_inputs

   Private attribute corresponding to the input argument :argument:`model_train_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_train_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.

      - **type**: ``dict`` (see :argument:`model_train_inputs` for the dictionary structure).

.. py:attribute:: DnnLik._DnnLik__resources_inputs

   Private attribute corresponding to the input argument :argument:`resources_inputs`. If the input argument is not given, 
   i.e. it is set to the default value ``None``, the dictionary is set by the 
   :meth:`DnnLik.__set_resources <DNNLikelihood.DnnLik._DnnLik__set_resources>` private method.

      - **type**: ``dict`` (see :argument:`resources_inputs` for the dictionary structure).

.. py:attribute:: DnnLik.act_func_out_layer

   String representing the activation function in the output layer. 
   It is set from the 
   :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: DnnLik.active_gpus

   List specifying the GPUs active in the current environment.
   For each GPU the list contains two strings with device ID and information, respectively.
   It is either set from the the 
   :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik._DnnLik__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`DnnLik.__set_resources <DNNLikelihood.DnnLik._DnnLik__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(n_active_gpus,2)``

.. py:attribute:: DnnLik.available_cpu

   List specifying the CPU resources.
   It contains three strings with device ID, specifications, number of cores, respectively.
   It is either set from the the 
   :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik._DnnLik__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`DnnLik.__set_resources <DNNLikelihood.DnnLik._DnnLik__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(3,)``

.. py:attribute:: DnnLik.available_gpus

   List specifying the GPUs available on the current machine.
   For each GPU the list contains two strings with device ID and information, respectively.
   It is either set from the the 
   :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik._DnnLik__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`DnnLik.__set_resources <DNNLikelihood.DnnLik._DnnLik__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(n_available_gpus,2)``

.. py:attribute:: DnnLik.batch_norm

   If ``True``, then |tf_keras_batch_normalization_link| layers are added after the input layer and between 
   each pair of hidden layers. It is set from the the 
   :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`
   dictionary (``False`` if not specified in the input argument :argument:`model_define_inputs`).

      - **type**: ``bool``

.. py:attribute:: DnnLik.batch_size

   Specifies the default batch size used for training, evaluation, and prediction.
   It is set from the  
   :attr:`DnnLik.__model_train_inputs <DNNLikelihood.DnnLik._DnnLik__model_train_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: DnnLik.callbacks

   List of |tf_keras_callbacks_link| objects.
   It is set by the :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>` 
   method by evaluating
   the corresponding strings in the :attr:`DnnLik.callbacks_strings <DNNLikelihood.DnnLik.callbacks_strings>`
   attribute.

      - **type**: ``list`` of |tf_keras_callbacks_link| objects

.. py:attribute:: DnnLik.callbacks_strings

   List of |tf_keras_callbacks_link| strings.
   It is set by the :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>` 
   method by parsing the content of the 
   :attr:`DnnLik.__model_callbacks_inputs <DNNLikelihood.DnnLik._DnnLik__model_callbacks_inputs>`
   dictionary.

      - **type**: ``list`` of ``str``

.. py:attribute:: DnnLik.data

   The :mod:`Data <data>` object used by the 
   :class:`DnnLik <DNNLikelihood.DnnLik>` one for data management.
   It is set to the value of the input ärgument :argument:`data` if passed. Otherwise,
   the :meth:`DnnLik.__set_data <DNNLikelihood.DnnLik._DnnLik__set_data>`
   sets it to a :mod:`Data <data>` object imported from the 
   :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>`.

      - **type**: :mod:`Data <data>` object

.. py:attribute:: DnnLik.data_max

   Dictionary containing the maximum of the DNNLikelihood computed with the 
   :meth:`DNNLik.model_compute_max <DNNLikelihood.DNNLik.model_compute_max>` method.
   It contains ``"X"`` and ``"Y"`` items corresponding to the X and Y values at
   the maximum of the :meth:`DNNLik.model_predict_scalar <DNNLikelihood.DNNLik.model_predict_scalar>`
   function.

      - **type**: ``dict`` with the following structure:

         - *"X"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
            |Numpy_link| array with the values of parameters at the model maximum.
         - *"Y"* (value type: ``float``)
            Value of the model at its maximum

.. py:attribute:: DnnLik.data_profiled_max

   To be written

.. py:attribute:: DnnLik.dropout_rate

   Specifies the dropout rate for the |tf_keras_dropout_link| layers.
   It is set from the  
   :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`
   dictionary. If the value is ``0`` no |tf_keras_dropout_link| layers are added to the model.

      - **type**: ``float``

.. py:attribute:: DnnLik.dtype

   Required data type for the generation of the train/validation/test datasets. The attribute is set by the
   :meth:`DnnLik.__set_dtype <DNNLikelihood.DnnLik._DnnLik__set_dtype>` methods to
   the value of the :argument:`dtype` if it is not ``None`` and to ``"float64"`` otherwise.

      - **type**: ``str``

.. py:attribute:: DnnLik.ensemble_folder

   If the :class:`DnnLik <DNNLikelihood.DnnLik>` object is a member of a 
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object, then this attribute is set to the
   parent directory of :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>`, where, by default,
   the :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object is stored
   (and corresponding to the 
   :attr:`DnnLikEnsemble.ensemble_folder <DNNLikelihood.DnnLikEnsemble.ensemble_folder>` attribute of the
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object).
   The attribute is by the 
   :meth:`DnnLik.__check_define_ensemble_folder <DNNLikelihood.DnnLik._DnnLik__check_define_ensemble_folder>`
   method based on the value of the 
   :attr:`DnnLik.ensemble_name <DNNLikelihood.DnnLik.ensemble_name>` attribute. If the latter is ``None``,
   then also :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` attribute is set to ``None``,
   the :class:`DnnLik <DNNLikelihood.DnnLik>` object is considered "standalone", and the 
   :attr:`DnnLik.standalone <DNNLikelihood.DnnLik.standalone>` is set to ``True``.

      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.ensemble_name

   Attribute corresponding to the input argument :argument:`ensemble_name`. 
   If the :class:`DnnLik <DNNLikelihood.DnnLik>` object is a member of a
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object, then the latter automatically passes
   the input argument :argument:`ensemble_name` when it creates the :class:`DnnLik <DNNLikelihood.DnnLik>` object.
   The attribute is by the 
   :meth:`DnnLik.__check_define_ensemble_folder <DNNLikelihood.DnnLik._DnnLik__check_define_ensemble_folder>`
   method. If the input argument is ``None``,
   then also the attribute is set to ``None``,
   the :class:`DnnLik <DNNLikelihood.DnnLik>` object is considered "standalone", and the 
   :attr:`DnnLik.standalone <DNNLikelihood.DnnLik.standalone>` is set to ``True``.

      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.epochs_available

   Number if epochs for which :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` has been trained. It is updated by
   the :meth:`DnnLik.model_train <DNNLikelihood.DnnLik.model_train>` method after each training.

      - **type**: ``int``

.. py:attribute:: DnnLik.epochs_required

   Number of epochs required for the training. The first time the object is created it is set to the value of the
   "epochs" item in the :attr:`DnnLik.__model_train_inputs <DNNLikelihood.DnnLik._DnnLik__model_train_inputs>`
   dictionary. It can be manually changed before calling the 
   :meth:`DnnLik.model_train <DNNLikelihood.DnnLik.model_train>` method to train for more epochs.
   Notice that each call to the 
   :meth:`DnnLik.model_train <DNNLikelihood.DnnLik.model_train>` method only trains for a number of epochs
   equal to the difference between :attr:`DnnLik.epochs_required <DNNLikelihood.DnnLik.epochs_required>`
   and :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>` (set by the
   private method :meth:`DnnLik.__set_epochs_to_run <DNNLikelihood.DnnLik._DnnLik__set_epochs_to_run>`).    

      - **type**: ``int``

.. py:attribute:: DnnLik.fig_base_title

   Common title for generated figures generated by the 
   :meth:`DNNLik.generate_fig_base_title <DNNLikelihood.DNNLik.generate_fig_base_title>` method and
   including information on 

      - :attr:`DNNLik.ndims <DNNLikelihood.DNNLik.ndims>`
      - :attr:`DNNLik.ndims <DNNLikelihood.DNNLik.npoints_train>`
      - :attr:`DNNLik.ndims <DNNLikelihood.DNNLik.hidden_layers>`
      - :attr:`DNNLik.ndims <DNNLikelihood.DNNLik.loss_string>`

      - **type**: ``list`` of ``str`` 

.. py:attribute:: DnnLik.figures_list

   List of absolute paths to the generated figures.

      - **type**: ``list`` of ``str`` 

.. py:attribute:: DnnLik.pars_labels_auto

   Copy of the :attr:`DnnLik.data.pars_labels_auto <DNNLikelihood.Data.pars_labels_auto>`
   attribute of the :mod:`Data <data>` object used for data management
   (see the doc of the :attr:`Data.pars_labels_auto <DNNLikelihood.Data.pars_labels_auto>` attribute).

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: DnnLik.gpu_mode

   Attribute set by the 
   :meth:`DnnLik.__set_resources <DNNLikelihood.DnnLik._DnnLik__set_resources>` method and indicating
   whether the object has GPU support (``True``) or not (``False``). If the object is a mamber of a 
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` one, then the attribute is set from the
   :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik._DnnLik__resources_inputs>` attribute.

      - **type**: ``bool`` 

.. py:attribute:: DnnLik.hidden_layers

   Attribute corresponding to the "hidden_layer" item of the 
   :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`
   attribute.
   It is a list with hidden layers specifications. All hidden layers in the module are |tf_keras_layers_dense_link| layers.
   Each hidden layer is represented by a list with number of nodes (``int``),
   activation function (``str``), and (optionally) initializer (``str``)

      - **type**: ``list``
      - **shape**: ``(n_hidden_layers,3)``

.. py:attribute:: DnnLik.history

   Attribute corresponding to the "history" dictionary of the ``History`` object returned by the |tf_keras_model_fit_link|
   method. It contains a list with the history of values of metrics for each training epoch. The contento of this 
   dictionary is saved into the file
   :attr:`DnnLik.output_history_json_file <DNNLikelihood.DnnLik.output_history_json_file>`.

      - **type**: ``dict`` with the following structure:

         - *"metric"* (value type: ``list``)
            List with values of metric for each epoch. This key is present for all metric with the corresponding 
            full name of the metric, such as, for instance, ``"mean_squared_error_best", etc. Metrics include ``"loss"``.
         - *"lr"* (value type: ``float``)
            List with values of the learning rate at each epoch.

.. py:attribute:: DnnLik.idx_test

   |Numpy_link| array with the integer position (indices) of the test data points in the data arrays contained in
   :attr:`DnnLik.data.data_X <DNNLikelihood.Data.data_X>` and
   :attr:`DnnLik.data.data_Y <DNNLikelihood.Data.data_Y>`.
   The same indices are also stored in the "idx_test" item of the 
   :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` object 
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_test,)``

.. py:attribute:: DnnLik.idx_train

   |Numpy_link| array with the integer position (indices) of the training data points in the data arrays contained in
   :attr:`DnnLik.data.data_X <DNNLikelihood.Data.data_X>` and
   :attr:`DnnLik.data.data_Y <DNNLikelihood.Data.data_Y>`.
   The same indices are also stored in the "idx_train" item of the 
   :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` object 
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_train,)``

.. py:attribute:: DnnLik.idx_val

   |Numpy_link| array with the integer position (indices) of the validation data points in the data arrays contained in
   :attr:`DnnLik.data.data_X <DNNLikelihood.Data.data.data_X>` and
   :attr:`DnnLik.data.data_Y <DNNLikelihood.Data.data.data_Y>`.
   The same indices are also stored in the "idx_val" item of the 
   :attr:`DnnLik.data.data_dictionary <DNNLikelihood.Data.data_dictionary>` attribute.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_val,)``

.. py:attribute:: DnnLik.input_data_file

   Absolute path corresponding to the input argument :argument:`input_data_file`. If the latter is not passed,
   :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>` is set to
   :class:`DnnLik.data.input_file <DNNLikelihood.Data.input_file>`.
           
        - **type**: ``str``

.. py:attribute:: DnnLik.input_files_base_name

   Absolute path with base file name corresponding to the input argument :argument:`input_summary_json_file` without
   the "_summary" suffix. Whenever this attribute is not ``None``, it is used to set all input file names and to
   reconstructed the object from input files (see the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>`
   method for details).
           
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_history_json_file

   Absolute path to the .json file containing saved :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>`
   attribute (see the :meth:`DnnLik.save_history_json <DNNLikelihood.DnnLik.save_history_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_idx_h5_file

   Absolute path to the .h5 file containing saved data indices (see the 
   :meth:`DnnLik.save_data_indices <DNNLikelihood.DnnLik.save_data_indices>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_log_file

   Absolute path to the .log file containing saved :class:`DnnLik <DNNLikelihood.DnnLik>`log (see the 
   :meth:`DnnLik.save_log <DNNLikelihood.DnnLik.save_log>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_predictions.h5_file

   Absolute path to the .json file containing saved 
   :attr:`DnnLik.predictions <DNNLikelihood.DnnLik.predictions>` (see the 
   :meth:`DnnLik.save_data_indices <DNNLikelihood.DnnLik.save_data_indices>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_scalers_pickle_file

   Absolute path to the .pickle file containing saved data standard scalers (see the 
   :meth:`DnnLik.save_scalers <DNNLikelihood.DnnLik.save_scalers>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_summary_json_file

   Absolute path to the .json file containing saved :class:`DnnLik <DNNLikelihood.DnnLik>` object 
   (see the :meth:`DnnLik.save_summary_json <DNNLikelihood.DnnLik.save_summary_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.input_tf_model_h5_file

   Absolute path to the .h5 file containing saved |tf_keras_model_link| corresponding to the 
   :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` 
   (see the :meth:`DnnLik.save_summary_json <DNNLikelihood.DnnLik.save_summary_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``
 
.. py:attribute:: DnnLik.load_on_RAM

   Attribute corresponding to the input argument :argument:`load_on_RAM`.
   If ``True`` all data available in the HDF5 dataset 
   :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>`
   are loaded into the RAM for faster generation of train/validation/test data. 
   If ``False`` the HDF5 file is open in read mode and train/validation/test data are generated on demand.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.log

   Dictionary containing a log of the :class:`Data <DNNLikelihood.DnnLik>` object calls. The dictionary has datetime 
   strings as keys and actions as values. Actions are also dictionaries, containing details of the methods calls.
           
      - **type**: ``dict``
      - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
      - **values**: ``dict`` with the following structure:

         - *"action"* (value type: ``str``)
            Short description of the action.
            **possible values**: ``"changed_output_folder"``, ``"created"``, ``""loaded summary json""``, ``"loaded history json"``, ``"loaded tf model h5"``, 
            ``"loaded scalers h5"``, ``"loaded data indices h5"``, ``"loaded predictions json"``, ``"optimizer set"``,
            ``"loss set"``, ``"metrics set"``, ``"callbacks set"``, ``"computed sample weights"``, ``"defined scalers"``,
            ``"generated train data"``, ``"generated test data"``, ``"defined tf model"``, ``"compiled tf model"``, ``"built tf model"``, 
            ``"trained tf model"``, ``"predicted with tf model"``, ``"computed maximum logpdf"``, ``"evaluated tf model"``,
            ``"saved figure"``, ``"computed predictions"``, ``"saved indices"``, ``"saved tf model json"``, ``"saved tf model h5"``,
            ``"saved tf model onnx"``, ``"saved history json"``, ``"saved summary json"``, ``"saved predictions json"``, 
            ``"saved scalers h5"``, ``"saved model graph pdf"``.
         - *"file name"* (value type: ``str``)
            File name of file involved in the action.
         - *"files names"* (value type: ``list`` of ``str``)
            List of file names of files involved in the action.
         - *"optimizer"* (value type: ``str``)
            String corresponding to the attribute :attr:`DnnLik.optimizer_string <DNNLikelihood.DnnLik.optimizer_string>`.
         - *"loss"* (value type: ``str``)
            String corresponding to the attribute :attr:`DnnLik.loss_string <DNNLikelihood.DnnLik.loss_string>`.
         - *"metrics"* (value type: ``list`` of ``str``)
            List corresponding to the attribute :attr:`DnnLik.metrics_string <DNNLikelihood.DnnLik.metrics_string>`.
         - *"callbacks"* (value type: ``list`` of ``str``)
            List corresponding to the attribute :attr:`DnnLik.callbacks_strings <DNNLikelihood.DnnLik.callbacks_strings>`.
         - *"scaler X"* (value type: ``bool``)
            Boolean corresponding to the attribute :attr:`DnnLik.scalerX_bool <DNNLikelihood.DnnLik.scalerX_bool>`.
         - *"scaler Y"* (value type: ``bool``)
            Boolean corresponding to the attribute :attr:`DnnLik.scalerY_bool <DNNLikelihood.DnnLik.scalerY_bool>`.
         - *"data"* (value type: ``list`` of ``str``)
            List of data involved in the action. It could include the strings: "idx_train", "X_train", "Y_train", 
            "idx_val", "X_val", "Y_val", "idx_test", "X_test", "Y_test".
         - *"npoints train"* (value type: ``int``)
            Number of training points corresponding to the attribute
            :attr:`DnnLik.npoints_train <DNNLikelihood.DnnLik.npoints_train>`.
         - *"npoints val"* (value type: ``int``)
            Number of validation points corresponding to the attribute
            :attr:`DnnLik.npoints_val <DNNLikelihood.DnnLik.npoints_val>`.
         - *"npoints test"* (value type: ``int``)
            Number of test points corresponding to the attribute
            :attr:`DnnLik.npoints_test <DNNLikelihood.DnnLik.npoints_test>`.
         - *"model summary"* (value type: ``list``)
            List rendering a print of the ``model.summary()`` method of |tf_keras_model_link|.
         - *"gpu mode"* (value type: ``bool``)
            Boolean corresponding to the attribute :attr:`DnnLik.gpu_mode <DNNLikelihood.DnnLik.gpu_mode>`. 
         - *"device id"* (value type: ``str``)
            String with the ID of the device on which model has been built (see the
            :meth:`DnnLik.model_build <DNNLikelihood.DnnLik.model_build>` method).
         - *"epochs run"* (value type: ``int``)
            Number of epochs of the current training run.
         - *"epochs total"* (value type: ``int``)
            Total number of training epochs.
         - *"batch size"* (value type: ``int``)
            Batch size used for action (training, predicting, evaluating), corresponding to the attribute
            :attr:`DnnLik.batch_size <DNNLikelihood.DnnLik.batch_size>`.
         - *"training time"* (value type: ``float``)
            Training time in seconds of the last training.
         - *"prediction time"* (value type: ``float``)
            Prediction time in seconds of the last prediction.
         - *"evaluation time"* (value type: ``float``)
            Evaluation time in seconds of the last evaluation.
         - *"npoints"* (value type: ``int``)
            Number of points involved in the corresponding action.
         - *"optimizer"* (value type: ``str``)
            String representing the optimizer used to maximise the logpdf.
         - *"optimization time"* (value type: ``float``)
            Time in second for computation of the maximum of logpdf.
         - *"probability intervals"* (value type: ``list``)
            Probability intervals used for predictions.
         - *"pars"* (value type: ``list``)
            Parameters involved in predictions.
         - *"old folder"* (value type: ``str``)
            Previous path.
         - *"new folder"* (value type: ``str``)
            New path.
                     
.. py:attribute:: DnnLik.loss

   |tf_keras_losses_link| object.
   It is set by the :meth:`DnnLik.__set_loss <DNNLikelihood.DnnLik._DnnLik__set_loss>` method by evaluating
   the corresponding strings in the :attr:`DnnLik.loss_string <DNNLikelihood.DnnLik.loss_string>`
   attribute.

      - **type**: |tf_keras_losses_link| object

.. py:attribute:: DnnLik.loss_string

   String representing a |tf_keras_losses_link| object.
   It is set by the :meth:`DnnLik.__set_loss <DNNLikelihood.DnnLik._DnnLik__set_loss>` method by parsing
   the content of the 
   :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: DnnLik.metrics

   List of |tf_keras_metrics_link| objects.
   It is set by the :meth:`DnnLik.__set_metrics <DNNLikelihood.DnnLik._DnnLik__set_metrics>` 
   method by evaluating
   the corresponding strings in the :attr:`DnnLik.metrics_string <DNNLikelihood.DnnLik.metrics_string>`
   attribute.

      - **type**: ``list`` of |tf_keras_metrics_link| objects

.. py:attribute:: DnnLik.metrics_string

   List of |tf_keras_metrics_link| strings.
   It is set by the :meth:`DnnLik.__set_metrics <DNNLikelihood.DnnLik._DnnLik__set_metrics>` 
   method by parsing the content of the 
   :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>`
   dictionary.

      - **type**: ``list`` of ``str``

.. py:attribute:: DnnLik.model

   |tf_keras_model_link| object representing the DNNLikelihood.

      - **type**: |tf_keras_model_link| object

.. py:attribute:: DnnLik.model_max

   Dictionary containing the maximum of the DNNLikelihood computed with the 
   :meth:`DNNLik.model_compute_max <DNNLikelihood.DNNLik.model_compute_max>` method.
   It contains x and y values at
   the maximum of the :meth:`DNNLik.model <DNNLikelihood.DNNLik.model>`
   function.
   It is initialized to an empty dictionary ``{}`` when the 
   :class:`DNNLik <DNNLikelihood.DNNLik>` object is created.

      - **type**: ``dict`` with the following structure:

         - *"x"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
            |Numpy_link| array with the values of parameters at the model maximum.
         - *"y"* (value type: ``float``)
            Value of the model at its maximum

.. py:attribute:: DnnLik.model_profiled_max

   To be written

.. py:attribute:: DnnLik.name

   Name of the :class:`DnnLik <DNNLikelihood.DnnLik>` object generated from
   the :argument:`name` input argument. If ``None`` is passed, then ``name`` is assigned the value 
   ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_DNNLikelihood"``.
   It is used to generate output files names.
       
      - **type**: ``str`` 

.. py:attribute:: DnnLik.ndims

   Number of dimensions of the input vector (i.e. number of 
   parameters entering in the logpdf). It is automatically set to the corresponding attribute 
   :attr:`Data.ndims <DNNLikelihood.Data.ndims>` of the :mod:`Data <data>` object
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`. 

      - **type**: ``int``

.. py:attribute:: DnnLik.npoints_available

   Total number of points available in the class:`Data <DNNLikelihood.Data>` object
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`, corresponding to the
   :attr:`Data.npoints <DNNLikelihood.Data.npoints>` attribute.

      - **type**: ``int``

.. py:attribute:: DnnLik.npoints_test

   Number of test points for the current model. It is set by the  
   :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: DnnLik.npoints_test_available

   Total number of test points available in the class:`Data <DNNLikelihood.Data>` object
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`, computed as the rounded product of
   one minus :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` and
   :attr:`DnnLik.npoints_available <DNNLikelihood.DnnLik.npoints_available>`.

      - **type**: ``int``

.. py:attribute:: DnnLik.npoints_train

   Number of training points for the current model. It is set by the  
   :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: DnnLik.npoints_train_val_available

   Total number of train plus validation points available in the class:`Data <DNNLikelihood.Data>` object
   :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`, computed as the rounded product of
   :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` and
   :attr:`DnnLik.npoints_available <DNNLikelihood.DnnLik.npoints_available>`.

         - **type**: ``int``

.. py:attribute:: DnnLik.npoints_val

   Number of validation points for the current model. It is set by the  
   :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: DnnLik.optimizer

   |tf_keras_optimizers_link| object.
   It is set by the :meth:`DnnLik.__set_optimizer <DNNLikelihood.DnnLik._DnnLik__set_optimizer>` 
   method by evaluating the corresponding strings in the 
   :attr:`DnnLik.optimizer_string <DNNLikelihood.DnnLik.optimizer_string>`
   attribute.

      - **type**: |tf_keras_optimizers_link| object

.. py:attribute:: DnnLik.optimizer_string

   String representing a |tf_keras_optimizers_link| object.
   It is set by the :meth:`DnnLik.__set_optimizer <DNNLikelihood.DnnLik._DnnLik__set_optimizer>`
   method by parsing the content of the 
   :attr:`DnnLik.__model_optimizer_inputs <DNNLikelihood.DnnLik._DnnLik__model_optimizer_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: DnnLik.output_checkpoints_files

   When the callback |tf_keras_model_checkpoint_callback_link| is present, this attribute represents the absolute path 
   to the .h5 files containing model checkpoints.
   It is initialized to ``None`` by the 
   :meth:`DnnLik.__check_define_output_files <DNNLikelihood.DnnLik._DnnLik__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>` method
   from the attribute 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.output_checkpoints_folder

   When the callback |tf_keras_model_checkpoint_callback_link| is present, this attribute represents the absolute path 
   to the folder where the .h5 files containing model checkpoints are saved.
   It is initialized to ``None`` by the 
   :meth:`DnnLik.__check_define_output_files <DNNLikelihood.DnnLik._DnnLik__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>` method
   from the attribute 
   :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.output_figure_plot_losses_keras_file

   When the callback |livelossplot_link| is present, this attribute represents the absolute path 
   to the folder where the corresponding figures are saved.
   It is initialized to ``None`` by the 
   :meth:`DnnLik.__check_define_output_files <DNNLikelihood.DnnLik._DnnLik__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>` method
   from the attribute 
   :attr:`DnnLik.output_figures_base_file <DNNLikelihood.DnnLik.output_figures_base_file>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.output_figures_base_file_name

   Base figures file name. It is 
   automatically generated from the 
   :attr:`DnnLik.name <DNNLikelihood.DnnLik.name>` attribute.

      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_figures_base_file_path

   Base figures file name including absolute path. It is 
   automatically generated from the
   :attr:`DnnLik.output_figures_folder <DNNLikelihood.DnnLik.output_figures_folder>` and 
   :attr:`DnnLik.output_figures_base_file_name <DNNLikelihood.DnnLik.output_figures_base_file_name>` attributes.

      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_figures_folder

   Absolute path to the folder where figures are saved. It includes the base figure name and is 
   automatically generated from the
   :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` attribute.

      - **type**: ``str``

.. py:attribute:: DnnLik.output_files_base_name

   Base name with absolute path of output files. It is 
   automatically generated from the
   :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` and
   :attr:`DnnLik.name <DNNLikelihood.DnnLik.name>` attributes.

      - **type**: ``str``

.. py:attribute:: DnnLik.output_folder

   Absolute path corresponding to the input argument
   :argument:`output_folder`. If the latter is ``None``, then 
   :attr:`output_folder <DNNLikelihood.DnnLik.output_folder>`
   is set to the code execution folder. If the folder does not exist it is created
   by the :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
   function.

      - **type**: ``str``

.. py:attribute:: DnnLik.output_history_json_file

   Absolute path to the .json file where the :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>` 
   attribute is saved (see the :meth:`DnnLik.save_history_json <DNNLikelihood.DnnLik.save_history_json>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_idx_h5_file

   Absolute path to the .h5 file where the data indices are saved (see the 
   :meth:`DnnLik.save_data_indices <DNNLikelihood.DnnLik.save_data_indices>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_log_file

   Absolute path to the .log file where the :attr:`DnnLik.log <DNNLikelihood.DnnLik.log>` 
   is saved (see the 
   :meth:`DnnLik.save_log <DNNLikelihood.DnnLik.save_log>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str``

.. py:attribute:: DnnLik.output_predictions_h5_file

   Absolute path to the .json file where the :attr:`DnnLik.predictions <DNNLikelihood.DnnLik.predictions>` 
   attribute is saved (see the :meth:`DnnLik.save_predictions_h5 <DNNLikelihood.DnnLik.save_predictions_h5>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_scalers_pickle_file

   Absolute path to the .pickle file where the data standard scalers are saved (see the 
   :meth:`DnnLik.save_scalers <DNNLikelihood.DnnLik.save_scalers>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_summary_json_file

   Absolute path to the .json file where where part of the :class:`DnnLik <DNNLikelihood.DnnLik>` 
   object is saved (see the 
   :meth:`DnnLik.save_summary_json <DNNLikelihood.DnnLik.save_summary_json>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_tensorboard_log_dir

   When the callback |tensorboard_link| is present, this attribute represents the absolute path 
   to the folder where log files are saved.
   It is automatically generated from the 
   :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` attribute.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: DnnLik.output_tf_model_graph_pdf_file

   Absolute path to the .pdf file where the model graph is saved (see the 
   :meth:`DnnLik.save_model_graph_pdf <DNNLikelihood.DnnLik.save_model_graph_pdf>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str``

.. py:attribute:: DnnLik.output_tf_model_h5_file

   Absolute path to the .h5 file where where the :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` 
   object is saved (see the 
   :meth:`DnnLik.save_model_h5 <DNNLikelihood.DnnLik.save_model_h5>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_tf_model_json_file

   Absolute path to the .json file where where the :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` 
   object is saved (see the 
   :meth:`DnnLik.save_model_json <DNNLikelihood.DnnLik.save_model_json>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.output_tf_model_onnx_file

   Absolute path to the .onnx file where where the :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` 
   object is saved in |onnx_link| format (see the 
   :meth:`DnnLik.save_model_onnx <DNNLikelihood.DnnLik.save_model_onnx>`
   method for details).
   It is automatically generated from the 
   :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: DnnLik.pars_bounds

   Copy of the :attr:`DnnLik.data.pars_bounds <DNNLikelihood.Data.pars_bounds>`
   attribute of the :mod:`Data <data>` object used for data management.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,2)``

.. py:attribute:: DnnLik.pars_bounds_train

   Bounds on the parameters computed from their minimum and maximum values in the training dataset.
   This attribute could be used to constrain the prediction of the DNNLikelihood within a region
   where ``X`` training data were available. See the :meth:`DNNLik.model_predict <DNNLikelihood.DNNLik.model_predict>`
   method for details.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,2)``

.. py:attribute:: DnnLik.pars_central   

   Copy of the :attr:`DnnLik.data.pars_central <DNNLikelihood.Data.pars_central>`
   attribute of the :mod:`Data <data>` object used for data management.
        
      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,)``

.. py:attribute:: DnnLik.pars_labels

   Copy of the :attr:`DnnLik.data.pars_labels <DNNLikelihood.Data.pars_labels>`
   attribute of the :mod:`Data <data>` object used for data management.

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: DnnLik.pars_pos_nuis

   Copy of the :attr:`DnnLik.data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`
   attribute of the :mod:`Data <data>` object used for data management.

      - **type**: ``list`` or ``numpy.ndarray``
      - **shape**: ``(n_nuis,)``

.. py:attribute:: DnnLik.pars_pos_poi

   Copy of the :attr:`DnnLik.data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`
   attribute of the :mod:`Data <data>` object used for data management.

      - **type**: ``list`` or ``numpy.ndarray``
      - **shape**: ``(n_poi,)``

.. py:attribute:: DnnLik.pred_bounds_train

   Maximum and minimum value of ``Y`` in training data.
   This attribute could be used to constrain the prediction of the DNNLikelihood within a region
   where ``Y`` training data were available. See the :meth:`DNNLik.model_predict <DNNLikelihood.DNNLik.model_predict>`
   method for details.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(2,)``

.. py:attribute:: DnnLik.predictions

   Nested dictionary containing all predictions generated by the 
   :meth:`DnnLik.model_compute_predictions <DNNLikelihood.DnnLik.model_compute_predictions>`
   method. The contento of this dictionary is saved into the file
   :attr:`DnnLik.output_predictions_h5_file <DNNLikelihood.DnnLik.output_predictions_h5_file>`.

      - **type**: ``dict`` with the following structure:

         - *"HPDI"* (value type: ``dict``)
            Highest posterior density intervals (HPDI) for the parameters. This dictionary has the following structure:

            - *"par"* (value type: ``dict``)
               Dictionary corresponding to parameter par. This dictionary has the following structure:

               - *"pred"* (value type: ``dict``)
                  Dictionary corresponding to the prediction computed reweighting distributions using the DNN prediction. 
                  Has the same structure as the item "true".
               - *"true"* (value type: ``dict``)
                  Dictionary corresponding to the prediction computed using the original data. 
                  This dictionary has the following structure:

                  - *"test"* (value type: ``dict``)
                     Dictionary corresponding to the prediction computed from test data.
                     Has the same structure as the items "train" and "val".
                  - *"train"* (value type: ``dict``)
                     Dictionary corresponding to the prediction computed from test data. 
                     Has the same structure as the items "test" and "val".
                  - *"val"* (value type: ``dict``)
                     Dictionary corresponding to the prediction computed from test data. 
                     This dictionary has the following structure:

                     - *"Probability"* (value type: ``dict``)
                        Dictionary with HPDI corresponding to the interval ``interval``. Default intervals for the
                        :meth:`DnnLik.model_compute_predictions <DNNLikelihood.DnnLik.model_compute_predictions>`
                        method are ``"0.5"``, ``"0.6826894921370859"``, ``"0.9544997361036416"``, and ``"0.9973002039367398"``.
                        The latter three correspond to 1,2,3 standard deviations for a 1D normal distribution (obtained through the
                        :func:`inference.CI_from_sigma <DNNLikelihood.inference.CI_from_sigma>` function).
                        This dictionary has the following structure:

                        - *"Bin width"* (value type: ``float``)
                           Width of the binning used to compute the HPDI (see the 
                           :func:`inference.HPDI <DNNLikelihood.inference.HPDI>`
                           function for details). It gives an estimate of the uncertainty due to the algorithm for computing
                           HPDI.
                        - *"Intervals"* (value type: ``list`` of ``list``)
                           List of intervals of parameter values for the given probability.
                        - *"Number of bins"* (value type: ``int``)
                           Number of bins used for the computation of HPDI (see the 
                           :func:`inference.HPDI <DNNLikelihood.inference.HPDI>`
                           function for details).
                        - *"Probability"* (value type: ``float``)
                           Probability interval.

         - *"HPDI_error"* (value type: ``dict``)
            Errors on the HPDI quantified as differences (absolute and relative) between the predictions of HPDI computed
            with input data and with DNN reweight. This dictionary has the following structure:

            - *"par"* (value type: ``dict``)
               Dictionary corresponding to parameter par. This dictionary has the following structure:

               - *"test"* (value type: ``dict``)
                  Dictionary corresponding to the prediction computed from test data.
                  Has the same structure as the items "train" and "val".
               - *"train"* (value type: ``dict``)
                  Dictionary corresponding to the prediction computed from test data. 
                  Has the same structure as the items "test" and "val".
               - *"val"* (value type: ``dict``)
                  Dictionary corresponding to the prediction computed from test data. 
                  This dictionary has the following structure:

                  - *"Probability"* (value type: ``dict``)
                     Dictionary with HPDI corresponding to the interval ``interval``. Default intervals for the
                     :meth:`DnnLik.model_compute_predictions <DNNLikelihood.DnnLik.model_compute_predictions>`
                     method are ``"0.5"``, ``"0.6826894921370859"``, ``"0.9544997361036416"``, and ``"0.9973002039367398"``.
                     The latter three correspond to 1,2,3 standard deviations for a 1D normal distribution (obtained through the
                     :func:`inference.CI_from_sigma <DNNLikelihood.inference.CI_from_sigma>` function).
                     This dictionary has the following structure:

                     - *"Absolute error"* (value type: ``list`` of ``list``)
                        List of absolute errors (computed as ``true-pred``) for each boundary of the intervals (see the 
                        :func:`inference.HPDI_error <DNNLikelihood.inference.HPDI_error>`
                        function for details)
                     - *"Probability"* (value type: ``float``)
                        Probability interval.
                     - *"Relative error"* (value type: ``list`` of ``list``)
                        List of relative errors (computed as ``(true-pred)/true``) for each boundary of the intervals (see the 
                        :func:`inference.HPDI_error <DNNLikelihood.inference.HPDI_error>`
                        function for details)

         - *"KS"* (value type: ``dict``)
            Weighted Kolmogorov-Smirnov test statistics and p-values for two sample distributions for each parameter
            (see the :func:`inference.ks_w <DNNLikelihood.inference.ks_w>` function for details).
            This dictionary has the following structure:

            - *"Test vs pred on train"* (value type: ``list``, shape: ``(ndims,2)``)
               List of KS test statistics and p-value for each parameter for two sample KS test between input test data and
               DNN reweighted prediction on train data.
            - *"Test vs pred on val"* (value type: ``list``, shape: ``(ndims,2)``)
               List of KS test statistics and p-value for each parameter for two sample KS test between input test data and
               DNN reweighted prediction on validation data.
            - *"Train vs pred on train"* (value type: ``list``, shape: ``(ndims,2)``)
               List of KS test statistics and p-value for each parameter for two sample KS test between input train data and
               DNN reweighted prediction on train data.
            - *"Val vs pred on test"* (value type: ``list``, shape: ``(ndims,2)``)
               List of KS test statistics and p-value for each parameter for two sample KS test between input validation data and
               DNN reweighted prediction on test data.

         - *"KS medians"* (value type: ``dict``)
            Median over parameters of the KS p-values for two sample distributions.
            This dictionary has the following structure:

            - *"Test vs pred on train"* (value type: ``float``)
               Median over parameters of the p-values for two sample KS test between input test data and
               DNN reweighted prediction on train data.
            - *"Test vs pred on val"* (value type: ``float``)
               Median over parameters of the p-values for two sample KS test between input test data and
               DNN reweighted prediction on validation data.
            - *"Train vs pred on train"* (value type: ``float``)
               Median over parameters of the p-values for two sample KS test between input train data and
               DNN reweighted prediction on train data.
            - *"Val vs pred on test"* (value type: ``float``)
               Median over parameters of the p-values for two sample KS test between input validation data and
               DNN reweighted prediction on test data.

         - *"Metrics on scaled data"* (value type: ``dict``)
            Value of all metrics (loss and metrics) evaluated on scaled data for train/val/test data.
            This dictionary has the following structure:

            - *"metric_best"* (value type: ``float``)
               Value of each metric evaluated on train data.
               This key is present for all metric with the corresponding full name of the metric, such as,
               for instance, ``"mean_squared_error_best", etc. Metrics include ``"loss"``.
            - *"test_metric_best"* (value type: ``float``)
               Value of each metric evaluated on test data.
               This key is present for all metric with the corresponding full name of the metric, such as,
               for instance, ``"test_mean_squared_error_best", etc. Metrics include ``"loss"``.
            - *"val_metric_best"* (value type: ``float``)
               Value of each metric evaluated on validation data.
               This key is present for all metric with the corresponding full name of the metric, such as,
               for instance, ``"val_mean_squared_error_best", etc. Metrics include ``"loss"``.
      
         - *"Metrics on unscaled data"* (value type: ``dict``)
            Value of all metrics (loss and metrics) evaluated on original data for train/val/test data.
            This dictionary has the same structure of the previous one with all keys having the suffix "_unscaled".

         - *"Prediction time"* (value type: ``float``)
            Averege prediction time per point in second with batch size equal to 
            :attr:`DnnLik.batch_size <DNNLikelihood.DnnLik.batch_size>`. 

      - **schematic example**:

         .. code-block:: python

            {'HPDI': {'par_1': {'pred': {'test': {'prob_1': {'Bin width': float,
                                                             'Intervals': list,
                                                             'Number of bins': int,
                                                             'Probability': float
                                                            },
                                                  ...,
                                                  {'prob_n': ...
                                                  },
                                         'train': ...,
                                         'val': ...
                                        },
                                'true': ...},
                      ...,
                      'par_n': ...
                     },
             'HPDI_error': {'par_1': {'test': {'prob_1': {'Absolute error': list,
                                                          'Probability': float,
                                                          'Relative error': list
                                                         },
                                               ...,
                                               {'prob_n': ...
                                               },
                                      'train': ...,
                                      'val': '...
                                     },
                            ...,
                            'par_n': ...
                           },
             'KS': {'Test vs pred on train': list,
                    'Test vs pred on val': list,
                    'Train vs pred on train': list,
                    'Val vs pred on test': list},
             'KS medians': {'Test vs pred on train': float,
                            'Test vs pred on val': float,
                            'Train vs pred on train': float,
                            'Val vs pred on test': float},
             'Metrics on scaled data': {'loss_best': float,
                                        'mean_absolute_error_best': float,
                                        ...
                                       },
             'Metrics on unscaled data': {'loss_best_unscaled': float,
                                          'mean_absolute_error_best_unscaled': float,
                                          ...
                                         },
             'Prediction time': float}

.. py:attribute:: DnnLik.same_data

   Attribute corresponding to the input argument :argument:`same_data`. 
   If ``True``, all the :class:`DnnLik <DNNLikelihood.DnnLik>` objects part of a  
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` one will share the same data.
   If false data will be generated differently for each :class:`DnnLik <DNNLikelihood.DnnLik>` object.
   When the :attr:`DnnLik.standalone <DNNLikelihood.DnnLik.standalone>` is ``True``, then this attribute
   is ``True`` by default and is not used.

      - **type**: ``bool``

.. py:attribute:: DnnLik.scalerX

   |standard_scaler_link| for X data points fit to the training data 
   :attr:`DnnLik.X_train <DNNLikelihood.DnnLik.X_train>`. Standard scalers are always defined 
   together with training data. Whenever the 
   :attr:`DnnLik.scalerX_bool <DNNLikelihood.DnnLik.scalerX_bool>` attribute is ``False``
   the scaler is just a representation of the identity matrix.

      - **type**: |standard_scaler_link| object

.. py:attribute:: DnnLik.scalerX_bool

   If ``True`` then the |standard_scaler_link| for the X data points is fit to the training data
   :attr:`DnnLik.X_train <DNNLikelihood.DnnLik.X_train>`, otherwise it is set to the identity matrix.
   It is set by the 
   :meth:`DnnLik.__set_model_hyperparameters <DNNLikelihood.DnnLik._DnnLik__set_model_hyperparameters>` 
   method by parsing the item ``"scalerX"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.

      - **type**: ``bool``

.. py:attribute:: DnnLik.scalerY

   |standard_scaler_link| for Y data points fit to the training data 
   :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.X_train>`. Standard scalers are always defined 
   together with training data. Whenever the 
   :attr:`DnnLik.scalerY_bool <DNNLikelihood.DnnLik.scalerY_bool>` attribute is ``False``
   the scaler is just a representation of the identity matrix.

      - **type**: |standard_scaler_link| object

.. py:attribute:: DnnLik.scalerY_bool

   If ``True`` then the |standard_scaler_link| for the Y data points is fit to the training data
   :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.Y_train>`, otherwise it is set to the identity matrix.
   It is set by the 
   :meth:`DnnLik.__set_model_hyperparameters <DNNLikelihood.DnnLik._DnnLik__set_model_hyperparameters>` 
   method by parsing the item ``"scalerY"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.

      - **type**: ``bool``

.. py:attribute:: DnnLik.seed

   Attribute corresponding to the input argument :argument:`seed`. If the latter is ``None`` then the attribute value is set
   to ``1``. It is used by the :meth:`DnnLik.__set_seed <DNNLikelihood.DnnLik._DnnLik__set_seed>`
   method to initialize the |numpy_link| and |tf_link| random state.

      - **type**: ``int``

.. py:attribute:: DnnLik.standalone

   If ``True`` the :class:`DnnLik <DNNLikelihood.DnnLik>` object is not part of a
   :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` one.
   It is set by the 
   :meth:`DnnLik.__check_define_ensemble_folder <DNNLikelihood.DnnLik._DnnLik__check_define_ensemble_folder>`
   method based on the value of the :attr:`DnnLik.ensemble_name <DNNLikelihood.DnnLik.ensemble_name>`
   attribute. If the latter is ``None`` then the attribute is set to ``True``.

      - **type**: ``bool``

.. py:attribute:: DnnLik.verbose

   Attribute corresponding to the input argument :argument:`verbose`.
   It represents the verbosity mode of the 
   :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` 
   method and the default verbosity mode of all class methods that accept a
   ``verbose`` argument.
   See :ref:`Verbosity mode <verbosity_mode>`.

       - **type**: ``bool`` or ``int``

.. py:attribute:: DnnLik.W_train

   |Numpy_link| array with the weights of the training points computed by the 
   :meth:`DnnLik.compute_sample_weights <DNNLikelihood.DnnLik.compute_sample_weights>` method.
   If :attr:`DnnLik.weighted <DNNLikelihood.DnnLik.weighted>`
   is ``True`` then the weights are computed automatically when generating training data by calling the
   :meth:`DnnLik.compute_sample_weights <DNNLikelihood.DnnLik.compute_sample_weights>` method
   with default parameters. Alternatively, the latter method can be called manually with custom arguments.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,)``

.. py:attribute:: DnnLik.weighted

   If ``True`` then, when generating training data, weights for the X data points are computed automatically 
   (with default parameters) by the 
   :meth:`DnnLik.compute_sample_weights <DNNLikelihood.DnnLik.compute_sample_weights>` method.
   It is set by the 
   :meth:`DnnLik.__set_model_hyperparameters <DNNLikelihood.DnnLik._DnnLik__set_model_hyperparameters>` 
   method by parsing the item ``"weighted"`` of the 
   :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
   dictionary.
   If ``True`` the :meth:`DnnLik.model_train <DNNLikelihood.DnnLik.model_train>` method calls the
   |tf_keras_model_fit_link| method with argument ``sample_weight`` equal to 
   :attr:`DnnLik.W_train <DNNLikelihood.DnnLik.W_train>`.

      - **type**: ``bool``

.. py:attribute:: DnnLik.X_test

   |Numpy_link| array with X test data points generated by the 
   :meth:`DnnLik.generate_test_data <DNNLikelihood.DnnLik.generate_test_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_test,ndim)`` 

.. py:attribute:: DnnLik.X_train

   |Numpy_link| array with X train data points generated by the 
   :meth:`DnnLik.generate_train_data <DNNLikelihood.DnnLik.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,ndim)`` 

.. py:attribute:: DnnLik.X_val

   |Numpy_link| array with X validation data points generated by the 
   :meth:`DnnLik.generate_train_data <DNNLikelihood.DnnLik.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_val,ndim)`` 

.. py:attribute:: DnnLik.Y_test

   |Numpy_link| array with Y test data points generated by the 
   :meth:`DnnLik.generate_test_data <DNNLikelihood.DnnLik.generate_test_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_test,)`` 

.. py:attribute:: DnnLik.Y_train

   |Numpy_link| array with Y train data points generated by the 
   :meth:`DnnLik.generate_train_data <DNNLikelihood.DnnLik.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,)`` 

.. py:attribute:: DnnLik.Y_val

   |Numpy_link| array with Y validation data points generated by the 
   :meth:`DnnLik.generate_train_data <DNNLikelihood.DnnLik.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_val,)`` 

.. include:: ../external_links.rst