Attributes
""""""""""

.. currentmodule:: DNNLikelihood

.. py:attribute:: Lik.input_file   

   Absolute path corresponding to the input argument :argument:`input_file`.
   Whenever this parameter is not ``None`` the :class:`Lik <DNNLikelihood.Lik>` object
   is reconstructed from input files (see the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>`
   method for details).
          
      - **type**: ``str`` or ``None``

.. py:attribute:: Lik.input_folder   

   Absolute path corresponding to the folder containing the :argument:`input_file` file.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: Lik.input_h5_file    

   Absolute path to the .h5 file containing a saved :class:`Lik <DNNLikelihood.Lik>` object (see
   the :meth:`Lik.save <DNNLikelihood.Lik.save>` method for details).
   It is automatically generated from the attribute
   :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`.
   When the latter is ``None``, the attribute is set to ``None``.
        
       - **type**: ``str`` or ``None``

.. py:attribute:: Lik.input_log_file    

   Absolute path to the .log file containing a saved :class:`Lik <DNNLikelihood.Lik>` object log (see
   the :meth:`Lik.save_log <DNNLikelihood.Lik.save_log>` method for details).
   It is automatically generated from the attribute
   :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`.
   When the latter is ``None``, the attribute is set to ``None``.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: Lik.log    

   Dictionary containing a log of the :class:`Lik <DNNLikelihood.Lik>` object calls. The dictionary has datetime 
   strings as keys and actions as values. Actions are also dictionaries, containing details of the methods calls.
          
      - **type**: ``dict``
      - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
      - **values**: ``dict`` with the following structure:

         - *"action"* (value type: ``str``)
            Short description of the action.
            **possible values**: ``"computed maximum logpdf"``, ``"computed profiled maxima"``, ``"created"``, ``"loaded"``, ``"saved"``, ``"saved figure"``
         - *"pars"* (value type: ``list`` of ``int``)
            Input argument of the :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` method.
         - *"pars_ranges"* (value type: ``list`` of ``list``)
            Input argument of the :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` method.
         - *"number of maxima"* (value type: ``int``)
            Number of maxima computed by the :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` method.
         - *"file name"* (value type: ``str``)
            File name of file involved in the action.
         - *"file path"* (value type: ``str``)
            Path of file involved in the action.
         - *"files names"* (value type: ``list`` of ``str``)
            List of file names of files involved in the action.
         - *"files paths"* (value type: ``list`` of ``str``)
            List of paths of files involved in the action.

.. py:attribute:: Lik.logpdf

   Attribute corresponding to the input argument :argument:`logpdf`. When this input is ``None`` the argument
   :argument:`input_file` should be given to import the attribute from files.
   This function is used to construct the :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` method.
       
      - **type**: ``callable``

   - **Could accept**

      - **x_pars**
           
         Values of the parameters for which logpdf is computed.
         It could be a single point in parameter space corresponding to an array with shape ``(ndims,)``
         or a list of points corresponding to an array with shape ``(n_points,ndims)``.
             
            - **type**: ``numpy.ndarray``
            - **possible shapes**: ``(ndims,)`` or ``(n_points,ndims)``

      - **args**

         List of additional arguments required 
         by the :argument:`logpdf` function and passed through the :argument:`logpdf_args` input argument. 
             
            - **type**: ``list`` or None
            - **shape**: ``(nargs,)``
    
   - **Could return**
   
      Value or array of values 
      of the logpdf.
   
         - **type**: ``float`` or ``numpy.ndarray``
         - **shape for numpy.ndarray**: ``(n_points,)``

.. .. py:attribute:: Lik.logpdf_args   
.. 
..     Attribute corresponding to the input argument :argument:`logpdf_args`.
..         
..         - **type**: ``list`` or ``None``
..         - **shape**: ``(nargs,)``

.. py:attribute:: Lik.name

   Name of the :class:`Lik <DNNLikelihood.Lik>` object generated from
   the :argument:`name` input argument. If ``None`` is passed, then ``name`` is assigned the value 
   ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_likelihood"``, 
   while if a string is passed, the ``"_likelihood"`` suffix is appended 
   (preventing duplication if it is already present).
   It is used to generate output files names.
       
      - **type**: ``str``  

.. py:attribute:: Lik.ndims

   Number of dimensions of the likelihood, corresponding to the number of parameters.
   It is set equal to the length of the array
   :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`.
       
      - **type**: ``int``  

.. py:attribute:: Lik.output_figures_base_file

   Absolute path to the saved figures. It includes the base figure name and is 
   automatically generated from the
   :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>` and 
   :attr:`Lik.name <DNNLikelihood.Lik.name>` attributes.

      - **type**: ``str`` 

.. py:attribute:: Lik.output_figures_folder

   Absolute path to the folder where figures are saved. It is 
   automatically generated (and created by the 
   :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
   if not present) from the
   :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` attribute.

      - **type**: ``str`` 
 
.. py:attribute:: Lik.output_folder

   Absolute path corresponding to the input argument
   :argument:`output_folder`. If the latter is ``None``, then 
   :attr:`output_folder <DNNLikelihood.Lik.output_folder>`
   is set to the code execution folder. If the folder does not exist it is created
   by the :func:`utils.check_create_folder <DNNLikelihood.utils.check_create_folder>`
   function.

      - **type**: ``str``

.. py:attribute:: Lik.output_h5_file

   Absolute path to the .h5 file where the :class:`Lik <DNNLikelihood.Lik>` 
   object is saved (see the :meth:`Lik.save <DNNLikelihood.Lik.save>`
   method for details).
   It is automatically generated from the attribute
   :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: Lik.output_log_file

   Absolute path to the .log file where the :class:`Lik <DNNLikelihood.Lik>` 
   object log is saved (see the :meth:`Lik.save_log <DNNLikelihood.Lik.save_log>`
   method for details).
   It is automatically generated from the
   :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` and 
   :attr:`Lik.name <DNNLikelihood.Lik.name>` attributes.
         
      - **type**: ``str`` 

.. py:attribute:: Lik.output_predictions_json_file

   Absolute path to the .json file where the 
   :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>`
   dictionary is saved (see the :meth:`Lik.save_predictions <DNNLikelihood.Lik.save_predictions>`
   method for details).
   It is automatically generated from the
   :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` and 
   :attr:`Lik.name <DNNLikelihood.Lik.name>` attributes.
   Notice that the file is only saved to provide human readable predictions. The 
   :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` attribute is saved
   and restored together with the other attributes in the 
   :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` h5 file.

      - **type**: ``str``

.. py:attribute:: Lik.pars_bounds   

   |Numpy_link| array corresponding to the input argument :argument:`pars_bounds`. If the input argument is ``None``
   then bounds for all parameters are set to ``[-np.inf,np.inf]``.
   It is used to build the :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` so that
   the logpdf evaluates to ``-np.inf`` if any of the parameter has a value outside these bounds.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,2)``

.. py:attribute:: Lik.pars_central   

   Attribute corresponding to the input argument :argument:`pars_central` and
   containing a |numpy_link| array with the central values of the parameters. 
   If the input argument is ``None``
   then central values for all parameters are set to ``0``.
   This attribute is used as initial value for optimizing 
   :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` with the
   :meth:`Lik.compute_maximum_logpdf <DNNLikelihood.Lik.compute_maximum_logpdf>` and the
   :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` methods.
       
      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,)``

.. py:attribute:: Lik.pars_labels   

   List corresponding to the input argument :argument:`pars_labels`. If the input argument is ``None`` then
   :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>` is set equal to the automatically
   generated :attr:`Lik.pars_labels_auto <DNNLikelihood.Lik.pars_labels_auto>`.
   Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
   and can contain latex expressions that are properly compiled when making plots.

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: Lik.pars_labels_auto   

   List containing parameters names automatically generated by the function
   :func:`utils.define_pars_labels_auto <DNNLikelihood.utils.define_pars_labels_auto>`.
   All parameters of interest are named ``r"$\theta_{i}$"`` with ``i`` ranging between
   one to the number of parameters of interest and all nuisance parameters are named
   ``r"$\nu_{j}$"`` with ``j`` ranging between one to the number of nuisance parameters.
   Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels_auto[0]``) 
   and can contain latex expressions that are properly compiled when making plots.

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: Lik.pars_pos_nuis   

   |Numpy_link| array corresponding to the input argument :argument:`pars_pos_nuis`.

      - **type**: ``list`` or ```numpy.ndarray``
      - **shape**: ``(n_nuis,)``

.. py:attribute:: Lik.pars_pos_poi   

   |Numpy_link| array corresponding to the input argument :argument:`pars_pos_poi`.

      - **type**: ``list`` or ```numpy.ndarray``
      - **shape**: ``(n_poi,)``

.. py:attribute:: Lik.predictions

   Nested dictionary containing all predictions generated by the
   :meth:`Lik.compute_maximum_logpdf <DNNLikelihood.Lik.compute_maximum_logpdf>` and
   :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` methods,
   and the list of figures generated by the
   :meth:`Lik.plot_logpdf_par <DNNLikelihood.Lik.plot_logpdf_par>` and
   :meth:`Lik.plot_tmu_1d <DNNLikelihood.Lik.plot_tmu_1d>` methods.   

      - **type**: ``dict`` with the following structure:

         - *"logpdf_max"* (value type: ``dict``)
            Dictionary containing information on global maximum. This dictionary has the following structure:

            - *"timestamp"* (value type: ``dict``)
               Dictionary containing information on global maximum computed at time ``timestamp``. 
               The key is a string of the form ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``.
               This dictionary has the following structure:

               - *"x"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
                  Value of the parameters at the maximum of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`.
               - *"y"* (value type: ``numpy.float64``)
                  Maximum of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`.
               - *"pars_init"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
                  Initial value of the parameters for the optimization. 
               - *"optimizer"* (value type: ``dict``)
                  Dictionary containing information on the optimizer. This dictionary has the following structure:
   
                  - *"name"* (value type: ``str``, default: ``"scipy"``)
                     Optimizer name. Up to now only ``"scipy"`` is supported.
                  - *"method"* (value type: ``str``, default: ``"Powell"``)
                     Optimizer method.
                  - *"name"* (value type: ``dict``)
                     Optimizer options. Keys are options names and values are option values.

               - *"optimization_time"* (value type: ``float``)
                  Optimization time.

         - *"logpdf_profiled_max"* (value type: ``dict``)
            Dictionary containing information on global maxima (profiled maxima). This dictionary has the following structure:

            - *"timestamp"* (value type: ``dict``)
               Dictionary containing information on global maximum computed at time ``timestamp``. 
               The key is a string of the form ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``.
               This dictionary has the following structure:

               - *"X"* (value type: ``numpy.ndarray``, shape: ``(npoints,ndims)``)
                  Value of the parameters at the maxima of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` for fixed values
                  of ``pars``.
               - *"Y"* (value type: ``numpy.ndarray``, shape: ``(npoints,)``)
                  Maxima of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` for fixed values of ``pars``.
               - *"pars"* (value type: ``list``, shape: ``(npars,)``)
                  Parameters that are kept fixed.
               - *"pars_ranges"* (value type: ``list``, shape: ``(npars,3)``)
                  Ranges of parameters that are kept fixed. For each parameter the list has the form ``[min,max,npoints]``.
               - *"pars_init"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
                  Same as in the ``logpdf_max`` dictionary above.
               - *"pars_bounds"* (value type: ``numpy.ndarray``, shape: ``(ndims,2)``)
                  Boundaries of the parameter space. For each of all parameters it has the form ``[min,max]``.
               - *"optimizer"* (value type: ``dict``)
                  Same as in the ``logpdf_max`` dictionary above.
               - *"optimization_times"* (value type: ``float``)
                  List with optimization times for each of the minima.
               - *"global_optimization_time"* (value type: ``float``)
                  Total optimization time for ``pars`` in ``pars_ranges``.

         - *"Figures"* (value type: ``dict``)
            Dictionary containing information on figures. This dictionary has the following structure:

            - *"timestamp"* (value type: ``list``)
               Dictionary containing the list of absolute paths of the figures produced at time ``timestamp``.

      - **schematic example**:

         .. code-block:: python

            {'logpdf_max': {'timestamp1': {'x': numpy.ndarray,
                                          'y': numpy.float64,
                                          'pars_init': numpy.ndarray,
                                          'optimizer': {'method': str,
                                                        'options': {'maxiter': int, 
                                                                    'ftol': float},
                                                        'name': str},
                                          'optimization_time': float},
                            'timestamp2': ...}
             'logpdf_profiled_max': {'timestamp1': {'X': numpy.ndarray,
                                                   'Y': numpy.ndarray,
                                                   'tmu': numpy.ndarray,
                                                   'pars': list,
                                                   'pars_ranges': list,
                                                   'pars_init': numpy.ndarray,
                                                   'pars_bounds': numpy.ndarray,
                                                   'optimizer': {'method': str,
                                                                 'options': {'maxiter': int, 
                                                                             'ftol': float},
                                                                 'name': str},
                                                   'optimization_times': list,
                                                   'global_optimization_time': numpy.float64},
                                     'timestamp2': ...},
             'Figures': {'timestamp1': [str, ...],
                         'timestamp2': [str, ...],
                         ...}

.. py:attribute:: Lik.script_file

   Absolute path to the .py script containing the code necessary to intantiate a 
   :class:`Lik <DNNLikelihood.Lik>` object and define the corresponing parameters. 
   This file can be generated using the 
   :meth:`Lik.save_script <DNNLikelihood.Lik.save_script>` method
   and is used to initialize a :class:`Sampler <DNNLikelihood.Sampler>` object 
   (see :ref:`the Sampler object <sampler_object>`). This is to ensure that that Markov Chain Monte Carlo properly 
   runs in parallel (using the |multiprocessing_link| package) inside Jupyter notebooks also on the Windows OS.
   It is automatically generated from the
   :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` and 
   :attr:`Lik.name <DNNLikelihood.Lik.name>` attributes.

      - **type**: ``str``

.. py:attribute:: Lik.verbose

   Attribute corresponding to the input argument :argument:`verbose`.
   It represents the verbosity mode of the 
   :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` 
   method and the default verbosity mode of all class methods that accept a
   ``verbose`` argument.
   See :ref:`Verbosity mode <verbosity_mode>`.

      - **type**: ``bool`` or ```int``

.. include:: ../external_links.rst