Attributes
""""""""""

.. currentmodule:: DNNLikelihood

.. py:attribute:: Sampler.backend

    An ``emcee.Backends`` object (see the |emcee_backend_link| documentation for details).
    It is initialized (either from scratch or through an existing file) by the 
    :meth:`Sampler.__init_backend <DNNLikelihood.Sampler._Sampler__init_backend>` method.

        - **type**: ``emcee.Backends`` object

.. py:attribute:: Sampler.backend_file

    Name (with absolute path) of the |emcee_link| HDF5 backend file. 
    It is set to :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` ``+ "_backend.h5"``.
    See the |emcee_backend_link| documentation for details about the ``emcee.Backends`` object.

        - **type**: ``str``

.. py:attribute:: Sampler.figures_list

    List of absolute paths to the generated figures.

       - **type**: ``list`` of ``str`` 

.. py:attribute:: Sampler.input_file   

    Absolute path corresponding to the input argument :argument:`input_file`.
    Whenever the :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` attribute is ``False``,
    it is used to reconstructed the object from input files 
    (see the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>`
    method for details).
          
       - **type**: ``str`` or ``None``

.. py:attribute:: Sampler.input_h5_file    

    Absolute path to the .h5 file containing a saved :class:`Sampler <DNNLikelihood.Sampler>` object (see
    the :meth:`Sampler.save <DNNLikelihood.Sampler.save>` method for details).
    It is automatically generated from the attribute
    :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`.
    When the latter is ``None``, the attribute is set to ``None``.
         
        - **type**: ``str`` or ``None``

.. py:attribute:: Sampler.input_log_file   

    Absolute path to the .log file containing a saved :class:`Sampler <DNNLikelihood.Sampler>` object log (see
    the :meth:`Sampler.save_log <DNNLikelihood.Sampler.save_log>` method for details).
    It is automatically generated from the attribute
    :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`.
    When the latter is ``None``, the attribute is set to ``None``.
          
       - **type**: ``str`` or ``None``

.. py:attribute:: Sampler.likelihood_script_file

    Absolute path to the .py script containing the code necessary to intantiate a 
    :class:`Lik <DNNLikelihood.Lik>` object and define the corresponing parameters.  
    The :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method
    loads it as a module, which instantiate a :class:`Lik <DNNLikelihood.Lik>` object
    and defines parameters and logpdf.
    See the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method for details on how the attribute
    is set.
        
        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. py:attribute:: Sampler.log    

    Dictionary containing a log of the :class:`Sampler <DNNLikelihood.Sampler>` object calls. The dictionary has datetime 
    strings as keys and actions as values. Actions are also dictionaries, containing details of the methods calls.
          
        - **type**: ``dict``
        - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
        - **values**: ``dict`` with the following structure:

            - *"action"* (value type: ``str``)
               Short description of the action.
               **possible values**: ``"created"``, ``"loaded"``, ``"created backend"``, ``"loaded backend"``, 
               ``"init sampler"``, ``"run sampler"``, ``"saved"``, ``"computed Gelman-Rubin"``, ``"saved figure"``,
               ``"created data object"``.
            - *"pars"* (value type: ``list`` of ``int``)
               ``pars`` involved in the corresponding action.
            - *"available steps"* (value type: ``int``)
               Number of steps available in backend.
            - *"nsteps"* (value type: ``int``)
               ``nsteps`` involved of the corresponding action.
            - *"file name"* (value type: ``str``)
               File name of file involved in the action.
            - *"file path"* (value type: ``str``)
               Path of file involved in the action.
            - *"files names"* (value type: ``list`` of ``str``)
               List of file names of files involved in the action.
            - *"files paths"* (value type: ``list`` of ``str``)
               List of paths of files involved in the action.

.. py:attribute:: Sampler.logpdf:

    Callable function that could accept ``x_pars`` either as a single point in parameter space corresponding 
    to an array with shape ``(ndims,)`` or as a list of points corresponding to an array with 
    shape ``(n_points,ndims)`` and that returns either a ``float`` or a list of computed logpdf values,
    depending on the input. 
    In case of a scalar function the attribute :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` 
    is automatically set to False.
    The function could also accept additional arguments ``args``, passed through the 
    :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>` attribute.
    The attribute is assigned by the :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

        - **type**: ``callable``

    - **Could accept**

        - **x_par**

            Values of the parameters for which logpdf is computed.
            It could be a single point in parameter space corresponding to an array with shape ``(ndims,)``
            or a list of points corresponding to an array with shape ``(n_points,ndims)``.
                
                - **type**: ``numpy.ndarray``
                - **possible shapes**: ``(ndims,)`` or ``(n_points,ndims)``

        - **args**

            List of additional arguments required 
            by :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` and passed through the 
            :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>` attribute. 
                
                - **type**: ``list`` or None
                - **shape**: ``(nargs,)``
    
    - **Could return**
    
        Value or array of values 
        of the logpdf.
    
            - **type**: ``float`` or ``numpy.ndarray``
            - **shape for numpy.ndarray**: ``(n_points,)``

.. py:attribute:: Sampler.logpdf_args   

    Attribute containing a list of optional arguments (``*args``) for the :argument:`logpdf` function.
        
        - **type**: ``list`` or ``None``
        - **shape**: ``(nargs,)``

.. py:attribute:: Sampler.logpdf_kwargs   

    Attribute containing a dictionary of optional keyword arguments (``**kwargs``) for the :argument:`logpdf` function.
        
        - **type**: ``dict`` or ``None``

.. py:attribute:: Sampler.moves

    An ``emcee.moves`` object generated by evaluating the 
    :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` attribute.
    See the |emcee_moves_link| documentation for details.

        - **type**: ``emcee.moves`` object

.. py:attribute:: Sampler.moves_str

    String representing an instance to an ``emcee.moves`` object. If ``None``
    is passed, the default ``emcee.moves.StretchMove()`` is passed.
    See the |emcee_moves_link| documentation for details.

        - **type**: ``str`` or ``None``
        - **example**: ``"[(moves.StretchMove(0.7), 0.2), (moves.GaussianMove(0.1, mode='random',factor=None),0.8)]"``
            
            where ``moves`` represents the ``emcee.moves`` class.
            This gives a move that is 20% StretchMove with parameter 0.7 and 80% GaussianMove with covariance 0.1 and mode "random" (i.e.
            updating a single randomly chosen parameter at each step).

.. py:attribute:: Sampler.name   

    Name of the :class:`Sampler <DNNLikelihood.Sampler>` object. It is used to generate 
    output files. It is automatically generated from the corresponding attribute of the
    :class:`Lik <DNNLikelihood.Lik>` object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` 
    by the :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method
    by replacing the suffix "_likelihood" with the suffix "_sampler"
    
        - **type**: ``str``

.. py:attribute:: Sampler.ndims

    Number of dimensions of the input vector (i.e. number of 
    parameters entering in the logpdf). It is automatically set to the length of
    the first vector in :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`.

        - **type**: ``int``

.. py:attribute:: Sampler.new_sampler

    Attribute corresponding to the input argument :argument:`new_sampler`.
    If it is ``True`` a new :class:`Sampler <DNNLikelihood.Sampler>` object, corresponding to a new 
    :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` is generated. 
    If it is ``False`` the :class:`Sampler <DNNLikelihood.Sampler>` object
    is loaded from saved files, or, if a backend file is not found ``new_sampler`` is automatically set to ``True``
    and a new one is created.
        
        - **type**: ``bool``

.. py:attribute:: Sampler.nsteps_available

    Number of MCMC steps available in the current :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>`. 
    When the object is initialized with the :argument:`new_sampler` argument set to ``False``
    then, if :argument:`nsteps_required` is larger than :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`,
    then the :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` attribute is set equal to
    :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`.

        - **type**: ``int``

.. py:attribute:: Sampler.nsteps_required

    Attribute corresponding to the input argument :argument:`nsteps_required` and representing the
    final number of MCMC steps to run. 
    When the object is initialized with the :argument:`new_sampler` argument set to ``False``
    then, if :argument:`nsteps_required` is larger than :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`,
    then the :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` attribute is set equal to
    :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`.
    The attribute always represents the final number of steps, meaning that the sampling will always run 
    for a number of steps given by the difference between
    :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` and 
    :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`.
        
        - **type**: ``int``

.. py:attribute:: Sampler.nwalkers

    Number of walkers (equivalent of chains 
    for Ensamble Sampler MCMC). It is automatically set to the length of
    :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>` vector.
        
        - **type**: ``int``

.. py:attribute:: Sampler.output_figures_base_file

    Absolute path to the saved figures. It includes the base figure name and is 
    automatically generated from the
    :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and 
    :attr:`Sampler.name <DNNLikelihood.Sampler.name>` attributes.

       - **type**: ``str``

.. py:attribute:: Sampler.output_folder

    Absolute path to the folder where all output files are saved.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

       - **type**: ``str``

.. py:attribute:: Sampler.output_h5_file

    Absolute path to the .h5 file where the :class:`Sampler <DNNLikelihood.Lik>` 
    object is saved (see the :meth:`Sampler.save <DNNLikelihood.Sampler.save>`
    method for details).
    It is automatically generated from the attribute
    :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`.
          
       - **type**: ``str`` 

.. py:attribute:: Sampler.output_log_file

    Absolute path to the .log file where the :class:`Sampler <DNNLikelihood.Sampler>` 
    object log is saved (see the :meth:`Sampler.save_log <DNNLikelihood.Sampler.save_log>`
    method for details).
    It is automatically generated from the
    :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and 
    :attr:`Sampler.name <DNNLikelihood.Sampler.name>` attributes.

.. py:attribute:: Sampler.parallel_CPU

    Attribute corresponding to the input argument :argument:`parallel_CPU`.
    If ``True`` the MCMC is run in parallel on the available CPU cores, otherwise only a single core is used.

        - **type**: ``bool``

.. py:attribute:: Sampler.pars_bounds   

    |Numpy_link| array containing the parameters bounds.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

        - **type**: ``numpy.ndarray``
        - **shape**: ``(ndims,2)``

.. py:attribute:: Sampler.pars_central   

    |Numpy_link| array containing central values of the parameters.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.
        
        - **type**: ``numpy.ndarray``
        - **shape**: ``(ndims,)``

.. py:attribute:: Sampler.pars_init_vec

    Array of points with parameters initialization for each 
    walker. 
        
        - **type**: ``numpy.ndarray``
        - **shape**: ``(nwalkers,ndims)``
 
.. py:attribute:: Sampler.pars_labels   

    List containing parameters names as strings.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.
    Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
    and can contain latex expressions that are properly compiled when making plots.

        - **type**: ``list``
        - **shape**: ``(ndims,)``

.. py:attribute:: Sampler.pars_labels_auto   

    List containing parameters names automatically generated by the function
    :func:`utils.define_pars_labels_auto <DNNLikelihood.utils.define_pars_labels_auto>`.
    All parameters of interest are named ``r"$\theta_{i}$"`` with ``i`` ranging between
    one to the number of parameters of interest and all nuisance parameters are named
    ``r"$\nu_{j}$"`` with ``j`` ranging between one to the number of nuisance parameters.
    Parameters labels are always used as "raw" strings (like, for instance, ``r"%s"%pars_labels_auto[0]``) 
    and can contain latex expressions that are properly compiled when making plots.

        - **type**: ``list``
        - **shape**: ``(ndims,)``

.. py:attribute:: Sampler.pars_pos_nuis   

    |Numpy_link| array containing the positions in the parameters list of the nuisance parameters.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

        - **type**: ``numpy.ndarray``
        - **shape**: ``(n_nuis,)``

.. py:attribute:: Sampler.pars_pos_poi   

    |Numpy_link| array containing the positions in the parameters list of the parameters of interest.
    It is automatically set to the corresponding attribute of the :class:`Lik <DNNLikelihood.Lik>
    object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
    :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

        - **type**: ``numpy.ndarray``
        - **shape**: ``(n_poi,)``

.. py:attribute:: Sampler.sampler

    An ``emcee.EnsembleSampler`` object (see the |emcee_ensemble_sampler_link| documentation for details).
    It is initialized by the :meth:`Sampler.__init_sampler <DNNLikelihood.Sampler._Sampler__init_sampler>`
    method.

        - **type**: ``emcee.EnsembleSampler`` object

.. py:attribute:: Sampler.vectorize

    If ``True``, the function :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` is expected to accept a list of
    points and to return a list of logpdf values. When it is set to ``True`` the
    :meth:`Sampler.__check_vectorize <DNNLikelihood.Sampler._Sampler__check_vectorize>` checks the consistency by
    calling :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` on an array of points and, in case it fails,
    it sets :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` to ``False``.
    When it is ``True``, the :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>`
    attribute is automatically set to ``False``. 
    See the |emcee_ensemble_sampler_link| documentation for more details.

       - **type**: ``bool`` 

.. py:attribute:: Sampler.verbose

    Attribute corresponding to the input argument :argument:`verbose`.
    It represents the verbosity mode of the 
    :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` 
    method and the default verbosity mode of all class methods that accept a
    ``verbose`` argument.
    See :ref:`Verbosity mode <verbosity_mode>`.

        - **type**: ``bool`` or ``int``

.. include:: ../external_links.rst