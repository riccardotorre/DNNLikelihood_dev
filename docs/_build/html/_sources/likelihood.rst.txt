.. _likelihood_object:

The Likelihood object
----------------------

Summary
^^^^^^^

The :class:`Likelihood <DNNLikelihood.Likelihood>` class acts as a container for the likelihood function. 
It contains information on parameters initializations, positions, bounds, and labels, the logpdf function and its arguments, and methods that allow 
one to plot the logpdf funcion and to compute its (profiled and global) maximiza. 
In case in which the likelihood is obtained using the interface to the ATLAS histfactory workspaces given by  
:ref:`the Histfactory object <histfactory_object>`, the logpdf is constructed from the |pyhf_model_logpdf_link| method.

Usage
^^^^^

We give here a bried introduction to the use of the :class:`Likelihood <DNNLikelihood.Histfactory>` class. Refer to the 
full class documentation for more details.

The :class:`Likelihood <DNNLikelihood.Likelihood>` object can be created both by directly inputing the relevant arguments or
automatically,
in case the likelihood function comes from an ATLAS histfactory workspace, by the :class:`Histfactory <DNNLikelihood.Histfactory>` object 
through the :meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>` method. In the 
:ref:`the Histfactory object Usage <histfactory_usage>` section of the documentation we already gave an example of the latter method.
We give here a very simple (toy) example of creation of the object from direct likelihood_input_json_file.

The first time a :class:`Likelihood <DNNLikelihood.Likelihood>` object is created, the :option:`logpdf`, :option:`logpdf_args` (
if required by :option:`logpdf`), :option:`pars_pos_poi`, :option:`pars_pos_nuis`, and :option:`pars_init` arguments need to be specified. 
Optionally, also the arguments :option:`pars_labels` and :option:`pars_bounds` related to likelihood parameters can be specified.
Moreover, the user may specify the additional arguments
:option:`output_folder` containing the path (either relative or absolute) to a folder where output files will be saved and
:option:`name` with the name of the object (which is otherwise automatically generated).

To give a simple example, let us start by creating a very simple toy experiment with ``10`` bins, one nuisance parameter per 
bin and one signal strength parameter. The ``logpdf`` function could be defined by the cose:

.. code-block:: python

    import numpy as np

    nbI_nominal = np.array(list(reversed([i for i in range(100,1100,100)])))    # Nominal background
    nbI_obs = np.random.poisson(nbI_nominal)                                    # Observed counts
    nsI_reference = np.array(list(reversed([i for i in range(10,110,10)])))     # Signal prediction for signal strength mu=1

    def nbI(delta):
        # Background in each bin as function of the 10 nuisance parameters delta
        delta = np.array(delta)
        return np.array([nbI_nominal[i]*(1+0.1)**delta[i] for i in range(len(delta))])

    def nsI(mu):
        # Signal in each bin as function of the signal strength parameter mu
        return mu*nsI_reference

    def nI(pars):
        # Expected counts in each bin
        mu = pars[0]
        delta = pars[1:]
        return np.array(nsI(mu)+nbI(delta))

    def loglik(pars, obs):
        # Log of Poisson likelihood
        exp = nI(pars)
        logfact = np.array(list(map(lambda x: np.math.lgamma(x+1), obs)))
        return np.sum(-1*logfact+obs*np.log(exp)-exp)   

    def logprior(pars):
        # Log of normal distribution for deltas and uniform [-5,5] distribution for mu
        mu = pars[0]
        delta = pars[1:]
        delta_prior = -1/2*np.sum(delta**2+np.full(len(delta),np.log(2*np.pi)))
        return delta_prior-np.log(1/(10))
        
    def logpdf(pars, obs):
        # Sum of log-likelihood and log-prior
        return loglik(pars, obs)+logprior(pars)

This takes as arguments the parameters (mu, delta) and the observed counts. We can now define arguments related to 
parameters (we will not define labels, that will be automatically set by the object initialization)
and initialize the :class:`Likelihood <DNNLikelihood.Likelihood>` with a few lines of code:

.. code-block:: python

    import DNNLikelihood
    
    pars_pos_poi = [0]
    pars_pos_nuis = range(1,11)
    pars_init = np.insert(np.full(10,0),0,1)
    pars_labels = None,
    pars_bounds = np.concatenate((np.array([[-5,5]]),
                                  np.vstack([np.full(10,-np.inf),
                                             np.full(10,np.inf)]).T))

    likelihood = DNNLikelihood.Likelihood(name = 'toy',
                                          logpdf = logpdf,
                                          logpdf_args = [nbI_obs],
                                          pars_pos_poi = pars_pos_poi,
                                          pars_pos_nuis = pars_pos_nuis,
                                          pars_init = pars_init,
                                          pars_labels = None,
                                          pars_bounds = pars_bounds,
                                          output_folder = "<my_output_folder>")

When the object is created, it is automatically saved and three files are created:

   - <my_output_folder>/toy_likelihood.pickle
   - <my_output_folder>/toy_likelihood.json 
   - <my_output_folder>/toy_likelihood.log

See the documentation of the :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` and of the corresponding methods
with a :meth:`_json <DNNLikelihood.Likelihood.save_likelihood_json>`,
:meth:`_log <DNNLikelihood.Likelihood.save_likelihood_log>`,
and :meth:`_pickle <DNNLikelihood.Likelihood.save_likelihood_pickle>` suffix.

The object can also be initialized importing it from saved files. In this case only the :option:`likelihood_input_file` argument needs to be specified,
while all other arguments are ignored. One could also optionally specify a new ``output_folder``. In case this is not specified, the 
:attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute from the imported object is used.
For instance we could import the object created above with

.. code-block:: python
    
   import DNNLikelihood

   likelihood = DNNLikelihood.Likelihood(likelihood_input_file="<my_output_folder>/toy_likelihood")

The logpdf for a given value of the parameters (for instance the 
:attr:`Likelihood.pars_init <DNNLikelihood.Likelihood.pars_init>`
can be obtained through

.. code-block:: python

    likelihood.logpdf_fn(likelihood.pars_init,*likelihood.logpdf_args)

We can check the logpdf depencence on the input parameters by plotting it with the method
:meth:`Likelihood.plot_logpdf_par <DNNLikelihood.Likelihood.plot_logpdf_par>`. For instance, one can get the plot
for the parameters ``0`` (signal strength) and ``5`` (nuisance parameter) in the range ``(-1,1)`` with all other
parameters set to their value in :attr:`Likelihood.pars_init <DNNLikelihood.Likelihood.pars_init>`, the plots can be
obtained through

.. code-block:: python

    likelihood.plot_logpdf_par([[0,-1,1],[5,-1,1]])

This prints the following plots in the active console

.. image:: figs/toy_likelihood_figure_par_0.png
    :class: with-shadow
    :scale: 50

.. image:: figs/toy_likelihood_figure_par_5.png
    :class: with-shadow
    :scale: 50

And saves two files, whose path is stored in the :attr:`Likelihood.figures_list <DNNLikelihood.Likelihood.figures_list>`.
One could also optionally choose a different central value for the parameters that are kept fixed by passing an argument
``pars_init`` to the :meth:`Likelihood.plot_logpdf_par <DNNLikelihood.Likelihood.plot_logpdf_par>` method.

The maximum of the logpdf, and the corresponding parameters values can be obtained with the 
:meth:`Likelihood.compute_maximum_logpdf <DNNLikelihood.Likelihood.compute_maximum_logpdf>` and is stored in the 
:attr:`Likelihood.X_logpdf_max <DNNLikelihood.Likelihood.X_logpdf_max>` and
:attr:`Likelihood.Y_logpdf_max <DNNLikelihood.Likelihood.Y_logpdf_max>` attributes:

.. code-block:: python

    likelihood.compute_maximum_logpdf()
    print(likelihood.X_logpdf_max)
    print(likelihood.Y_logpdf_max)

    >>> [0.04380427  0.27652363  0.02134356 -0.15662528 -0.0267759  -0.30837557
         0.35269854 -0.36984361 -0.08494277 -0.13147428  0.52011438]
    >>> 47.26988825197074

Finally, one could profile the logpdf with respect to some of the parameters and compute local maxima through
the :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method. This
is useful both to initialize chains in an MCMC or to perform profiled likelihood inference. The result is stored in the 
:attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>` and
:attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Likelihood.Y_prof_logpdf_max>` attributes.
For instance, profiling with respect to the nuisance parameters for ``10`` values of the signal strength parameter
on a grid in the ``(-1,1)`` interval can be obtained as follows:

.. code-block:: python

    likelihood.compute_profiled_maxima(pars=[0],pars_ranges=[[-1,1,10]],spacing="grid",verbose=2)
    print(likelihood.X_prof_logpdf_max)
    print(likelihood.Y_prof_logpdf_max)

    >>> [[-1.          1.21169326  0.96592494  0.7891095   0.89669271  0.61408032
           1.21501547  0.49455945  0.71604701  0.58415075  1.07251509]
         [-0.77777778  1.01767165  0.76971591  0.59234143  0.7041785   0.42121161
           1.03431711  0.31223278  0.54606212  0.430467    0.95205154]
        ...]
    >>> [51.71637546558055 50.06649195316649 48.78240790167662 47.88015165741762
         47.37632133357318 47.28808819443674 47.63319733946957 48.429965264153346
         49.69727384417462 51.4545602580758]

The ``verbose=2`` argument allows to print a progress bar to monitor the evolution of the calculation of the maxima.
If one prefers to generate signal strength values randomly (with a flat distribution) instead that on a grid, the
argument ``spacing="random"`` can be passed.

Each of the above calls :class:`Likelihood <DNNLikelihood.Likelihood>` methods have updated the 
:attr:`Likelihood.log <DNNLikelihood.Likelihood.log>` attribute and the corresponding 
:attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>` file. The full
object can be saved at any time through

.. code-block:: python

    likelihood.save_likelihood(overwrite=True)

The ``overwrite=True`` ensure that the output files (generated when initializing the object) are updated.

Classs
^^^^^^

.. autoclass:: DNNLikelihood.Likelihood
   :undoc-members:

.. _likelihood_arguments:

Arguments
"""""""""

    .. option:: name

        Name of the :class:`Likelihood <DNNLikelihood.Likelihood>` object.
        It is used to build the :attr:`Likelihood.name <DNNLikelihood.Likelihood.name>` attribute.
         
            - **type**: ``str`` or ``None``
            - **default**: ``None``   

    .. option:: logpdf

        Callable function returning the logpdf given parameters and additional arguments, passed through the
        :option:`logpdf_args` argument.
            
            - **type**: ``callable`` or ``None``
            - **default**: ``None`` 

        - **Could accept**

            - **x_pars**
                
                Values of the parameters for which logpdf is computed.
                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``.
                    
                    - **type**: ``numpy.ndarray``
                    - **possible shapes**: ``(n_pars,)`` or ``(n_points,n_pars)``

            - **args**

                List of additional arguments required 
                by the :option:`logpdf` function and passed through the :option:`logpdf_args` input argument. 
                    
                    - **type**: ``list`` or None
                    - **shape of list**: ``[ ]``
         
        - **Could return**

            ``float`` or ``numpy.ndarray`` with shape ``(n_points,)``

    .. option:: logpdf_args   

        Additional arguments required by the 
        :option:`logpdf` function.
            
            - **type**: ``list`` or ``None``
            - **shape of list**: ``[ ]``
            - **default**: ``None`` 

    .. option:: pars_pos_poi   

        List or |numpy_link| array containing the positions in the parameters list of the
        parameters of interest.
        It is used to build the :attr:`Likelihood.pars_pos_poi <DNNLikelihood.Likelihood.pars_pos_poi>` attribute.

            - **type**: ``list`` or ```numpy.ndarray``
            - **shape**: ``(n_poi,)``
            - **default**: ``None`` 

    .. option:: pars_pos_nuis   

        List or |numpy_link| array containing the positions in the parameters list of the
        nuisance parameters.
        It is used to build the :attr:`Likelihood.pars_pos_nuis <DNNLikelihood.Likelihood.pars_pos_nuis>` attribute.

            - **type**: ``list`` or ```numpy.ndarray``
            - **shape**: ``(n_nuis,)``
            - **default**: ``None`` 

    .. option:: pars_init   

        List or |numpy_link| array containing an initial value for the parameters.
        It is used to build the :attr:`Likelihood.pars_init <DNNLikelihood.Likelihood.pars_init>` attribute.
            
            - **type**: ``list`` or ```numpy.ndarray``
            - **shape**: ``(n_pars,)``
            - **default**: ``None`` 

    .. option:: pars_labels   

        List containing the parameters names as strings.
        Parameters labels are always used as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.
        It is used to build the :attr:`Likelihood.pars_labels <DNNLikelihood.Likelihood.pars_labels>` attribute.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``
            - **default**: ``None`` 

    .. option:: pars_bounds   

        List or |numpy_link| array containing containing bounds for the parameters.
        It is used to build the :attr:`Likelihood.pars_bounds <DNNLikelihood.Likelihood.pars_bounds>` attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``(n_pars,2)``
            - **default**: ``None`` 

    .. option:: output_folder
         
        Path (either relative to the code execution folder or absolute) where output files are saved.
        It is used to set the :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>` attribute.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``

    .. option:: likelihood_input_file   

        File name (either relative to the code execution folder or absolute, with or without any of the
        .json or .pickle extensions) of a saved :class:`Likelihood <DNNLikelihood.Likelihood>` object. 
        It is used to set the 
        :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Likelihood.likelihood_input_file>` 
        attribute.

           - **type**: ``str`` or ``None``
           - **default**: ``None``

   .. option:: verbose

        Argument used to set the verbosity mode of the :meth:`Likelihood.__init__ <DNNLikelihood.Likelihood.__init__>` 
        method and the default verbosity mode of all class methods that accept a ``verbose`` argument.
        See :ref:`Verbosity mode <verbosity_mode>`.

           - **type**: ``bool``
           - **default**: ``True``

Attributes
""""""""""

    .. py:attribute:: DNNLikelihood.Likelihood.figure_files_base_path

        Absolute path to the exported figures. It includes the base figure name and is 
        automatically set from the attribute
        :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`.
        

           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Likelihood.figures_list

        List of absolute paths to the generated figures.

           - **type**: ``list`` of ``str`` 

    .. py:attribute:: DNNLikelihood.Likelihood.generic_pars_labels   

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

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_input_file   

        Attribute corresponding to the input argument :option:`likelihood_input_file`.
        Whenever this parameter is not ``None`` the :class:`Likelihood <DNNLikelihood.Likelihood>` object
        is reconstructed from input files (see the :meth:`Likelihood.__init__ <DNNLikelihood.Likelihood.__init__>`
        method for details).
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_input_json_file    

        Absolute path to the .json file containing saved :class:`Likelihood <DNNLikelihood.Likelihood>` json (see
        the :meth:`Likelihood.save_likelihood_json <DNNLikelihood.Likelihood.save_likelihood_json>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Likelihood.likelihood_input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
             
            - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_input_log_file    

        Absolute path to the .log file containing saved :class:`Likelihood <DNNLikelihood.Likelihood>` log (see
        the :meth:`Likelihood.save_likelihood_log <DNNLikelihood.Likelihood.save_likelihood_log>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Likelihood.likelihood_input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_input_pickle_file    

        Absolute path to the .pickle file containing saved :class:`Likelihood <DNNLikelihood.Likelihood>` pickle (see
        the :meth:`Likelihood.save_likelihood_pickle <DNNLikelihood.Likelihood.save_likelihood_pickle>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Likelihood.likelihood_input_file>`.
        When the latter is ``None``, the attribute is set to ``None``.
             
          - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_output_json_file

        Absolute path to the .json file where part of the :class:`Likelihood <DNNLikelihood.Likelihood>` 
        object is saved (see the :meth:`Likelihood.save_likelihood_json <DNNLikelihood.Likelihood.save_likelihood_json>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_output_log_file

        Absolute path to the .log file where the :class:`Likelihood <DNNLikelihood.Likelihood>` 
        object log is saved (see the :meth:`Likelihood.save_likelihood_log <DNNLikelihood.Likelihood.save_likelihood_log>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_output_pickle_file

        Absolute path to the .pickle file where part of the :class:`Likelihood <DNNLikelihood.Likelihood>` 
        object is saved (see the :meth:`Likelihood.save_likelihood_pickle <DNNLikelihood.Likelihood.save_likelihood_pickle>`
        method for details).
        This is automatically generated from the attribute
        :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`.
             
          - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_script_file

        Absolute path to the .py script containing the code necessary to intantiate a 
        :class:`Likelihood <DNNLikelihood.Likelihood>` object and define the corresponing parameters. 
        This file can be generated using the 
        :meth:`Likelihood.save_likelihood_script <DNNLikelihood.Likelihood.save_likelihood_script>` method
        and is used to initialize a :class:`Sampler <DNNLikelihood.Sampler>` object 
        (see :ref:`the Sampler object <sampler_object>`). This is to ensure that that Markov Chain Monte Carlo properly 
        runs in parallel (using the |multiprocessing_link| package) inside Jupyter notebooks also on the Windows OS.
        This is automatically generated from the attribute
        :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`.

            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Histfactory.log    

         Dictionary containing a log of the :class:`Histfactory <DNNLikelihood.Histfactory>` object calls. The dictionary has datetime strings as keys
         and actions as values. Actions are also dictionaries, containing details of the methods calls.
               
            - **type**: ``dict``
            - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
            - **values**: ``dict`` with the following structure:

                - *"action"* (value type: ``str``)
                   Short description of the action.
                   **possible values**: ``"computed maximum logpdf"``, ``"computed profiled maxima"``, ``"created"``, ``"loaded"``, ``"saved"``, ``"saved figure"``
                - *"pars"* (value type: ``list`` of ``int``)
                   Input argument of the :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method.
                - *"pars_ranges"* (value type: ``list`` of ``list``)
                   Input argument of the :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method.
                - *"number of maxima"* (value type: ``int``)
                   Number of maxima computed by the :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method.
                - *"file name"* (value type: ``str``)
                   File name of file involved in the action.
                - *"file path"* (value type: ``str``)
                   Path of file involved in the action.

    .. py:attribute:: DNNLikelihood.Likelihood.logpdf

        Attribute corresponding to the input argument :option:`logpdf`. When this input is ``None`` the argument
        :option:`likelihood_input_file` should be given to import the attribute from files.
        This function is used to construct the :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>` method.
            
            - **type**: ``callable``

        - **Could accept**

            - **x_pars**
                
                Values of the parameters for which logpdf is computed.
                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``.
                    
                    - **type**: ``numpy.ndarray``
                    - **possible shapes**: ``(n_pars,)`` or ``(n_points,n_pars)``

            - **args**

                List of additional arguments required 
                by the :option:`logpdf` function and passed through the :option:`logpdf_args` input argument. 
                    
                    - **type**: ``list`` or None
                    - **shape of list**: ``[ ]``
         
        - **Could return**

            ``float`` or ``numpy.ndarray`` with shape ``(n_points,)``

    .. py:attribute:: DNNLikelihood.Likelihood.logpdf_args   

        Attribute corresponding to the input argument :option:`logpdf_args`.
            
            - **type**: ``list`` or ``None``
            - **shape of list**: ``[ ]``

    .. py:attribute:: DNNLikelihood.Likelihood.name

        Name of the :class:`Likelihood <DNNLikelihood.Likelihood>` object generated from
        the :option:`name` input argument. If ``None`` is passed, then ``name`` is assigned the value 
        ``model_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_likelihood"``, 
        while if a string is passed, the ``"_likelihood"`` suffix is appended 
        (preventing duplication if it is already present).
        It is used to generate output files names.
            
            - **type**: ``str``  

    .. py:attribute:: DNNLikelihood.Histfactory.output_folder

        Absolute path corresponding to the input argument
        :option:`output_folder`. If the latter is ``None``, then 
        :attr:`output_folder <DNNLikelihood.Histfactory.output_folder>`
        is set to the code execution folder.

           - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Likelihood.pars_bounds   

        |Numpy_link| array corresponding to the input argument :option:`pars_bounds`. If the input argument is ``None``
        then bounds for all parameters are set to ``[-np.inf,np.inf]``.
        It is used to build the :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>` so that
        the logpdf evaluates to ``-np.inf`` if any of the parameter has a value outside these bounds.

            - **type**: ``numpy.ndarray``
            - **shape**: ``(n_pars,2)``

    .. py:attribute:: DNNLikelihood.Likelihood.pars_init   

        |Numpy_link| array corresponding to the input argument :option:`pars_init`.
        This initialization is used as initial value for optimizing 
        :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>` with the
        :meth:`Likelihood.compute_maximum_logpdf <DNNLikelihood.Likelihood.compute_maximum_logpdf>` and the
        :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` methods.
            
            - **type**: ``numpy.ndarray``
            - **shape**: ``(n_pars,)``

    .. py:attribute:: DNNLikelihood.Likelihood.pars_labels   

        List corresponding to the input argument :option:`pars_labels`. If the input argument is ``None`` then
        :attr:`Likelihood.pars_labels <DNNLikelihood.Likelihood.pars_labels>` is set equal to the automatically
        generated :attr:`Likelihood.generic_pars_labels <DNNLikelihood.Likelihood.generic_pars_labels>`.
        Parameters labels are always used as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``

    .. py:attribute:: DNNLikelihood.Likelihood.pars_pos_nuis   

        |Numpy_link| array corresponding to the input argument :option:`pars_pos_nuis`.

            - **type**: ``list`` or ```numpy.ndarray``
            - **shape**: ``(n_nuis,)``

    .. py:attribute:: DNNLikelihood.Likelihood.pars_pos_poi   

        |Numpy_link| array corresponding to the input argument :option:`pars_pos_poi`.

            - **type**: ``list`` or ```numpy.ndarray``
            - **shape**: ``(n_poi,)``

    .. py:attribute:: DNNLikelihood.Histfactory.verbose

        Attribute corresponding to the input argument :option:`verbose`.
        It represents the verbosity mode of the 
        :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` 
        method and the default verbosity mode of all class methods that accept a
        ``verbose`` argument.
        See :ref:`Verbosity mode <verbosity_mode>`.

            - **type**: ``bool`` or ```int``

    .. py:attribute:: DNNLikelihood.Likelihood.X_logpdf_max
            
        |Numpy_link| array containing the values of parameters at the global maximum
        of the logpdf computed with the 
        :meth:`Likelihood.compute_maximum_logpdf <DNNLikelihood.Likelihood.compute_maximum_logpdf>` method.
        The attribute is ``None`` unless the latter method
        has been called or the :class:`Likelihood <DNNLikelihood.Likelihood>` object has been imported from file 
        and already contained a value for the attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``(n_pars,)``

    .. py:attribute:: DNNLikelihood.Likelihood.X_prof_logpdf_max

        |Numpy_link| array containing the values of parameters at different local maxima of the logpdf computed
        with the :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method
        The attribute is ``None`` compute_profiled_maxima
        object has been imported from file and already contained a value for the attribute.
        This attribute is used to initialize walkers in the :class:``Sampler <DNNLikelihood.Sampler>`` object
        (see :ref:`the Likelihood object <likelihood_object>`).

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``np.array(n_points,n_pars)``

    .. py:attribute:: DNNLikelihood.Likelihood.X_prof_logpdf_max_tmp

        Same as :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>`. 
        It is assigned only when attempting to append newly 
        generated profiled maxima to an incompatible existing 
        :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>`.
        This is a temporary attribute and it is not saved by the 
        :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` method.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``np.array(n_points,n_pars)``

    .. py:attribute:: DNNLikelihood.Likelihood.Y_logpdf_max  

        Value of logpdf at its global maximum computed with the 
        :meth:`Likelihood.compute_maximum_logpdf <DNNLikelihood.Likelihood.compute_maximum_logpdf>` method.
        The attribute is ``None`` unless the latter method
        has been called or the :class:`Likelihood <DNNLikelihood.Likelihood>` object has been imported from file 
        and already contained a value for the attribute.

            - **type**: ``float`` or ``None``
                
    .. py:attribute:: DNNLikelihood.Likelihood.Y_prof_logpdf_max

        |Numpy_link| array containing the values of logpdf at different local maxima computed
        with the 
        :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` method.
        The attribute is ``None`` unless the latter method
        has been called or the :class:`Likelihood <DNNLikelihood.Likelihood>` object has been imported from file 
        and already contained a value for the attribute.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``np.array(n_points,)``

    .. py:attribute:: DNNLikelihood.Likelihood.Y_prof_logpdf_max_tmp

        Same as :attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Likelihood.Y_prof_logpdf_max>`. 
        It is assigned only when attempting to append newly 
        generated profiled maxima to an incompatible existing 
        :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>`.
        This is a temporary attribute and it is not saved by the 
        :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` method.

            - **type**: ``numpy.ndarray`` or ``None``
            - **shape**: ``np.array(n_points,)``

Methods
"""""""

    .. automethod:: DNNLikelihood.Likelihood.__init__

    .. automethod:: DNNLikelihood.Likelihood._Likelihood__check_define_name

    .. automethod:: DNNLikelihood.Likelihood._Likelihood__load_likelihood

    .. automethod:: DNNLikelihood.Likelihood._Likelihood__set_pars_labels

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood_log

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood_json

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood_pickle

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood_script

    .. automethod:: DNNLikelihood.Likelihood.logpdf_fn

    .. automethod:: DNNLikelihood.Likelihood.compute_maximum_logpdf

    .. automethod:: DNNLikelihood.Likelihood.compute_profiled_maxima

    .. automethod:: DNNLikelihood.Likelihood.plot_logpdf_par


.. |pyhf_model_logpdf_link| raw:: html
    
    <a href="https://scikit-hep.org/pyhf/_generated/pyhf.pdf.Model.html?highlight=logpdf#pyhf.pdf.Model.logpdf"  target="_blank"> pyhf.Model.logpdf</a>

.. |numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> numpy</a>

.. |Numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> Numpy</a>

.. |multiprocessing_link| raw:: html
    
    <a href="https://docs.python.org/3/library/multiprocessing.html"  target="_blank"> multiprocessing</a>
