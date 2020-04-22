.. _sampler_object:

The Sampler object
------------------

Summary
^^^^^^^

The :class:`Sampler <DNNLikelihood.Sampler>` class is an API to the |emcee_link| Python package that can be used to sample
:class:`Likelihood <DNNLikelihood.Likelihood>` objects (more precisely the corresponding logpdf function) and export data
as a :class:`Data <DNNLikelihood.Data>` objects. The :class:`Sampler <DNNLikelihood.Sampler>` object offers several methods to
perform MCMC sampling, analyze convergence, and produce different kind of plots. 
The API uses |emcee_link| to perform the sampling and manage the backend, which ensures that samples are safely stored to file 
at any time of the sampling process. See also :ref:`the Likelihood object <likelihood_object>` and 
:ref:`the Data object <data_object>`, which are respectively used to initialize the 
:class:`Sampler <DNNLikelihood.Sampler>` class and to export the :class:`Data <DNNLikelihood.Data>` object.

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Sampler <DNNLikelihood.Sampler>` class. Refer to the 
full class documentation for more details. All examples will be referred to the toy likelihood discussed in 
:ref:`the Likelihood object Usage <likelihood_usage>` section of the documentation.

The :class:`Sampler <DNNLikelihood.Sampler>` class has been thought to be as flexible as possible in terms of initialization
and input parameters. Indeed the object can be initialized by passing different combinations of arguments and the 
:meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method always extracts (or creates if it does not exist) the file
corresponding to the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute.
The likelihood related parameters and the logpdf are always set by importing this file as a module, which instantiates a
:class:`Likelihood <DNNLikelihood.Likelihood>` object, and by extracting the parameters from it.

The procedure of passing the :class:`Likelihood <DNNLikelihood.Likelihood>` object through a script file instead of 
passing it directly may seem redundant. However, this is needed to ensure that MCMC properly runs in parallel also on
Windows machines, where the |multiprocessing_link| package may have unexpected behavior.

Let us start by creating a new :class:`Sampler <DNNLikelihood.Sampler>` object, which requires the input argument
``new_sampler=True``. We can proceed in three different ways, by giving as input one of the three arguments
:option:`likelihood_script_file`, :option:`likelihood`, and, in case we have previously saved a :class:`Sampler <DNNLikelihood.Sampler>`
and we are interested in creating a new sampler with the same parameters, the argument :option:`sampler_input_file`. Each of these
arguments is used to determine the attribute :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>`
which is then used to import the likelihood related attributes. If more than one of these inputs is give, only one is used,
with priority in the order :option:`likelihood_script_file`, :option:`likelihood`, and :option:`sampler_input_file`. 
Let us see the three options in code:

    - Initialization from :option:`likelihood_script_file`

        We assume that a likelihood script file has been already created (see then :ref:`the Likelihood object Usage <likelihood_usage>` 
        section of the documentation). The :class:`Sampler <DNNLikelihood.Sampler>` can then be initialized as

        .. code-block:: python

            import DNNLikelihood

            sampler = DNNLikelihood.Sampler(new_sampler=True,
                                             likelihood_script_file=<my_output_folder>/toy_likelihood_script,
                                             nsteps=50000)
                                            
        This initialize the object with a required number of steps ``nsteps=50000`` (this could then be changed to run for more steps). We have not
        passed a ``move_str`` input so that the default |emcee_link| move ``emcee.moves.StretchMove()`` will be set (see the 
        :attr:`Sampler.move_str <DNNLikelihood.Sampler.move_str>` attribute documentation for more details). Moreover, since 
        ``parallel_CPU`` has not been specified, the attribute :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` 
        is automatically set to ``True`` and the sampler will be running in parallel mode.

        When the object is created, it is automatically saved and three files are created:

            - <my_output_folder>/toy_sampler.json
            - <my_output_folder>/toy_sampler.log 
            - <my_output_folder>/toy_sampler_backend.h5

    - Initialization from :option:`likelihood`

        The exact same object could be initialized directly from a :class:`Likelihood <DNNLikelihood.Likelihood>` one. In this case the
        :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method takes care of generating the script file:

        .. code-block:: python

            import DNNLikelihood

            likelihood = DNNLikelihood.Likelihood(likelihood_input_file="<my_output_folder>/toy_likelihood")

            sampler = DNNLikelihood.Sampler(new_sampler=True,
                                             likelihood=likelihood,
                                             nsteps=50000)

    - Initialization from :option:`sampler_input_file`

        Finally, in the case the object has been initialized in the past, and the output files are available, a new object can be initialized
        with the same parameters by passing the :option:`sampler_input_file` input. In this case, again, the 
        :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method uses the argument to determine or create the
        :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` and then proceeds in the usual way.
        The following code produces an object identical to the previous ones:

        .. code-block:: python

            import DNNLikelihood

            sampler = DNNLikelihood.Sampler(new_sampler=True,
                                             nsteps=50000,
                                             sampler_input_file=<my_output_folder>/toy_sampler)


A previously saved :class:`Sampler <DNNLikelihood.Sampler>` could be imported by using the ``new_sampler=False`` input. 
Again, one could provide one of the three arguments :option:`likelihood_script_file`, :option:`likelihood`, 
and :option:`sampler_input_file`. This time, each of these
arguments is used to determine the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` and
:attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>` attributes, used to build the object. If more than 
one of these inputs is give, only one is used, with priority in the order :option:`sampler_input_file`, 
:option:`likelihood_script_file`, and :option:`likelihood`. Obviously, independently on the passed input arguments, 
the files corresponding to :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>` should exist (i.e. a 
saved :class:`Sampler <DNNLikelihood.Sampler>` object should be available).

The following three code options produce the same result, importing the object saved before:

    - Import from :option:`sampler_input_file`

        .. code-block:: python

            import DNNLikelihood

            sampler = DNNLikelihood.Sampler(new_sampler=False,
                                             sampler_input_file=<my_output_folder>/toy_sampler)

    - Import from :option:`likelihood_script_file`

        .. code-block:: python

            import DNNLikelihood

            sampler = DNNLikelihood.Sampler(new_sampler=False,
                                             likelihood_script_file=<my_output_folder>/toy_likelihood_script)

    - Import from :option:`likelihood`

        The exact same object could be initialized directly from a :class:`Likelihood <DNNLikelihood.Likelihood>` one. In this case the
        :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method takes care of generating the script file:

        .. code-block:: python

            import DNNLikelihood

            likelihood = DNNLikelihood.Likelihood(likelihood_input_file="<my_output_folder>/toy_likelihood")

            sampler = DNNLikelihood.Sampler(new_sampler=False,
                                             likelihood=likelihood)

When the object is imported, the :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute is updated, as well as the corresponding file
<my_output_folder>/toy_sampler.log. If a new ``nsteps`` input, larger than the number of steps available in the existing backedn is passed,
then this is saved in the :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`, which is otherwise set to the number of available steps.
Also the ``moves_str`` input can be passed to update (change) the move of the sampler.

Now that we have discussed how to create and import the object, let us see how to use it. The first thing we want to do is to produce 
a sampling, which is done as follows:

.. code-block:: python

    sampler.run_sampler(verbose=2)

This runs the sampler and, through the ``verbose=2`` argument, shows a progress bar, together with the remaining time. Since the object
has attribute :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` set to ``True`` the sampling is run in ``n`` parallel
processes, with ``n`` equal to the number of physical (not logical) cores. While the sampler runs, all produced sampler are saved in related
time in the backend file corresponding to the :attr:`Sampler.sampler_output_backend_file <DNNLikelihood.Sampler.sampler_output_backend_file>`
attribute. At the end of the sampling also the log attribute and file are updated. 
The :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` and :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` correspond to the
|emcee_link| objects |emcee_ensemble_sampler_link| and |emcee_backend_link| and we refer to the |emcee_link| documentation for their use.
Nevertheless, the :class:`Sampler <DNNLikelihood.Sampler>` object, includes several methods that allow to perform convergence studies,
produce plots, and extract samples without the need of directly digging into the |emcee_link| objects and documentation.

As a first step we can compute the the Gelman and Rubin metrics :cite:`Gelman:1992zz` for some parameters. For instance let us
compute the metrics for parameters ``0`` (poi) and ``5`` (nuis) after ``[500,1000,5000,10000,50000]`` steps:

.. code-block:: python

    sampler.gelman_rubin(pars=[0,5],nsteps=[500,1000,5000,10000,50000])

    >>> array([[0.00000000e+00, 5.00000000e+02, 1.09139684e+00, 2.47454986e-01, 2.08260547e-01],
               [0.00000000e+00, 1.00000000e+03, 1.05672208e+00, 1.84330055e-01, 1.65349725e-01],
               [0.00000000e+00, 5.00000000e+03, 1.01555635e+00, 1.26862627e-01, 1.23053024e-01],
               [0.00000000e+00, 1.00000000e+04, 1.00790794e+00, 1.20567487e-01, 1.18705125e-01],
               [0.00000000e+00, 5.00000000e+04, 1.00124959e+00, 1.15572381e-01, 1.15290829e-01],
               [5.00000000e+00, 5.00000000e+02, 1.13686644e+00, 3.49600912e-01, 2.71401392e-01],
               [5.00000000e+00, 1.00000000e+03, 1.06837042e+00, 3.03575872e-01, 2.66503447e-01],
               [5.00000000e+00, 5.00000000e+03, 1.01267127e+00, 2.61092073e-01, 2.54676062e-01],
               [5.00000000e+00, 1.00000000e+04, 1.00587726e+00, 2.62245501e-01, 2.59235550e-01],
               [5.00000000e+00, 5.00000000e+04, 1.00122323e+00, 2.56878972e-01, 2.56262912e-01]])

This returns an array where in the first column there is the parameter, in the second the number of steps and in the third to fifth
the values of the metrics. See the documentation of the :meth:`Sampler.gelman_rubin <DNNLikelihood.Sampler.gelman_rubin>` method for
details.

We could also produce a plot of these metrics using the :meth:`Sampler.plot_gelman_rubin <DNNLikelihood.Sampler.plot_gelman_rubin>`
method

.. code-block:: python

    sampler.plot_gelman_rubin(pars=[0,5], npoints=10)

.. image:: figs/toy_sampler_figure_GR_Rc_0.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_GR_sqrtVhat_0.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_GR_sqrtW_0.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_GR_Rc_5.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_GR_sqrtVhat_5.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_GR_sqrtW_5.png
    :class: with-shadow
    :scale: 36

Convergence could also be monitored through the autocorrelation time, which could be plot using the 
:meth:`Sampler.plot_autocorr <DNNLikelihood.Sampler.plot_autocorr>`. See the method documentation for details on the various
algorithms and options. For instance, using all available methods we can monitor parameters ``0`` and ``5`` by

.. code-block:: python

    sampler.plot_autocorr(pars=[0,5])

.. image:: figs/toy_sampler_figure_autocorr_0.png
    :class: with-shadow
    :scale: 54

.. image:: figs/toy_sampler_figure_autocorr_5.png
    :class: with-shadow
    :scale: 54

The 1D distribution of the parameters can be obtained through the :meth:`Sampler.plot_dist <DNNLikelihood.Sampler.plot_dist>` 
method by

.. code-block:: python

    sampler.plot_dist(pars=[0,5])

.. image:: figs/toy_sampler_figure_distr_0.png
    :class: with-shadow
    :scale: 54

.. image:: figs/toy_sampler_figure_distr_5.png
    :class: with-shadow
    :scale: 54

The last series of plots that can be automatically obtained are the parameters and logpdf evolution for a given number of walkers
with the number of steps.
They are plot by the :meth:`Sampler.plot_chains <DNNLikelihood.Sampler.plot_chains>` and 
:meth:`Sampler.plot_chains_logpdf <DNNLikelihood.Sampler.plot_chains_logpdf>` methods respectively. For instance we could plot
``30`` randomly selected walkers as follows:

.. code-block:: python

    sampler.plot_chains(pars=[0,5],n_chains=30)
    sampler.plot_chains_logprob(n_chains=30)

.. image:: figs/toy_sampler_figure_chains_0.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_chains_5.png
    :class: with-shadow
    :scale: 36

.. image:: figs/toy_sampler_figure_chains_logpdf.png
    :class: with-shadow
    :scale: 36

The full list of figures we produced can be extracted from the :attr:`Sampler.figures_list <DNNLikelihood.Sampler.figures_list>`

.. code-block:: python

    sampler.figures_list()

    >>> ['<my_output_folder>/toy_sampler_figure_GR_Rc_0.pdf',
         '<my_output_folder>/toy_sampler_figure_GR_sqrtVhat_0.pdf',
         '<my_output_folder>/toy_sampler_figure_GR_sqrtW_0.pdf',
         '<my_output_folder>/toy_sampler_figure_GR_Rc_5.pdf',
         '<my_output_folder>/toy_sampler_figure_GR_sqrtVhat_5.pdf',
         '<my_output_folder>/toy_sampler_figure_GR_sqrtW_5.pdf',
         '<my_output_folder>/toy_sampler_figure_distr_0.pdf',
         '<my_output_folder>/toy_sampler_figure_distr_5.pdf',
         '<my_output_folder>/toy_sampler_figure_autocorr_0.pdf',
         '<my_output_folder>/toy_sampler_figure_autocorr_5.pdf',
         '<my_output_folder>/toy_sampler_figure_chains_0.pdf',
         '<my_output_folder>/toy_sampler_figure_chains_5.pdf',
         '<my_output_folder>/toy_sampler_figure_chains_logpdf.pdf']

The :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>` can be increased at any time, and the method 
:meth:`Sampler.run_sampler <DNNLikelihood.Sampler.run_sampler>` can be called again to run the steps missing to reach 
:attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`. For instance, to run for another 20K steps one does

.. code-block:: python

    sampler.nsteps = 70000
    sampler.run_sampler()

Once a satisfactory sampling has been obtained, we can generate a :class:`Data <DNNLikelihood.Data>` object storing the
desired dataset (see :ref:`the Data object <data_object>`). The dataset is obtained from the sampling by taking samples after a possible ``burnin`` number of steps, and
to avoid large correlation between samples, by taking a step every ``thin`` (a fully unbiased sampling, that is a faithful
random number generator would need ``thin`` equal or slightly larger than the autocorrelation time). This is done with the
:meth:`Sampler.get_data_object <DNNLikelihood.Sampler.get_data_object>` as follows:

.. code-block:: python

    data = sampler.get_data_object(nsamples=100000, burnin=5000, thin=10, dtype="float64", test_fraction=0)

This generates a :class:`Data <DNNLikelihood.Data>` object with ``100000`` sampler obtained by discarding from the sampling the first
``5000`` steps and by taking one every ``10`` steps after (until the required ``nsamples`` is reached). If the number ``nsamples`` is
larger than the one available consistently with the ``burnin`` and ``thin`` options, then all available samples are taken (and a warning
message is printed). The user may choose the data type of the exported samples ``dtype`` (default is ``"float64"``), which is useful
to format data with the precision needed to train the DNNLikelihood. Finally, the user may already choose a ``test_fraction``, so that
data are divided, inside the :class:`Data <DNNLikelihood.Data>` object, into two samples, one used to feedtrain and validation data 
and the other used as test data. If ``test_fraction`` is left to the default value ``0``, it could simply be updated afterwards from
the :class:`Data <DNNLikelihood.Data>` object itself.

As a final step we can save the state of the :class:`Sampler <DNNLikelihood.Sampler>` object by using the 
:meth:`Sampler.save_sampler <DNNLikelihood.Sampler.save_sampler>` method.

.. code-block:: python

    sampler.save_sampler(overwrite=True)

The user should remember that the default value of the ``overwrite`` argument for saving functions is ``False``. Therefore, in order not
to produce new files, ``overwrite=True`` should be explicitly specified when making intermediate or final saving of the objects.


Class
^^^^^

.. autoclass:: DNNLikelihood.Sampler
   :undoc-members:

.. _sampler_arguments:

Arguments
"""""""""

    .. option:: new_sampler

        If ``True`` a new :class:`Sampler <DNNLikelihood.Sampler>` object is created from input arguments, while if 
        ``False`` the object is reconstructed from saved files (see the :meth:`__init__ <DNNLikelihood.Sampler.__init__>`
        method).
        It is used to build the :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` attribute.

            - **type**: ``bool``
            - **default**: ``None``

    .. option:: likelihood_script_file

        File name (either relative to the code execution folder or absolute) of a ``likelihood_script_file`` 
        genetated by the :meth:`Likelihood.save_likelihood_script <DNNLikelihood.Likelihood.save_likelihood_script>` method. 
        It is used to build the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute.

            - **type**: ``str`` or ``None``
            - **default**: ``None``

    .. option:: likelihood

        A :py:class:`Likelihood <DNNLikelihood.Likelihood>` object. 
        It is used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` object directly from the 
        :py:class:`Likelihood <DNNLikelihood.Likelihood>` one. This argument is not saved into an attribute:
        the :py:class:`Likelihood <DNNLikelihood.Likelihood>` object is copied, used to save a likelihood script file and to
        set the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute if the latter 
        is not passed through the argument :option:`likelihood_script_file` (see the :meth:`__init__ <DNNLikelihood.Sampler.__init__>`
        method).

            - **type**: :py:class:`Likelihood <DNNLikelihood.Likelihood>` object or ``None``
            - **default**: ``None``

    .. option:: nsteps

        Final number of MCMC steps. When the object is initialized with the :option:`new_sampler` argument set to ``False``
        then, if :option:`nsteps` is larger than the number of steps available in the backend, it is saved in the 
        :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`, otherwise the latter is set equal to the number of steps available
        in the backend.

            - **type**: ``int`` or ``None``
            - **default**: ``None`` 

    .. option:: moves_str

        String containing an |emcee_moves_link| object. If ``None`` is passed, the default ``emcee.moves.StretchMove()`` is passed.
        It is used to set the :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` attribute.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 
            - **example**: "[(emcee.moves.StretchMove(0.7), 0.2), (emcee.moves.GaussianMove(0.1, mode='random',factor=None),0.8)]"
                
                This gives a move that is 20% ``StretchMove`` with parameter 0.7 and 80% ``GaussianMove`` with covariance 0.1 and 
                mode ``'random'`` (i.e. updating a single random parameter at each step). 
                See the |emcee_moves_link| documentation for more details.

    .. option:: parallel_CPU

        If ``True`` the MCMC is run in 
        parallel on the available CPU cores, otherwise only a single core is used.
        It is used to set the :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` attribute.

            - **type**: ``bool``
            - **default**: ``True``    

    .. option:: vectorize

        If ``True``, the method :meth:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` is expected to accept a list of
        points and to return a list of logpdf values, and :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` 
        is automatically set to ``False``. See the |emcee_ensemble_sampler_link| documentation for more details.
        It is used to set the :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` attribute.

    .. option:: sampler_input_file   

        File name (either relative to the code execution folder or absolute, with or without any of the
        .json extensions) of a saved :class:`Sampler <DNNLikelihood.Sampler>` object. 
        It is used to set the :attr:`Likelihood.sampler_input_file <DNNLikelihood.Likelihood.sampler_input_file>` 
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

    .. py:attribute:: DNNLikelihood.Sampler.backend

        An ``emcee.Backends`` object (see the |emcee_backend_link| documentation for details).
        It is initialized (either from scratch or through an existing file) by the 
        :meth:`Sampler.__init_backend <DNNLikelihood.Sampler._Sampler__init_backend>` method.

            - **type**: ``emcee.Backends`` object

    .. py:attribute:: DNNLikelihood.Sampler.sampler_output_backend_file

        Name (with absolute path) of the |emcee_link| HDF5 backend file. 
        It is set to :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` ``+ "_backend.h5"``.
        See the |emcee_backend_link| documentation for details about the ``emcee.Backends`` object.

            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Sampler.figure_files_base_path

        Absolute path to the exported figures. It includes the base figure name and is 
        automatically generated from the
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and 
        :attr:`Sampler.name <DNNLikelihood.Sampler.name>` attributes.

           - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Sampler.figures_list

        List of absolute paths to the generated figures.

           - **type**: ``list`` of ``str`` 

    .. py:attribute:: DNNLikelihood.Sampler.generic_pars_labels   

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

    .. py:attribute:: DNNLikelihood.Sampler.likelihood_script_file

        Absolute path to the .py script containing the code necessary to intantiate a 
        :class:`Likelihood <DNNLikelihood.Likelihood>` object and define the corresponing parameters.  
        The :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method
        loads it as a module, which instantiate a :class:`Likelihood <DNNLikelihood.Likelihood>` object
        and defines parameters and logpdf.
        See the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method for details on how the attribute
        is set.
            
            - **type**: ``str`` or ``None``
            - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Sampler.log    

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
                - *"updated file name"*
                   File name of file involved in the action.
                - *"updated file path"* (value type: ``str``)
                   Path of file involved in the action.

        
            
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

    .. py:attribute:: DNNLikelihood.Sampler.logpdf:

        Callable function that could accept ``x_pars`` either as a single point in parameter space corresponding 
        to an array with shape ``(n_pars,)`` or as a list of points corresponding to an array with 
        shape ``(n_points,n_pars)`` and that returns either a ``float`` or a list of computed logpdf values,
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
                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``.
                    
                    - **type**: ``numpy.ndarray``
                    - **possible shapes**: ``(n_pars,)`` or ``(n_points,n_pars)``

            - **args**

                List of additional arguments required 
                by :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` and passed through the 
                :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>` attribute. 
                    
                    - **type**: ``list`` or None
                    - **shape of list**: ``[ ]``
        
        - **Could return**
        
            Value or array of values 
            of the logpdf.
        
                - **type**: ``float`` or ``numpy.ndarray``
                - **shape for numpy.ndarray**: ``(n_points,)``

    .. py:attribute:: DNNLikelihood.Sampler.logpdf_args   

        Attribute corresponding to the input argument :option:`logpdf_args`.
            
            - **type**: ``list`` or ``None``
            - **shape of list**: ``[ ]``

    .. py:attribute:: DNNLikelihood.Sampler.moves

        An ``emcee.moves`` object generated by evaluating the 
        :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` attribute.
        See the |emcee_moves_link| documentation for details.

            - **type**: ``emcee.moves`` object

    .. py:attribute:: DNNLikelihood.Sampler.moves_str

        String representing an instance to an ``emcee.moves`` object. If ``None``
        is passed, the default ``emcee.moves.StretchMove()`` is passed.
        See the |emcee_moves_link| documentation for details.

            - **type**: ``str`` or ``None``
            - **example**: ``"[(emcee.moves.StretchMove(0.7), 0.2), (emcee.moves.GaussianMove(0.1, mode='random',factor=None),0.8)]"``
                
                This gives a move that is 20% StretchMove with parameter 0.7 and 80% GaussianMove with covariance 0.1 and mode "random" (i.e.
                updating a random single parameter at each step).

    .. py:attribute:: DNNLikelihood.Sampler.name   

        Name of the :class:`Sampler <DNNLikelihood.Sampler>` object. It is used to generate 
        output files. It is automatically generated from the corresponding attribute of the
        :class:`Likelihood <DNNLikelihood.Likelihood>` object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` 
        by the :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method
        by replacing the suffix "_likelihood" with the suffix "_sampler"
        
            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Sampler.ndims

        Number of dimensions of the input vector (i.e. number of 
        parameters entering in the logpdf). It is automatically set to the length of
        the first point in :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`.

            - **type**: ``int``

    .. py:attribute:: DNNLikelihood.Sampler.new_sampler

        Attribute corresponding to the input argument :option:`new_sampler`.
        If it is ``True`` a new :class:`Sampler <DNNLikelihood.Sampler>` object, corresponding to a new 
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` is generated. 
        If it is ``False`` the :class:`Sampler <DNNLikelihood.Sampler>` object
        is loaded from saved files, or, if a backend file is not found ``new_sampler`` is automatically set to ``True``
        and a new one is created.
            
            - **type**: ``bool``

    .. py:attribute:: DNNLikelihood.Sampler.nsteps

        Attribute corresponding to the input argument :option:`nsteps` and representing the
        number of MCMC steps to run. When an existing :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` is loaded 
        if the value of the input argument :option:`nsteps` is smaller than the number of steps available, then
        :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>` is automatically updated to the available steps.
        The attribute always represents the final number of steps, so if
        the number of existing steps is not zero, the sampling will only run until it reaches
        :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`.
            
            - **type**: ``int``

    .. py:attribute:: DNNLikelihood.Sampler.nwalkers

        Number of walkers (equivalent of chains 
        for Ensamble Sampler MCMC). It is automatically set to the length of
        :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>` vector.
            
            - **type**: ``int``

    .. py:attribute:: DNNLikelihood.Sampler.output_folder

        Absolute path to the folder where all output files are saved.
        It is automatically set to the corresponding attribute of the :class:`Likelihood <DNNLikelihood.Likelihood>
        object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
        :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

           - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Sampler.parallel_CPU

        Attribute corresponding to the input argument :option:`parallel_CPU`.
        If ``True`` the MCMC is run in parallel on the available CPU cores, otherwise only a single core is used.

            - **type**: ``bool``

    .. py:attribute:: DNNLikelihood.Sampler.pars_init_vec

        Array of points with parameters initialization for each 
        walker. 
            
            - **type**: ``numpy.ndarray``
            - **shape**: ``(nwalkers,n_pars)``
 
     .. py:attribute:: DNNLikelihood.Sampler.pars_labels   

        List containing parameters names as strings.
        It is automatically set to the corresponding attribute of the :class:`Likelihood <DNNLikelihood.Likelihood>
        object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
        :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.
        Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
        and can contain latex expressions that are properly compiled when making plots.

            - **type**: ``list``
            - **shape**: ``[ ]``
            - **length**: ``n_pars``

    .. py:attribute:: DNNLikelihood.Sampler.pars_pos_nuis   

        |Numpy_link| array containing the positions in the parameters list of the nuisance parameters.
        It is automatically set to the corresponding attribute of the :class:`Likelihood <DNNLikelihood.Likelihood>
        object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
        :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

            - **type**: ``numpy.ndarray``
            - **shape**: ``(n_nuis,)``

    .. py:attribute:: DNNLikelihood.Sampler.pars_pos_poi   

        |Numpy_link| array containing the positions in the parameters list of the parameters of interest.
        It is automatically set to the corresponding attribute of the :class:`Likelihood <DNNLikelihood.Likelihood>
        object used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` by the
        :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` method.

            - **type**: ``numpy.ndarray``
            - **shape**: ``(n_poi,)``

    .. py:attribute:: DNNLikelihood.Sampler.sampler

        An ``emcee.EnsembleSampler`` object (see the |emcee_ensemble_sampler_link| documentation for details).
        It is initialized by the :meth:`Sampler.__init_sampler <DNNLikelihood.Sampler._Sampler__init_sampler>
        method.

            - **type**: ``emcee.EnsembleSampler`` object

    .. py:attribute:: DNNLikelihood.Sampler.sampler_input_file   

        Attribute corresponding to the input argument :option:`sampler_input_file`.
        Whenever the :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` attribute is ``False``,
        it is used to reconstructed the object from input files 
        (see the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>`
        method for details).
              
           - **type**: ``str`` or ``None``

    .. py:attribute:: DNNLikelihood.Sampler.sampler_output_json_file
         
        Absolute path to the .json file where part of the :class:`Sampler <DNNLikelihood.Sampler>` 
        object is saved (see the :meth:`Sampler.save_sampler_json <DNNLikelihood.Sampler.save_sampler_json>`
        method for details).
        This is automatically generated from the
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and 
        :attr:`Sampler.name <DNNLikelihood.Sampler.name>` attributes.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Sampler.sampler_output_log_file
         
        Absolute path to the .log file where the :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute
        is saved (see the :meth:`Sampler.save_sampler_log <DNNLikelihood.Sampler.save_sampler_log>`
        method for details).
        This is automatically generated from the
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and 
        :attr:`Sampler.name <DNNLikelihood.Sampler.name>` attributes.
              
           - **type**: ``str`` 

    .. py:attribute:: DNNLikelihood.Sampler.vectorize

        If ``True``, the function :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` is expected to accept a list of
        points and to return a list of logpdf values. When it is set to ``True`` the
        :meth:`Sampler.__check_vectorize <DNNLikelihood.Sampler._Sampler__check_vectorize>` checks the consistency by
        calling :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` on an array of points and, in case it fails,
        it sets :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` to ``False``.
        When it is ``True``, the :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>`
        attribute is automatically set to ``False``. 
        See the |emcee_ensemble_sampler_link| documentation for more details.

           - **type**: ``bool`` 

    .. py:attribute:: DNNLikelihood.Sampler.verbose

        Attribute corresponding to the input argument :option:`verbose`.
        It represents the verbosity mode of the 
        :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` 
        method and the default verbosity mode of all class methods that accept a
        ``verbose`` argument.
        See :ref:`Verbosity mode <verbosity_mode>`.

            - **type**: ``bool`` or ``int``

Methods
"""""""

    .. automethod:: DNNLikelihood.Sampler.__init__

    .. automethod:: DNNLikelihood.Sampler._Sampler__get_likelihood_script_file_from_sampler_input_file

    .. automethod:: DNNLikelihood.Sampler._Sampler__get_likelihood_script_file_from_likelihood

    .. automethod:: DNNLikelihood.Sampler._Sampler__get_sampler_input_file_from_likelihood_script_file

    .. automethod:: DNNLikelihood.Sampler._Sampler__get_sampler_input_file_from_likelihood

    .. automethod:: DNNLikelihood.Sampler._Sampler__init_likelihood

    .. automethod:: DNNLikelihood.Sampler._Sampler__load_sampler
    
    .. automethod:: DNNLikelihood.Sampler._Sampler__init_backend

    .. automethod:: DNNLikelihood.Sampler._Sampler__init_sampler
    
    .. automethod:: DNNLikelihood.Sampler._Sampler__check_vectorize

    .. automethod:: DNNLikelihood.Sampler._Sampler__check_params_backend

    .. automethod:: DNNLikelihood.Sampler._Sampler__set_steps_to_run

    .. automethod:: DNNLikelihood.Sampler._Sampler__set_pars_labels

    .. automethod:: DNNLikelihood.Sampler.run_sampler

    .. automethod:: DNNLikelihood.Sampler.save_sampler_log

    .. automethod:: DNNLikelihood.Sampler.save_sampler_json

    .. automethod:: DNNLikelihood.Sampler.save_sampler

    .. automethod:: DNNLikelihood.Sampler.get_data_object

    .. automethod:: DNNLikelihood.Sampler.autocorr_func_1d

    .. automethod:: DNNLikelihood.Sampler.auto_window

    .. automethod:: DNNLikelihood.Sampler.autocorr_gw2010

    .. automethod:: DNNLikelihood.Sampler.autocorr_new

    .. automethod:: DNNLikelihood.Sampler.autocorr_ml

    .. automethod:: DNNLikelihood.Sampler.gelman_rubin

    .. automethod:: DNNLikelihood.Sampler.plot_gelman_rubin

    .. automethod:: DNNLikelihood.Sampler.plot_dist

    .. automethod:: DNNLikelihood.Sampler.plot_autocorr

    .. automethod:: DNNLikelihood.Sampler.plot_chains

    .. automethod:: DNNLikelihood.Sampler.plot_chains_logpdf

    .. automethod:: DNNLikelihood.Sampler.set_verbosity

References
^^^^^^^^^^

.. bibliography:: bib/sampler.bib
    :all:

.. |emcee_moves_link| raw:: html
    
    <a href="https://emcee.readthedocs.io/en/stable/user/moves/"  target="_blank"> emcee.moves</a>

.. |emcee_ensemble_sampler_link| raw:: html
    
    <a href="https://emcee.readthedocs.io/en/stable/user/sampler/#emcee.EnsembleSampler"  target="_blank"> emcee.EnsembleSampler</a>

.. |emcee_backend_link| raw:: html
    
    <a href="https://emcee.readthedocs.io/en/stable/user/backends/"  target="_blank"> emcee.Backends</a>

.. |numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> numpy</a>

.. |Numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> Numpy</a>