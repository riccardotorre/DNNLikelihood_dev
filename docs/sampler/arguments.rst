.. _sampler_arguments:

Arguments
"""""""""

.. currentmodule:: Sampler

.. argument:: new_sampler

    If ``True`` a new :class:`Sampler <DNNLikelihood.Sampler>` object is created from input arguments, while if 
    ``False`` the object is reconstructed from saved files (see the :meth:`__init__ <DNNLikelihood.Sampler.__init__>`
    method).
    It is used to build the :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` attribute.

        - **type**: ``bool``
        - **default**: ``None``

.. argument:: likelihood_script_file

    File name (either relative to the code execution folder or absolute) of a ``likelihood_script_file`` 
    genetated by the :meth:`Lik.save_script <DNNLikelihood.Lik.save_script>` method. 
    It is used to build the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute.

        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: likelihood

    A :py:class:`Lik <DNNLikelihood.Lik>` object. 
    It is used to initialize the :class:`Sampler <DNNLikelihood.Sampler>` object directly from the 
    :py:class:`Lik <DNNLikelihood.Lik>` one. This argument is not saved into an attribute:
    the :py:class:`Lik <DNNLikelihood.Lik>` object is copied, used to save a likelihood script file and to
    set the :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute if the latter 
    is not passed through the argument :argument:`likelihood_script_file` (see the :meth:`__init__ <DNNLikelihood.Sampler.__init__>`
    method).

        - **type**: :py:class:`Lik <DNNLikelihood.Lik>` object or ``None``
        - **default**: ``None``

.. argument:: nsteps_required

    Final number of MCMC steps. When the object is initialized with the :argument:`new_sampler` argument set to ``False``
    then, if :argument:`nsteps_required` is larger than the number of steps available in the backend, it is saved in the 
    :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>`, otherwise the latter is set equal to the number of steps available
    in the backend.

        - **type**: ``int`` or ``None``
        - **default**: ``None``

.. argument:: moves_str

    String containing an |emcee_moves_link| object. If ``None`` is passed, the default ``emcee.moves.StretchMove()`` is passed.
    It is used to set the :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` attribute.
        
        - **type**: ``str`` or ``None``
        - **default**: ``None`` 
        - **example**: "[(emcee.moves.StretchMove(0.7), 0.2), (emcee.moves.GaussianMove(0.1, mode='random',factor=None),0.8)]"
            
            This gives a move that is 20% ``StretchMove`` with parameter 0.7 and 80% ``GaussianMove`` with covariance 0.1 and 
            mode ``'random'`` (i.e. updating a single random parameter at each step). 
            See the |emcee_moves_link| documentation for more details.

.. argument:: parallel_CPU

    If ``True`` the MCMC is run in 
    parallel on the available CPU cores, otherwise only a single core is used.
    It is used to set the :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` attribute.

        - **type**: ``bool``
        - **default**: ``True``    

.. argument:: vectorize

    If ``True``, the method :meth:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` is expected to accept a list of
    points and to return a list of logpdf values, and :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` 
    is automatically set to ``False``. See the |emcee_ensemble_sampler_link| documentation for more details.
    It is used to set the :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` attribute.

.. argument:: input_file   

    File name (either relative to the code execution folder or absolute, with or without extension) 
    of a saved :class:`Sampler <DNNLikelihood.Sampler>` object. 
    It is used to set the :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>` 
    attribute.

       - **type**: ``str`` or ``None``
       - **default**: ``None``

.. argument:: verbose

    Argument used to set the verbosity mode of the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` 
    method and the default verbosity mode of all class methods that accept a ``verbose`` argument.
    See :ref:`Verbosity mode <verbosity_mode>`.

       - **type**: ``bool``
       - **default**: ``True``

.. include:: ../external_links.rst