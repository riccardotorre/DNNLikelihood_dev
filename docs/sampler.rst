The Sampler object
----------------------

Summary
^^^^^^^

Bla bla bla

Usage
^^^^^

Class
^^^^^

.. autoclass:: DNNLikelihood.Sampler
   :undoc-members:

Arguments
"""""""""

    .. py:attribute:: DNNLikelihood.Sampler.name

            Sampler name. If ``None`` is passed the name is generated as 
            ``"sampler_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")``, while if a string is passed, the 
            ``"_sampler"`` suffix is appended (preventing duplication if it is already present).
                - **type**: ``str`` or ``None``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Sampler.logpdf_input_file

            Bla bla
                - **type**: ``str`` or ``None``
                - **default**: ``None``
                - **type**: ``int`` or ``None```
                - **default**: ``None`` 

    .. py:method:: DNNLikelihood.Samplerlogpdf(x_pars,*args)   

            Function that takes parameters values ``x_pars`` and additional 
            arguments ``args``, passed through the ``logpdf_args`` argument, and returns a ``float``
            representing the logpdf value.
            As a class argument it should be passed as a callable function without arguments.
                - **type**: ``callable`` or ``None``
                - **default**: ``None`` 

            - **Arguments**

                - **x_par**

                    Array containing the parameters values for which the lofpdf
                    is computed.
                        - **type**: ``numpy.ndarray``
                        - **shape**: ``(n_pars,)``

                - **args**

                    List containing additional inputs needed by the logpdf function. For instance when exporting a
                    ``Likelihood`` object from the in the case of :ref:``Histfactory object <histfactory_class>``,
                    args is set to ``args = [Histfactory.likelihood_dict[n]["obs_data"]]`` where *n* corresponds to
                    the selected likelihood in ``Histfactory.likelihood_dict``.
                        - **type**: ``list`` or None
                        - **shape of list**: ``[]``
            
            - **Must return**

                ``float``

    .. py:attribute:: DNNLikelihood.Samplerlogpdf_args   

            Additional arguments required by ``Sampler.logpdf``. 
            See :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`.
                - **type**: ``list`` or ``None``
                - **shape of list**: ``[]``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Samplerpars_pos_poi   

            Array containing the positions in the parameters list of the
            parameters of interest.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``(n_poi,)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Samplerpars_pos_nuis   

            Array containing the positions in the parameters list of the
            nuisance parameters.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``(n_nuis,)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Sampler.pars_init_vec

            Array of points with parameter initialization for each 
            walker. 
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``(nwalkers,n_pars)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Sampler.pars_labels   

            List containing the parameters names
            as string.
                - **type**: ``list`` or ``None``
                - **shape**: ``[]``
                - **length**: ``n_pars``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Sampler.nwalkers

            Number of walkers (equivalent of chains 
            for Ensamble Sampler MCMC).
                - **type**: ``int`` or ``None```
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Sampler.nsteps

            Number of MCMC steps to run.
            When ``Sampler`` is imported from file and already contains
            samples, if ``nsteps`` is less than the existing number of 
            steps, it is automatically updated to the available steps.
            ``nsteps`` always represents the final number of steps, so if
            the number of existing steps is not zero, the sampling will only
            run until it reaches ``nsteps``.
                - **type**: ``int`` or ``None```
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Sampler.output_folder

            Path (either relative to the code execution folder or absolute)
            where output files are saved. The __init__ method automatically converts 
            the path into an absolute path. If no output folder is specified, ``output_folder`` 
            is set to the code execution folder.
                - **type**: ``str`` or ``None``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Sampler.new_sampler

            If ``new_sampler=True`` a new ``Sampler`` object, corresponding to a new ``Sampler.backend``
            is generated. In case 
                - **type**: ``str`` or ``None``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Sampler.moves_str

        Bla bla

        :type: ``boh``
        :default: ``boh`` 

    .. py:attribute:: DNNLikelihood.Sampler.parallel_CPU

        Bla bla

        :type: ``boh``
        :default: ``boh``         

    .. py:attribute:: DNNLikelihood.Sampler.vectorize

        Bla bla

        :type: ``boh``
        :default: ``boh`` 

    .. py:attribute:: DNNLikelihood.Sampler.seed

        Bla bla

        :type: ``boh``
        :default: ``boh`` 


Additional attributes
"""""""""""""""""""""

    .. py:attribute:: DNNLikelihood.Sampler.pippo

        Bla bla

        :type: ``boh``
        :default: ``boh`` 


Methods
"""""""
