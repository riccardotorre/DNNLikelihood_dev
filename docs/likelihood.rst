The Likelihood object
----------------------

Summary
^^^^^^^

The likelihood class acts as a container for the original likelihood function. It contains information on parameters,
parameters initializations, information on likelihood maxima and the logpdf function. In case in which the likelihood
is obtained using the histfactory interface, the logpdf is constructed from the pyhf.Workspace.model.logpdf method.

Usage
^^^^^

Describe the usage of the ``logpdf_fn`` method and the need of the ``generate_define_logpdf_file`` one.

Class
^^^^^

.. autoclass:: DNNLikelihood.Likelihood
   :undoc-members:

Arguments
"""""""""

    .. py:attribute:: DNNLikelihood.Likelihood.name   

            Likelihood name. If ``None`` is passed the name is generated as 
            ``"sampler_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")``, while if a string is passed, the 
            ``"_likelihood"`` suffix is appended (preventing duplication if it is already present).
                - **type**: ``str`` or ``None``
                - **default**: ``None``

    .. py:method:: DNNLikelihood.Likelihood.logpdf(x_pars,*args)   

            Function that calculates the logpdf given parameters values ``x_pars`` and additional 
            arguments ``args``, passed through the ``logpdf_args`` argument.
            As a class argument it should be passed as a callable function without arguments.
            Notice that access to the logpdf function given parameters is provided by the class method 
            :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>`.
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

                ``float`` or ``numpy.ndarray`` with shape ``(1,)``

    .. py:attribute:: DNNLikelihood.Likelihood.logpdf_args   

            Additional arguments required by ``Likelihood.logpdf``. 
            See :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>`.
                - **type**: ``list`` or ``None``
                - **shape of list**: ``[]``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.pars_pos_poi   

            Array containing the positions in the parameters list of the
            parameters of interest.
                - **type**: ``numpy.ndarray``
                - **shape**: ``(n_poi,)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.pars_pos_nuis   

            Array containing the positions in the parameters list of the
            nuisance parameters.
                - **type**: ``numpy.ndarray``
                - **shape**: ``(n_nuis,)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.pars_init   

            Array containing an initial value 
            for the parameters.
                - **type**: ``numpy.ndarray``
                - **shape**: ``(n_pars,)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.pars_labels   

            List containing the parameters names
            as string.
                - **type**: ``list``
                - **shape**: ``[]``
                - **length**: ``n_pars``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.pars_bounds   

            Array containing bounds 
            for the parameters.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``(n_pars,2)``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.output_folder   

            Path (either relative to the code execution folder or absolute)
            where output files are saved.
            The __init__ method automatically converts the path into an absolute path.
            If no output folder is specified, ``output_folder`` is set to the code execution folder.
                - **type**: ``str`` or ``None``
                - **default**: ``None`` 

    .. py:attribute:: DNNLikelihood.Likelihood.likelihood_input_file   

            File name (either relative to the code execution folder or absolute) of a saved ``Likelihood`` object.
            Whenever this parameter is not ``None``` all other parameters are ignored and the object is
            reconstructed from the imported file using the ``Likelihood.__load_likelihood`` private method.
            The attribute ``Likelihood.likelihood_input_file`` can contain or not the ".pickle" extension. In case it does not, 
            the extension is added by the ``__init__`` method, which also adds the full path.
            - **type**: ``str`` or ``None``
            - **default**: ``None`` 


Additional attributes
"""""""""""""""""""""

    .. py:attribute:: DNNLikelihood.Likelihood.output_base_filename   

            Base name (with full path) of the output files for the ``Likelihood.plot_logpdf_par``,
            ``Likelihood.save_likelihood``, and ``Likelihood.generate_define_logpdf_file``` methods. 
            It is set to ``path.join(self.output_foldermutils.check_add_suffix(name,"_likselihood"))``. 
            - **type**: ``str``

    .. py:attribute:: DNNLikelihood.Likelihood.X_logpdf_max
            
            Array containing the values of parameters at the global maximum
            of the logpdf computed with the ``Likelihood.compute_maximum_logpdf`` method.
            The attribute is ``None`` unless the ``Likelihood.compute_maximum_logpdf`` method
            has been called or the ``Likelihood`` object has been imported from file and
            already contained a value for the attribute.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``(n_pars,)``

    .. py:attribute:: DNNLikelihood.Likelihood.Y_logpdf_max  

            Value of logpdf at its global maximum computed with the ``Likelihood.compute_maximum_logpdf`` 
            method. The attribute is ``None`` unless the ``Likelihood.compute_maximum_logpdf`` method
            has been called or the ``Likelihood`` object has been imported from file and already contained a  
            value for the attribute.
                - **type**: ``float`` or ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.X_prof_logpdf_max

            Array containing the values of parameters at different local maxima of the logpdf computed
            with the ``Likelihood.compute_profiled_maxima`` method. The attribute is ``None`` unless
            the ``Likelihood.compute_profiled_maxima`` method has been called or the ``Likelihood`` 
            object has been imported from file and already contained a value for the attribute.
            This attribute can be used to initialize walkers in the :ref:``Sampler <sampler_class>`` object.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``np.array(n_points,n_pars)``
                - **default**: ``None``
                
    .. py:attribute:: DNNLikelihood.Likelihood.Y_prof_logpdf_max

            Array containing the values of logpdf at different local maxima computed
            with the ``Likelihood.compute_profiled_maxima`` method. The attribute is ``None`` unless
            the ``Likelihood.compute_profiled_maxima`` method has been called or the ``Likelihood`` 
            object has been imported from file and already contained a value for the attribute.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``np.array(n_points,)``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.X_prof_logpdf_max_tmp

            Same as ``X_prof_logpdf_max``. It is assigned only when attempting to append newly 
            generated profiled maxima to an incompatible existing ``X_prof_logpdf_max``.
            This is a temporary attribute and it is not saved by ``Likelihood.save_likelihood``.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``np.array(n_points,n_pars)``
                - **default**: ``None``
                
    .. py:attribute:: DNNLikelihood.Likelihood.Y_prof_logpdf_max_tmp

            Same as ``Y_prof_logpdf_max``. It is assigned only when attempting to append newly 
            generated profiled maxima to an incompatible existing ``Y_prof_logpdf_max``.
            This is a temporary attribute and it is not saved by ``Likelihood.save_likelihood``.
                - **type**: ``numpy.ndarray`` or ``None``
                - **shape**: ``np.array(n_points,)``
                - **default**: ``None``

    .. py:attribute:: DNNLikelihood.Likelihood.define_logpdf_file

            Name (with absolute path) of the output file containing the code necessary to intantiate a 
            ``Likelihooh`` object and define the corresponing parameters. This file can be generated using 
            the :ref:`Likelihood.generate_define_logpdf_file <likelihood_generate_define_logpdf_file>` method.
            and is sometimes needed to properly run Markov Chain Monte Carlo in parallel (using ``Multiprocessing``) 
            through the ``Sampler`` object inside Jupyter notebooks on the Windows platform.
            The atribute is set to ``Likelihood.output_base_filename+"_define_logpdf"+".py"`` while the path
            is set to ``Likelihood.output_folder``.
                - **type**: ``str``

Methods
"""""""

    .. automethod:: DNNLikelihood.Likelihood.__init__

    .. automethod:: DNNLikelihood.Likelihood.plot_logpdf_par

    .. automethod:: DNNLikelihood.Likelihood.compute_maximum_logpdf

    .. automethod:: DNNLikelihood.Likelihood.compute_profiled_maxima

    .. automethod:: DNNLikelihood.Likelihood.save_likelihood

    .. automethod:: DNNLikelihood.Likelihood.load_likelihood

    .. automethod:: DNNLikelihood.Likelihood.logpdf_fn

    .. automethod:: DNNLikelihood.Likelihood.generate_define_logpdf_file
