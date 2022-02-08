Methods
"""""""

.. currentmodule:: DNNLikelihood

.. automethod:: Lik.__init__

.. automethod:: Lik._Lik__check_define_input_files

.. automethod:: Lik._Lik__check_define_name

.. automethod:: Lik._Lik__check_define_ndims

.. automethod:: Lik._Lik__check_define_output_files

.. automethod:: Lik._Lik__check_define_pars

.. automethod:: Lik._Lik__load

.. automethod:: Lik._Lik__set_pars_labels

.. automethod:: Lik.compute_maximum_logpdf

.. automethod:: Lik.compute_profiled_maxima_logpdf

.. _likelihood_logpdf_method:

.. py:method:: Lik.logpdf(x_pars)

    Method built through the :class:`_FunctionWrapper <DNNLikelihood.utils._FunctionWrapper>` class
    to collect the logpdf function corresponding to the input argument :argument:`logpdf` and
    its additional arguments corresponding to the optional input arguments
    :argument:`logpdf_args` and :argument:`logpdf_kwargs`. This allows to save the logpdf function 
    with all its arguments as a :class:`Lik <DNNLikelihood.Lik>` class "attribute".

    The function gives the value of the logpdf given the vector of parameters and it contains three
    attributes:

        - ``logpdf.f``: the actual callable function corresponding to the input argument :argument:`logpdf`
        - ``logpdf.args``: the optional positional arguments corresponding to the input argument :argument:`logpdf_args`
        - ``logpdf.kwargs``: the optional keyword arguments corresponding to the input argument :argument:`logpdf_kwargs`

    Evaluating the :meth:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` method given an input vector ``x_pars``
    corresponds to evaluating the function ``logpdf.f`` with all arguments set in ``logpdf.args`` 
    and ``logpdf.kwargs``, which can be manually changed (``logpdf(x)`` gives the same result as 
    ``logpdf.f(x,*logpdf_args,**logpdf_kwargs)``)

    This function is not meant to be used in practica and is used to construct the 
    :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` method, which implements parameter bounds and avoids
    ``nan`` output, and is therefore more suitable to be used for computations and as input 
    for the :obj:`Sampler <sampler>` object.
    
    - **Arguments**
    
      - **x_pars**
            
        Values of the parameters for which logpdf is computed.
        It could be a single point in parameter space corresponding to an array with shape ``(ndims,)``
        or a list of points corresponding to an array with shape ``(n_points,ndims)``.
            
            - **type**: ``numpy.ndarray``
            - **possible shapes**: ``(ndims,)`` or ``(n_points,ndims)``
     
    - **Returns**
    
        Value or array of values 
        of the logpdf.

            - **type**: ``float`` or ``numpy.ndarray``
            - **shape for numpy.ndarray**: ``(n_points,)``

.. automethod:: Lik.logpdf_fn

.. automethod:: Lik.plot_logpdf_par

.. automethod:: Lik.plot_tmu_1d

.. automethod:: Lik.reset_predictions

.. automethod:: Lik.save

.. automethod:: Lik.save_h5

.. automethod:: Lik.save_json

.. automethod:: Lik.save_log

.. automethod:: Lik.save_predictions_json

.. automethod:: Lik.save_script

.. py:method:: Lik.set_verbosity

    Method inherited from the :class:`Verbosity <DNNLikelihood.Verbosity>` object.
    See the documentation of :meth:`Verbosity.set_verbosity <DNNLikelihood.Verbosity.set_verbosity>`.

.. automethod:: Lik.update_figures

.. include:: ../external_links.rst