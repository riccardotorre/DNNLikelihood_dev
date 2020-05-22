.. _likelihood_arguments:

Arguments
"""""""""

.. currentmodule:: Lik

.. argument:: name

    Name of the :class:`Lik <DNNLikelihood.Lik>` object.
    It is used to build the :attr:`Lik.name <DNNLikelihood.Lik.name>` attribute.
     
        - **type**: ``str`` or ``None``
        - **default**: ``None``   

.. argument:: logpdf

    Callable function returning the logpdf given parameters and additional arguments, passed through the
    :argument:`logpdf_args` argument.
        
        - **type**: ``callable`` or ``None``
        - **default**: ``None`` 

    - **Could accept**

        - **x_pars**
            
            Values of the parameters for which logpdf is computed.
            It could be a single point in parameter space corresponding to an array with shape ``(ndims,)``
            or a list of points corresponding to an array with shape ``(n_points,ndims)``.
                
                - **type**: ``numpy.ndarray``
                - **possible shapes**: ``(ndims,)`` or ``(n_points,ndims)``

        - **args**

            Optional list of additional arguments required 
            by the :argument:`logpdf` function and passed through the :argument:`logpdf_args` input argument. 
                
                - **type**: ``list`` or None
                - **shape**: ``(nargs,)``

        - **kwargs**

            Optional dictionary of additional keyword arguments required 
            by the :argument:`logpdf` function and passed through the :argument:`logpdf_kwargs` input argument. 
                
                - **type**: ``dict`` or None
     
    - **Could return**

        ``float`` or ``numpy.ndarray`` with shape ``(n_points,)``

.. argument:: logpdf_args   

    Optional list of additional positional arguments required by the 
    :argument:`logpdf` function.
        
        - **type**: ``list`` or ``None``
        - **shape**: ``(nargs,)``

.. argument:: logpdf_kwargs   

    Optional dictionary of additional keyword arguments required by the 
    :argument:`logpdf` function.
        
        - **type**: ``dict`` or ``None``

.. argument:: pars_central   

    List or |numpy_link| array containing central values of the parameters.
    It is used to build the :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>` attribute.
        
        - **type**: ``list`` or ```numpy.ndarray``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_pos_poi   

    List or |numpy_link| array containing the positions in the parameters list of the
    parameters of interest.
    It is used to build the :attr:`Lik.pars_pos_poi <DNNLikelihood.Lik.pars_pos_poi>` attribute.

        - **type**: ``list`` or ```numpy.ndarray``
        - **shape**: ``(n_poi,)``
        - **default**: ``None`` 

.. argument:: pars_pos_nuis   

    List or |numpy_link| array containing the positions in the parameters list of the
    nuisance parameters.
    It is used to build the :attr:`Lik.pars_pos_nuis <DNNLikelihood.Lik.pars_pos_nuis>` attribute.

        - **type**: ``list`` or ``numpy.ndarray``
        - **shape**: ``(n_nuis,)``
        - **default**: ``None`` 

.. argument:: pars_labels   

    List containing the parameters names as strings.
    Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
    and can contain latex expressions that are properly compiled when making plots.
    It is used to build the :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>` attribute.

        - **type**: ``list``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_bounds   

    List or |numpy_link| array containing containing bounds for the parameters.
    It is used to build the :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` attribute.

        - **type**: ``numpy.ndarray`` or ``None``
        - **shape**: ``(ndims,2)``
        - **default**: ``None`` 

.. argument:: output_folder
     
    Path (either relative to the code execution folder or absolute) where output files are saved.
    It is used to set the :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` attribute.
        
        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: input_file   

    File name (either relative to the code execution folder or absolute, with or without extension) 
    of a saved :class:`Lik <DNNLikelihood.Lik>` object. 
    It is used to set the 
    :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>` 
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