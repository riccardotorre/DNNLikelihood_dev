.. _likelihood_arguments:

Arguments
"""""""""

.. currentmodule:: Lik

.. argument:: name

    See :argument:`name <common_classes_arguments.name>`.

.. argument:: logpdf

    Callable function returning the logpdf given parameters and additional arguments, passed through the
    :argument:`logpdf_args` argument (for positional arguments) and 
    :argument:`logpdf_kwargs` argument (for keyword arguments).
        
        - **type**: ``callable`` or ``None``
        - **default**: ``None`` 

    - **Arguments**

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
     
    - **Returns**

        - ``Float`` if input shape is ``(ndims,)``
        - |Numpy_link| array with shape ``(n_points,)`` if input shape is ``(n_points,ndims)``

.. argument:: logpdf_args   

    Optional list of additional positional arguments required by the 
    :argument:`logpdf` function.
        
        - **type**: ``list`` or ``None``
        - **shape**: ``(nargs,)``
        - **default**: ``None`` 

.. argument:: logpdf_kwargs   

    Optional dictionary of additional keyword arguments required by the 
    :argument:`logpdf` function.
        
        - **type**: ``dict`` or ``None``
        - **default**: ``None`` 

.. argument:: pars_central   

    See :argument:`pars_central <common_classes_arguments.pars_central>`.

.. argument:: pars_pos_poi   

    See :argument:`pars_pos_poi <common_classes_arguments.pars_pos_poi>`.

.. argument:: pars_pos_nuis   

    See :argument:`pars_pos_nuis <common_classes_arguments.pars_pos_nuis>`.

.. argument:: pars_labels   

    See :argument:`pars_labels <common_classes_arguments.pars_labels>`.

.. argument:: pars_bounds   

    See :argument:`pars_bounds <common_classes_arguments.pars_bounds>`.

.. argument:: output_folder

    See :argument:`output_folder <common_classes_arguments.output_folder>`.

.. argument:: input_file

    See :argument:`input_file <common_classes_arguments.input_file>`.

.. argument:: verbose

    See :argument:`verbose <common_classes_arguments.verbose>`.

.. include:: ../external_links.rst