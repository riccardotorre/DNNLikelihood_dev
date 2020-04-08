.. _verbosity_mode:
Verbosity mode
--------------

The verbosity mode is implemented in this package through a modification of the built-in ``print`` function, 
corresponding to the following piece of code

.. code-block:: python
    
    import builtins

    verbose = True
    def print(*args, **kwargs):
        global verbose
        if type(verbose) is bool:
            if verbose:
                return builtins.print(*args, **kwargs)
        if type(verbose) is int:
            if verbose != 0:
                return builtins.print(*args, **kwargs)

in the file show_prints.py. Each of the modules in the package then contain in their initialization the code

.. code-block:: python
    
    from . import show_prints
    from .show_prints import print
    
In this way the ``print`` statement in each module is replaced by the customly defined one, which depends on the value
of the shared variable ``show_prints.verbose``. 

To control verbosity in the various classes, methods, and functions that accept a ``verbose`` argument, the global 
variable ``show_prints.verbose`` is set to the value of the ``verbose`` argument at the beginning of their definitions
through

.. code-block:: python
    
    class Class(..., verbose=True):
        show_prints.verbose = verbose

        ...
    
        def Method(..., verbose=False):
            show_prints.verbose = verbose
            
        ...
        
Accepted values for ``show_prints.verbose`` are ``True``, ``False``, and integers, and correspond, unless stated 
otherwise in the ``verbose`` argument documentation of the object, to the following general behavior:

    - ``ShowPrints=False`` or ``ShowPrints=0``
    
        All prints from the current function and all functions called in it
        are muted.

    - ``ShowPrints=True`` or ``ShowPrints>=1``
    
        All prints from the current function, as well as from the functions called in it are active. Whenever a 
        called function has more verbosity modes, corresponding to ``verbose>1``, the value of the ``verbose`` 
        argument is passed directly to the function (for instance, see the guide to the verbosity mode of the
        |keras_model_fit_link| function).

    - ``ShowPrints<0``

        All prints from the current function are active, while prints from the functions called in it are muted.

.. |keras_model_fit_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/builtins.html"  target="_blank"> tensorflow.keras.model.fit</a>