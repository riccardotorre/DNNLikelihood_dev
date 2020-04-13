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
        try:
            show = kwargs.pop("show")
        except:
            show = None
        if show is None:
            if verbose:
                return builtins.print(*args, **kwargs)
        elif show:
            return builtins.print(*args, **kwargs)

in the file show_prints.py. Each of the modules in the package then contain in their code the initialization

.. code-block:: python
    
    from . import show_prints
    from .show_prints import print
    
In this way the ``print`` statement in each module is replaced by the customly defined one, which depends on the value
of the shared variable ``show_prints.verbose``. 

The ``show_prints.verbose`` argument (and therefore all ``verbose`` arguments in the package) accept values ``True``, 
``False``, and integers, corresponding, unless stated otherwise in the ``verbose`` argument documentation of the object, 
to the following general behavior:

    - ``ShowPrints=False`` or ``ShowPrints=0``
    
        All prints from the current function and all functions called in it are muted.

    - ``ShowPrints=True`` or ``ShowPrints>=1``
    
        All prints from the current function, as well as from the functions called in it are active. Whenever a 
        called function has more verbosity modes, corresponding to ``verbose>1``, the value of the ``verbose`` 
        argument is passed directly to the function (for instance, see the guide to the verbosity mode of the
        |keras_model_fit_link| function).

    - ``ShowPrints<0``

        All prints from the current function are active, while prints from the functions called in it are muted.

All classes inherit the :class:`show_prints.Verbosity <DNNLikelihood.show_prints.Verbosity>` class defined in the 
file show_prints.py as

.. code-block:: python

    class Verbosity():
        def set_verbosity(self, v):
            global verbose
            if v is None:
                verbose = self.verbose
            else:
                verbose = v
            if verbose < 0:
                verbose_sub = 0
            else:
                verbose_sub = verbose
            return [verbose, verbose_sub]

which makes the ``set_verbosity`` method available. This is used to control verbosity in the various classes, methods, 
and functions as explained below.

All classes have a ``verbose=True`` default argument and the ``verbose`` argument is stored in a the ``self.verbose`` 
attribute. This is used as verbosity mode for the "__init__" method. Moreover, the value of ``self.verbose`` is passed
as default to each methods and functions in the class that accept a ``verbose`` argument. This is done by setting
their default ``verbose`` argument to ``None`` and by calling the ``set_verbosity(verbose)`` method in the first 
line of the method or function body.

This sets the global variable ``show_prints.verbose`` to ``self.verbose`` if ``verbose=False`` and to ``verbose`` otherwise.
Moreover it can return two local variables ``verbose`` and ``verbose_sub`` used to implement the behavior discussed above.
In particular, the local variable ``verbose`` is only used to print plots (all other prints are controlled by the custom ``print`` function
and the value of the global variable ``show_prints.verbose``), while the local variable ``verbose_sub`` is passed as
``verbose`` argument to all methods and functions called within the given one.

To further clarify the verbosity implementation a scheme is provided by the following code:

.. code-block:: python
    
    from . import show_prints
    from .show_prints import print
    
    class Class(show_prints.Verbosity)
        def __init__(self,..., verbose=True):
            show_prints.verbose = verbose
            self.verbose = verbose
        ...
    
        def MethodA(..., verbose=None):
            self.set_verbosity(verbose)
            ...
            print(...)
            ...

        def MethodB(..., verbose=None):
            _, verbose_sub = self.set_verbosity(verbose)
            ...
            self.MethodA(...,verbose=verbose_sub)
            ...
            print(...)
            ...

        def MethodC(..., verbose=None):
            verbose, verbose_sub = self.set_verbosity(verbose)
            ...
            self.MethodB(...,verbose=verbose_sub)
            ...
            print(...)
            ...
            if verbose:
                plt.show(...)
            ...

.. autoclass:: DNNLikelihood.show_prints.Verbosity
    :undoc-members:

    .. automethod:: DNNLikelihood.show_prints.Verbosity.set_verbosity

.. autofunction:: DNNLikelihood.show_prints.print

.. |keras_model_fit_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/builtins.html"  target="_blank"> tensorflow.keras.model.fit</a>

