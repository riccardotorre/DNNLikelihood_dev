.. _verbosity_mode:

Verbosity mode
--------------

Summary
^^^^^^^

The verbosity mode is implemented through the file ``show_prints.py``. This contains a modification of the built-in ``print`` function,
that accepts an additional argument ``show`` allowing to switch print on and off, and a class 
:class:`Verbosity <DNNLikelihood.Verbosity>` that, through inheritance, makes the 
:meth:`Verbosity.set_verbosity <DNNLikelihood.Verbosity.set_verbosity>` method available to all classes in the module.

The ``verbose`` argument available in most classes and methods in the package could be ``True``, 
``False``, or any integers, corresponding, unless stated otherwise in the ``verbose`` argument documentation of the object, 
to the following general behavior:

    - ``ShowPrints=False`` or ``ShowPrints=0``
    
        All prints from the current function and all functions called in it are muted.

    - ``ShowPrints=True`` or ``ShowPrints>=1``
    
        All prints from the current function, as well as from the functions called in it are active. Whenever a 
        called function has more verbosity modes, corresponding to ``verbose>1``, the value of the ``verbose`` 
        argument is passed directly to the function (for instance, see the guide to the verbosity mode of the
        |tf_keras_model_fit_link| function).

    - ``ShowPrints<0``

        All prints from the current function are active, while prints from the functions called in it are muted.

All classes have a ``verbose=True`` default argument and the ``verbose`` argument is stored in the ``self.verbose`` 
attribute. This is used as verbosity mode for the class ``__init__`` method, as well as default argument of all methods 
in that class that accept a ``verbose`` argument (i.e. that involve, directly or indirectly, any call to the
``print`` function or to the ``plt.show`` method).

To further clarify the verbosity implementation a scheme how classes are structured is provided by the following code:

.. code-block:: python
    
    from .show_prints import Verbosity, print
    
    class ClassA(Verbosity)
        def __init__(self,..., verbose=True):
            self.verbose = verbose
            verbose, verbose_sub = self.set_verbosity(verbose)
        ...
    
        def MethodA(..., verbose=None):
            verbose, _ = self.set_verbosity(verbose)
            ...
            print(...,show=verbose)
            ...

        def MethodB(..., verbose=None):
            verbose, verbose_sub = self.set_verbosity(verbose)
            ...
            self.MethodA(...,verbose=verbose_sub)
            ...
            print(...,show=verbose)
            ...

        def MethodC(..., verbose=None):
            verbose, verbose_sub = self.set_verbosity(verbose)
            ...
            self.MethodB(...,verbose=verbose_sub)
            ...
            print(...,show=verbose)
            ...
            if verbose:
                plt.show(...)
            ...

Code documentation
^^^^^^^^^^^^^^^^^^

.. autoclass:: DNNLikelihood.Verbosity
    :undoc-members:

    .. automethod:: DNNLikelihood.Verbosity.set_verbosity

.. autofunction:: DNNLikelihood.print

.. include:: external_links.rst