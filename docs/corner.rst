.. module:: corner_module
Notes on corner.py
------------------

The corner.py file is a slight customization of the |corner_link| package. 
Differences implemented up to now are:

    - added ``levels_lists`` argument to the function :func:`DNNLikelihood.corner.corner`

    - added ``normalize1d`` argument to the function :func:`DNNLikelihood.corner.corner`

.. |corner_link| raw:: html
    
    <a href="https://corner.readthedocs.io/en/latest/"  target="_blank"> corner</a>

The documentation below corresponds to the one from |corner_link| with the aforementioned changes.

.. autofunction:: DNNLikelihood.corner.corner

.. autofunction:: DNNLikelihood.corner.quantile

.. autofunction:: DNNLikelihood.corner.hist2d