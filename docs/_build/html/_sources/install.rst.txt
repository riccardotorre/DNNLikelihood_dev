Installation
============

Quick start
-----------

Requirements
------------
Several packages are required by the DNNLikelihood module. We provide a script to automatically set up the environment.
And a script for environment check, which allows the user to tune its existing environment to run the DNNLikelihood 
package. We also list here all required packages with links to their documentation. The list is divided in general
requirements, i.e. packages that are needed by most of the objects and functions and the various classes with their
specific requirements.

General requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following modules are required by most objects in the DNNLikelihood package.

    - Default Python modules

        - |builtins_link|
        
        - |datetime_link|

        - |os_link|

        - |pickle_link|

        - |sys_link|

        - |timeit_link|

    - External modules

        - |matplotlib_link|
        
        - |numpy_link|

The Histfactory object
^^^^^^^^^^^^^^^^^^^^^^

    - Default modules

        - |json_link|

    - External modules

        - (optional) |ipywidgets_link|

        - |json_patch_link|
        
        - |json_schema_link|
        
        - |pyhf_link|

        - |requests_link|

    - Internal modules

        - :obj:`DNNLikelihood.utils <utils_module>`

        - :class:`DNNLikelihood.Likelihood`

The Likelihood object
^^^^^^^^^^^^^^^^^^^^^^

    - External modules

        - |cloudpickle_link|        
        
        - (optional) |ipywidgets_link|

        - |json_patch_link|
        
        - |json_schema_link|
        
        - |pyhf_link|

        - |requests_link|

    - Internal modules

        - :obj:`DNNLikelihood.utils <utils_module>`

        - :obj:`DNNLikelihood.inference <inference_module>`


The Sampler object
^^^^^^^^^^^^^^^^^^

    - Default modules

        - |copy_link|

        - |importlib_link|

        - |multiprocessing_link|

    - External modules

        - |psutil_link|

        - |scipy_link|
        
        - |emcee_link|

    - Internal modules

        - :obj:`DNNLikelihood.utils <utils_module>`

        - :class:`DNNLikelihood.Data`

The Data object
^^^^^^^^^^^^^^^

    - External modules

        - |h5py_link|

        - |sklearn_link|

    - Internal modules

        - :obj:`DNNLikelihood.utils <utils_module>`

The DNN_likelihood object
^^^^^^^^^^^^^^^^^^^^^^^^^

    - Default modules

        - ||

    - External modules

        - ||

    - Internal modules

        - :obj:`DNNLikelihood.inference <inference_module>`

        - :obj:`DNNLikelihood.set_resources <set_resources_module>`

        - :obj:`DNNLikelihood.utils <utils_module>`

        - :class:`DNNLikelihood.Data`

        - :class:`DNNLikelihood.Data`

The DNN_likelihood_ensemble object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Environment
-----------

.. |builtins_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/builtins.html"  target="_blank"> builtins</a>

.. |datetime_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/datetime.html"  target="_blank"> datetime</a>

.. |os_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/os.html"  target="_blank"> os</a>

.. |pickle_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/pickle.html"  target="_blank"> pickle</a>

.. |sys_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/sys.html"  target="_blank"> sys</a>

.. |timeit_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/timeit.html"  target="_blank"> timeit</a>

.. |matplotlib_link| raw:: html
    
    <a href="https://matplotlib.org/"  target="_blank"> matplotlib</a>

.. |numpy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/numpy/index.html"  target="_blank"> numpy</a>

.. |json_link| raw:: html
    
    <a href="https://docs.python.org/3.8/library/json.html"  target="_blank"> json</a>

.. |json_patch_link| raw:: html
    
    <a href="https://python-json-patch.readthedocs.io/en/stable/"  target="_blank"> json_patch</a>

.. |json_schema_link| raw:: html
    
    <a href="https://python-jsonschema.readthedocs.io/en/stable/"  target="_blank"> json_schema</a>

.. |pyhf_link| raw:: html
    
    <a href="https://scikit-hep.org/pyhf/"  target="_blank"> pyhf</a>

.. |requests_link| raw:: html
    
    <a href="https://requests.readthedocs.io/en/master/"  target="_blank"> requests</a>

.. |ipywidgets_link| raw:: html
    
    <a href="https://ipywidgets.readthedocs.io/en/latest/"  target="_blank"> ipywidgets</a>

.. |cloudpickle_link| raw:: html
    
    <a href="https://pypi.org/project/cloudpickle/1.3.0/"  target="_blank"> cloudpickle</a>

.. |importlib_link| raw:: html
    
    <a href="https://docs.python.org/3/library/importlib.html"  target="_blank"> importlib</a>

.. |copy_link| raw:: html
    
    <a href="https://docs.python.org/3/library/copy.html"  target="_blank"> copy</a>

.. |multiprocessing_link| raw:: html
    
    <a href="https://docs.python.org/3/library/multiprocessing.html"  target="_blank"> multiprocessing</a>

.. |psutil_link| raw:: html
    
    <a href="https://psutil.readthedocs.io/en/latest/"  target="_blank"> psutil</a>

.. |scipy_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/scipy/reference/"  target="_blank"> scipy</a>

.. |emcee_link| raw:: html
    
    <a href="https://emcee.rhttps://emcee.readthedocs.io/en/stable/"  target="_blank"> emcee</a>

.. |h5py_link| raw:: html
    
    <a href="http://docs.h5py.org/en/stable/"  target="_blank"> h5py</a>

.. |sklearn_link| raw:: html
    
    <a href="https://scikit-learn.org/stable/"  target="_blank"> sklearn</a>






