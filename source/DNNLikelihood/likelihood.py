__all__ = ["LikFileManager",
           "LikParsManager",
           "LikPredictions",
           "Lik"]

import builtins
import codecs
import h5py #type: ignore
import json
import time
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
from copy import copy
from datetime import datetime
from os import path, sep, stat
from timeit import default_timer as timer

import cloudpickle #type: ignore
import deepdish as dd #type: ignore
from IPython.core.display import display # type: ignore
import matplotlib.pyplot as plt #type: ignore
import numpy as np
from numpy import typing as npt

from . import inference, utils
from .show_prints import Verbosity, print
from .base import LogPDF, FileManager, ParsManager, Predictions, Inference,  Figures, Plots

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

Array = Union[List, npt.NDArray]
ArrayInt = Union[List[int], npt.NDArray[np.int_]]
ArrayStr = Union[List[str], npt.NDArray[np.str_]]
StrPath = Union[str,Path]
IntBool = Union[int, bool]
StrBool = Union[str, bool]
LogPredDict = Dict[str,Dict[str,Any]]

header_string = "=============================="
footer_string = "------------------------------"


class LikFileManager(FileManager):
    obj_name = "Lik"

    def __init__(self,
                 name: Union[str,None] = None,
                 input_file: Optional[StrPath] = None, 
                 output_folder: Optional[StrPath] = None, 
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Define self.input_file, self.output_folder
        super().__init__(name=name,
                         input_file=input_file,
                         output_folder=output_folder,
                         verbose=verbose)
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__define_lik_script_file()

    def __define_lik_script_file(self) -> None:
        self.script_file = self.output_folder.joinpath(self.name.name+"_script.py")

    def __load(self, 
               obj: "Lik",
               verbose: Optional[IntBool] = None
              ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        obj_dict = dd.io.load(self.input_h5_file)
        tmp = obj_dict["logpdf_dump"]
        obj_dict["logpdf"] = cloudpickle.loads(tmp.tostring())
        obj_dict.pop("logpdf_dump")
        with self.input_log_file.open() as json_file:
            obj.log = json.load(json_file)
        obj.__dict__.update(obj_dict)
        end = timer()
        timestamp = utils.generate_timestamp()
        obj.log[timestamp] = {"action": "loaded",
                              "files names": [self.input_h5_file.name,
                                              self.input_log_file.name]}
        print(header_string, "\n",self.obj_name," object loaded in", str(end-start), ".\n", show=verbose)
        time.sleep(3)  # Removing this line prevents multiprocessing to work properly on Windows

    def save_script(self,
                    str_to_save: str,
                    log: LogPredDict,
                    overwrite: StrBool = False,
                    verbose: Optional[Union[int, bool]] = None
                   ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        start = timer()
        output_script_file = self.get_target_file_overwrite(input_file=self.script_file,
                                                            timestamp = timestamp,
                                                            overwrite=overwrite,
                                                            verbose=verbose_sub)
        with open(output_script_file, "w") as out_file:
            out_file.write(str_to_save)
        log[timestamp] = {"action": "saved script file",
                          "file name": output_script_file.name}
        end = timer()
        self.print_save_info(filename=output_script_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)


class LikParsManager(ParsManager):
    """
    """
    def __init__(self,
                 pars_central: Optional[Array],
                 pars_pos_poi: Optional[ArrayInt],
                 pars_pos_nuis: Optional[ArrayInt],
                 pars_labels: Optional[ArrayStr],
                 pars_bounds: Optional[Array],
                 logpdf: Optional[LogPDF] = None,
                 verbose: Optional[IntBool] = None) -> None:
        super().__init__(pars_central = pars_central,
                         pars_pos_poi = pars_pos_poi,
                         pars_pos_nuis = pars_pos_nuis,
                         pars_labels = pars_labels,
                         pars_bounds = pars_bounds,
                         logpdf = logpdf,
                         verbose = verbose)

class LikPredictions(Predictions):
    """
    """
    def __init__(self,
                 obj_name: str,
                 verbose = None) -> None:
        super().__init__(obj_name = obj_name,
                         verbose=verbose)


class LikPlots(Plots):
    """
    """
    def __init__(self,
                 verbose = None) -> None:
        super().__init__(verbose=verbose)

    


class Lik(Verbosity):
    """
    This class is a container for the :mod:`Likelihood <likelihood>` object, storing all information of the likelihood function.
    The object can be directly created or obtained from an ATLAS histfactory workspace through the 
    :class:`DNNLikelihood.Histfactory` object (see the :mod:`Histfactory <histfactory>` object documentation).
    """
    def __init__(self,
                 file_manager: LikFileManager,
                 logpdf: LogPDF,
                 parameters: LikParsManager,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Declaration of needed types for attributes
        self.log: LogPredDict
        self.parameters: LikParsManager
        self.logpdf: LogPDF
        self.name: str
        # Initialization of parent class
        super().__init__(verbose)
        # Initialization of verbosity mode
        verbose, verbose_sub = self.set_verbosity(self.verbose)
        # Initialization of object
        timestamp = utils.generate_timestamp()
        print(header_string, "\nInitialize Likelihood object.\n", show=verbose)
        self.file_manager = file_manager
        self.predictions = LikPredictions(self.file_manager.obj_name, verbose=verbose_sub)         # Predictions need to be saved and loaded in a robust way
        self.figures = Figures(verbose=verbose_sub)
        if self.file_manager.input_file is None:
            self.log = {timestamp: {"action": "created"}}
            self.name = self.file_manager.name.name              #OK
            self.parameters = parameters                         # Parameters need to be saved with the object itself
            self.logpdf = logpdf                                 # Needs to be saved and loaded in a robust way
        else:
            self.__load(verbose=verbose_sub)
            # Check predictions
            self.predictions.validate_predictions()
        if self.file_manager.input_file is None:
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)

    @property
    def exclude_attrs(self) -> list:
        tmp = ["exclude_attrs",
               "file_manager",
               "logpdf",
               "log",
               "verbose"]
        return tmp

    def __load(self,
               verbose: Optional[IntBool] = None
               ) -> None:
        self.file_manager.__load(obj = self, verbose = verbose)

    def save_log(self,
                 overwrite: StrBool = False,
                 verbose: Optional[Union[int, bool]] = None
                ) -> None:
        """
        See HistFactoryFileManager.save_log
        """
        self.file_manager.save_log(log=self.log, 
                                   overwrite = overwrite,
                                   verbose = verbose)

    def save_object(self, 
                    overwrite: StrBool = False,
                    verbose: Optional[Union[int, bool]] = None
                   ) -> None:
        """
        Save object json (predictions not included)
        """
        dictionary = utils.dic_minus_keys(self.__dict__, self.exclude_attrs)
        dump = np.void(cloudpickle.dumps(self.logpdf))
        dictionary_h5 = {**dictionary, **{"logpdf_dump": dump}}
        dictionary_json = utils.dic_minus_keys(dictionary, ["predictions"])        
        kwargs: Dict[str, Any] = {"log": self.log,
                                  "overwrite": overwrite,
                                  "verbose": verbose}
        self.file_manager.save_h5(dict_to_save=dictionary_h5, **kwargs)
        self.file_manager.save_json(dict_to_save=dictionary_json, **kwargs)

    def save_predictions_json(self, 
                              overwrite: StrBool = False,
                              verbose: Optional[Union[int, bool]] = None
                             ) -> None:
        """ 
        Save predictions json
        """
        dic_pred = utils.convert_types_dict(self.predictions.__dict__)
        dic_figs = {"figures": self.figures.figures}
        dictionary = {**dic_pred,  **dic_figs}
        self.file_manager.save_json(dict_to_save=dictionary,
                                    log=self.log,
                                    overwrite = overwrite,
                                    verbose = verbose)

    def save(self,
             overwrite: StrBool = False,
             verbose: Optional[Union[int, bool]] = None
             ) -> None:
        """
        """
        verbose, _ = self.set_verbosity(verbose)
        kwargs: Dict[str, Any] = {"overwrite": overwrite,
                                  "verbose": verbose}
        self.save_object(**kwargs)
        self.save_predictions_json(**kwargs)
        self.save_log(**kwargs)

    def save_script(self, 
                    timestamp: Optional[str] = None, # timestamp of prediction used to extract logpdf_profiled_max to initialize pars_init_vec
                    overwrite: StrBool = False,
                    verbose: Optional[Union[int, bool]] = None
                   ) -> None:
        """ 
        Save predictions json
        """
        str_to_save = "import DNNLikelihood\n"+ \
                      "import numpy as np\n" + "\n" + \
                      "lik = DNNLikelihood.Lik(name=None,\n" + \
                      "\tinput_file="+r"'" + r"%s" % str(self.file_manager.output_h5_file)+"', \n"+ \
                      "verbose = "+str(self.verbose)+")"+"\n"+"\n" + \
                      "name = lik.name\n" + \
                      "def logpdf(x_pars,*args,**kwargs):\n" + \
                      "\treturn lik.logpdf_fn(x_pars,*args,**kwargs)\n" + \
                      "logpdf_args = lik.logpdf.args\n" + \
                      "logpdf_kwargs = lik.logpdf.kwargs\n" + \
                      "pars_pos_poi = lik.pars_pos_poi\n" + \
                      "pars_pos_nuis = lik.pars_pos_nuis\n" + \
                      "pars_central = lik.pars_central\n" + \
                      "try:\n" + \
                      "\tpars_init_vec = lik.predictions['logpdf_profiled_max']['%s']['X']\n"%timestamp + \
                      "except:\n" + \
                      "\tpars_init_vec = None\n" \
                      "pars_labels = lik.pars_labels\n" + \
                      "pars_bounds = lik.pars_bounds\n" + \
                      "ndims = lik.ndims\n" + \
                      "output_folder = lik.output_folder"
        self.file_manager.save_script(str_to_save=str_to_save,
                                      log=self.log,
                                      overwrite = overwrite,
                                      verbose = verbose)

    #def update_figures(self,
    #                   figure_file: StrPath,
    #                   timestamp: Optional[str] = None,
    #                   overwrite: Union[str,bool] = False,
    #                   verbose: Union[int, bool, None] = None
    #                   ) -> Path:
    #    """
    #    """
    #    new_figure_file = self.figures.update_figures(figure_file = figure_file,
    #                                                  file_manager = self.file_manager,
    #                                                  log = self.log,
    #                                                  timestamp = timestamp,
    #                                                  overwrite = overwrite,
    #                                                  verbose = verbose
    #                                                  )
    #    return new_figure_file


    def logpdf_fn(self, 
                  x_pars: Array, 
                  logpdf_args: Optional[list] = None, 
                  logpdf_kwargs: Optional[Dict[str,Any]] = None
                 ) -> Union[float, npt.NDArray]:
        """
        This function is used to add constraints and standardize input/output of the
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` callable attribute.
        It is constructed from the :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute. 
        In the case :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` accepts a single array of 
        parameters ``x_pars`` and returns the logpdf value one point at a time, then the function returns a ``float``, 
        while if :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` is vectorized,
        i.e. accepts an array of ``x_pars`` arrays and returns an array of logpdf values, then the function returns an array. 
        Moreover, the :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` method is constructed to return the
        logpdf value ``-np.inf`` if any of the parameters lies outside 
        :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` or if the 
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` function returns ``nan``.

        - **Arguments**

            - **x_pars**

                Values of the parameters for which the logpdf is computed.
                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``) 
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``, depending on the 
                equivalent argument accepted by ``Lik.logpdf``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(n_pars,)`` or ``(n_points,n_pars)``

            - **args**

                Optional list of additional positional arguments needed by the :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` function. 

                    - **type**: ``list`` or None
                    - **shape of list**: ``[]``

            - **kwargs**

                Optional dictionary of additionale keyword arguments needed by the :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` function. 

                    - **type**: ``dict`` or None

        - **Returns**

            Value (values)
            of the logpdf.
            
                - **type**: ``float`` or ``numpy.ndarray``
                - **shape for numpy.ndarray**: ``(n_points,)``
        """
        logpdf = copy(self.logpdf)
        logpdf.args = list(self.logpdf.args) if logpdf_args is None else logpdf_args
        logpdf.kwargs = dict(self.logpdf.kwargs) if logpdf_kwargs is None else logpdf_kwargs
        x = np.array(x_pars)
        if len(np.shape(x)) == 1:
            if not (np.all(x >= self.parameters.pars_bounds[:, 0]) and np.all(x <= self.parameters.pars_bounds[:, 1])):
                return -np.inf
            tmp = np.array([logpdf(x)]).flatten()[0]
            if np.isnan(tmp):
                return -np.inf
            else:
                return tmp
        else:
            x_list = x
            tmp = np.array([logpdf(x)]).flatten()
            for i in range(len(x_list)):
                x = x_list[i]
                if not (np.all(x >= self.parameters.pars_bounds[:, 0]) and np.all(x <= self.parameters.pars_bounds[:, 1])):
                    tmp[i] = -np.inf
            tmp = np.where(np.isnan(tmp), -np.inf, tmp)
            return tmp
        
    def compute_maximum_logpdf(self,
                               pars_init=None,
                               pars_bounds=None,
                               optimizer={},
                               minimization_options={},
                               timestamp = None,
                               save=True,
                               overwrite=True,
                               verbose=None):
        """
        Computes the maximum of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. 
        All information on the maximum, including parameters initialization, parameters bounds, and optimizer, 
        are stored in the ``"logpdf_max"`` item of the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary.
        The method uses the function :func:`inference.compute_maximum_logpdf <DNNLikelihood.inference.compute_maximum_logpdf>`
        based on |scipy_optimize_minimize_link| to find the minimum of minus 
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. If ``pars_bounds`` is ``None``, then
        the parameter bounds stored in the :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` attribute are used. 
        See the documentation of the
        :func:`inference.compute_maximum_logpdf <DNNLikelihood.inference.compute_maximum_logpdf>` 
        function for details.

        - **Arguments**

            - **pars_init**
            
                Starting point for the optimization. If it is ``None``, then
                it is set to the parameters central value :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,)``
                    - **default**: ``None`` (automatically modified to :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`)

            - **pars_bounds**
            
                Bounds on the parameters. If it is ``None``, then default parameters bounds stored in the 
                :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` attribute are used.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,2)``
                    - **default**: ``None``

            - **optimizer**

                Dictionary containing information on the optimizer and its options.
                    
                    - **type**: ``dict`` with the following structure:

                        - *"name"* (value type: ``str``)
                          This is always set to ``"scipy"``, which is, by now, the only available optimizer for this task. 
                          As more optimizers will be supported the ``"name"`` key will indicate the chosen one.
                        - *"args"* (value type: ``str``)
                          Additional positional arguments passed to the |scipy_optimize_minimize_link| method.
                        - *"kwargs"* (value type: ``dict``)
                          Additional keyword arguments passed to the |scipy_optimize_minimize_link| method (set to the
                          ``{"method": "Powell"}`` dictionary by default).

                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            optimizer={"name": "scipy",
                                       "args": [],
                                       "kwargs": {"method": "Powell"}},

            - **minimization_options**

                Dictionary containing options to be passed to the |scipy_optimize_minimize_link| method 
                (i.e. value of the "options" keyword argument of the |scipy_optimize_minimize_link| method).
                    
                    - **type**: ``dict``
                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            minimization_options={"maxiter": 10000,
                                                  "ftol": 0.0001},

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **save**
            
                If ``True`` the object is saved after the calculation.
                    
                    - **type**: ``bool``
                    - **default**: ``True``

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nComputing global maximum.\n",show=verbose)
        start = timer()
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy",[],{"method": "Powell"}],verbose=verbose_sub)
        if pars_init is None:
            pars_init = np.array(self.parameters.pars_central)
        else:
            pars_init = np.array(pars_init)
        utils.check_set_dict_keys(self.predictions.logpdf_max,
                                  [timestamp],
                                  [{}], verbose=False)
        res = inference.compute_maximum_logpdf(logpdf=lambda x: self.logpdf_fn(x,*self.logpdf.args, *self.logpdf.kwargs), 
                                               pars_init=pars_init,
                                               pars_bounds=pars_bounds,
                                               optimizer=optimizer,
                                               minimization_options=minimization_options,
                                               verbose=verbose_sub)
        self.predictions.logpdf_max[timestamp]["x"], self.predictions.logpdf_max[timestamp]["y"] = res
        end = timer()
        self.predictions.logpdf_max[timestamp]["pars_init"] = pars_init
        if pars_bounds is None:
            self.predictions.logpdf_max[timestamp]["pars_bounds"] = self.parameters.pars_bounds
        else:
            self.predictions.logpdf_max[timestamp]["pars_bounds"] = pars_bounds
        self.predictions.logpdf_max[timestamp]["optimizer"] = optimizer
        self.predictions.logpdf_max[timestamp]["minimization_options"] = minimization_options
        self.predictions.logpdf_max[timestamp]["optimization_time"] = end-start
        self.log[timestamp] = {"action": "computed maximum logpdf"}
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)
        print("Maximum logpdf computed in",end-start,"s.\n",show=verbose)

    def compute_profiled_maxima_logpdf(self,
                                       pars=None,
                                       pars_ranges=None,
                                       pars_init = None,
                                       pars_bounds = None,
                                       spacing="grid",
                                       optimizer = {},
                                       minimization_options={},
                                       progressbar=False,
                                       timestamp = None,
                                       save=True,
                                       overwrite=True,
                                       verbose=None):
        """
        Computes local (profiled) maxima of the logpdf for different values of the parameters ``pars``.
        For the list of parameters ``pars``, ranges are passed as ``pars_ranges`` in the form ``(par_min,par_max,n_points)``
        and an array of points is generated according to the argument ``spacing`` (either a grid or a random 
        flat distribution) in the interval. If ``pars_bounds`` is ``None``, then
        the parameter bounds stored in the :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` attribute are used. 
        The points in the grid falling outside 
        :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` are automatically removed.
        All information on the maximum, including parameters initialization, parameters bounds, and optimizer, 
        are stored in the ``"logpdf_profiled_max"`` item of the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary.
        The method also automatically computes, with the same optimizer, the global maximum and the :math:`t_{\\pmb\\mu}`
        test statistics. The latter is defined, given a vector of parameters under which the logpdf is not profiled 
        :math:`\\pmb\\mu` and a vector of parameters under which it is profiled :math:`\\pmb\\delta` as
        
        .. math::

            t_{\\pmb\\mu}=-2\\left(\\sup_{\\pmb\\delta}\\log\\mathcal{L}(\\pmb\\mu,\\pmb\\delta)-\\sup_{\\pmb\\mu,\\pmb\\delta}\\log\\mathcal{L}(\\pmb\\mu,\\pmb\\delta)\\right).


        Profiled maxima could be used both for frequentist inference and as initial condition for
        Markov Chain Monte Carlo sampling through the :class:`Sampler <DNNLikelihood.Sampler>` object
        (see the :mod:`Sampler <sampler>` object documentation). 
        The method uses the function 
        :func:`inference.compute_profiled_maximum_logpdf <DNNLikelihood.inference.compute_profiled_maximum_logpdf>`
        based on |scipy_optimize_minimize_link| to find the (local) minimum of minus
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. See the documentation of the
        :func:`inference.compute_profiled_maximum_logpdf <DNNLikelihood.inference.compute_profiled_maximum_logpdf>` 
        function for details.

        When using interactive python in Jupyter notebooks if ``progressbar=True`` then a progress bar is shown through 
        the |ipywidgets_link| package.

        - **Arguments**

            - **pars**
            
                List of positions of the parameters under which logpdf 
                is not profiled.

                    - **type**: ``list``
                    - **shape**: ``[ ]``
                    - **example**: ``[1,5,8]``

            - **pars_ranges**
            
                Ranges of the parameters ``pars``
                containing ``(min,max,n_points)``.

                    - **type**: ``list``
                    - **shape**: ``[[ ]]``
                    - **example**: ``[[0,1,5],[-1,1,5],[0,5,3]]``

            - **pars_init**
            
                Starting point for the optimization. If it is ``None``, then
                it is set to the parameters central value :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,)``
                    - **default**: ``None`` (automatically modified to :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`)

            - **pars_bounds**
            
                Bounds on the parameters. If it is ``None``, then default parameters bounds stored in the 
                :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` attribute are used.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,2)``
                    - **default**: ``None``

            - **spacing**
            
                It can be either ``"grid"`` or ``"random"``. Depending on its value the ``n_points`` for each parameter are taken on an 
                equally spaced grid or are generated randomly in the interval.

                    - **type**: ``str``
                    - **accepted**: ``"grid"`` or ``"random"``
                    - **default**: ``grid``

            - **optimizer**

                Dictionary containing information on the optimizer and its options.
                    
                    - **type**: ``dict`` with the following structure:

                        - *"name"* (value type: ``str``)
                          This is always set to ``"scipy"``, which is, by now, the only available optimizer for this task. 
                          As more optimizers will be supported the ``"name"`` key will indicate the chosen one.
                        - *"args"* (value type: ``str``)
                          Additional positional arguments passed to the |scipy_optimize_minimize_link| method.
                        - *"kwargs"* (value type: ``dict``)
                          Additional keyword arguments passed to the |scipy_optimize_minimize_link| method (set to the
                          ``{"method": "Powell"}`` dictionary by default).

                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            optimizer={"name": "scipy",
                                       "args": [],
                                       "kwargs": {"method": "Powell"}},

            - **minimization_options**

                Dictionary containing options to be passed to the |scipy_optimize_minimize_link| method 
                (i.e. value of the "options" keyword argument of the |scipy_optimize_minimize_link| method).
                    
                    - **type**: ``dict``
                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            minimization_options={"maxiter": 10000,
                                                  "ftol": 0.0001},

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **progressbar**
            
                If ``True`` 
                then  a progress bar is shown.
                    
                    - **type**: ``bool``
                    - **default**: ``False`` 

            - **save**
            
                If ``True`` the object is saved after the calculation.
                    
                    - **type**: ``bool``
                    - **default**: ``True``

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nComputing profiled maxima.\n",show=verbose)
        start = timer()
        iterator = 0
        overall_progress = None
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        if pars is None:
            raise Exception("The 'pars' input argument cannot be empty.")
        if pars_ranges is None:
            raise Exception("The 'pars_ranges' input argument cannot be empty.")
        if len(pars)!=len(pars_ranges):
            raise Exception("The input arguments 'pars' and 'pars_ranges' should have the same length.")
        pars_string = str(np.array(pars).tolist())
        if progressbar:
            try:
                import ipywidgets as widgets # type: ignore
                overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={"width": "500px", "height": "14px",
                                                                                              "padding": "0px", "margin": "-5px 0px -20px 0px"})
                display(overall_progress)
            except:
                progressbar = False
                print(header_string, "\nIf you want to show a progress bar please install the ipywidgets package.\n", show=verbose)            
        if progressbar:
            
            iterator = 0
        utils.check_set_dict_keys(self.predictions.logpdf_profiled_max,
                                  [timestamp],
                                  [{}], verbose=verbose_sub)
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy",[],{"method": "Powell"}],verbose=verbose_sub)
        if pars_init is None:
            pars_init = np.array(self.parameters.pars_central)
        else:
            pars_init = np.array(pars_init)
        pars_vals = utils.get_sorted_grid(pars_ranges=pars_ranges, spacing=spacing)
        print("Total number of points:", len(pars_vals),".",show=verbose)
        pars_vals_bounded = []
        if pars_bounds is None:
            for i in range(len(pars_vals)):
                if (np.all(pars_vals[i] >= self.parameters.pars_bounds[pars, 0]) and np.all(pars_vals[i] <= self.parameters.pars_bounds[pars, 1])):
                    pars_vals_bounded.append(pars_vals[i])
        else:
            pars_bounds=np.array(pars_bounds)
            if len(pars_bounds)!=len(pars_init):
                raise Exception("The specified parameter bounds do not match the number of parameters ("+str(len(pars_init))+").")
            for i in range(len(pars_vals)):
                if (np.all(pars_vals[i] >= pars_bounds[pars, 0]) and np.all(pars_vals[i] <= pars_bounds[pars, 1])):
                    pars_vals_bounded.append(pars_vals[i])
        if len(pars_vals) != len(pars_vals_bounded):
            print("Deleted", str(len(pars_vals)-len(pars_vals_bounded)),"points outside the parameters allowed range.",show=verbose)
        res = []
        try:
            optimization_times = self.predictions.logpdf_profiled_max[timestamp]["optimization_times"]
        except:
            optimization_times = []
        for pars_val in pars_vals_bounded:
            print("Optimizing for parameters:",pars," - values:",pars_val.tolist(),".",show=verbose)
            start_sub = timer()
            res.append(inference.compute_profiled_maximum_logpdf(logpdf=lambda x: self.logpdf_fn(x, *self.logpdf.args,*self.logpdf.kwargs),
                                                                 pars=pars, 
                                                                 pars_val=pars_val,
                                                                 ndims=self.parameters.ndims,
                                                                 pars_init=pars_init,
                                                                 pars_bounds=pars_bounds,
                                                                 optimizer=optimizer,
                                                                 minimization_options=minimization_options,
                                                                 verbose=verbose_sub))
            end_sub = timer()
            optimization_times.append(end_sub-start_sub)
            if progressbar and overall_progress is not None:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(pars_vals_bounded))
        X_tmp = np.array([x[0].tolist() for x in res])
        Y_tmp = np.array(res)[:, 1]
        self.predictions.logpdf_profiled_max[timestamp]["X"] = X_tmp
        self.predictions.logpdf_profiled_max[timestamp]["Y"] = Y_tmp
        print("Computing global maximum to estimate tmu test statistics.",show=verbose)
        self.compute_maximum_logpdf(pars_init=pars_init,
                                    optimizer=optimizer,
                                    minimization_options={},
                                    timestamp=timestamp,
                                    save=False,
                                    overwrite=False,
                                    verbose=False)
        self.predictions.logpdf_profiled_max[timestamp]["tmu"] = np.array(list(zip(X_tmp[:, pars].flatten(), -2*(Y_tmp-self.predictions.logpdf_max[timestamp]["y"]))))
        self.predictions.logpdf_profiled_max[timestamp]["pars"] = pars
        self.predictions.logpdf_profiled_max[timestamp]["pars_ranges"] = pars_ranges
        self.predictions.logpdf_profiled_max[timestamp]["pars_init"] = pars_init
        if pars_bounds is None:
            self.predictions.logpdf_profiled_max[timestamp]["pars_bounds"] = self.parameters.pars_bounds
        else:
            self.predictions.logpdf_profiled_max[timestamp]["pars_bounds"] = pars_bounds
        self.predictions.logpdf_profiled_max[timestamp]["optimizer"] = optimizer
        self.predictions.logpdf_profiled_max[timestamp]["minimization_options"] = minimization_options
        self.predictions.logpdf_profiled_max[timestamp]["optimization_times"] = optimization_times
        end = timer()
        self.predictions.logpdf_profiled_max[timestamp]["total_optimization_time"] = np.array(optimization_times).sum()
        self.log[timestamp] = {"action": "computed profiled maxima", 
                               "pars": pars,
                               "pars_ranges": pars_ranges, 
                               "number of maxima": len(X_tmp)}
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\n"+str(len(pars_vals_bounded)),"local maxima computed in", end-start, "s.",show=verbose)
        print("Log-pdf values lie in the range [", np.min(self.predictions.logpdf_profiled_max[timestamp]["Y"]), ",", np.max(self.predictions.logpdf_profiled_max[timestamp]["Y"]), "].\n", show=verbose)

    def plot_logpdf_par(self, 
                        pars=[[0,0,1]], 
                        npoints=100, 
                        pars_init=None, 
                        pars_labels="original", 
                        title_fontsize=12, 
                        show_plot=False,
                        timestamp=None,
                        save=True,
                        overwrite=True, 
                        verbose=None):
        """
        Plots the logpdf as a function of the parameter ``par`` in the range ``(min,max)``
        using a number ``npoints`` of points. Only the parameter ``par`` is veried, while all other parameters are kept
        fixed to their value given in ``pars_init``. The logpdf function used for the plot is provided by the 
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` method.

        - **Arguments**

            - **pars**
            
                List of lists containing the position of the parametes in the parameters vector, 
                and their minimum and maximum values for the plot.
                For example, to plot parameter ``1`` in the rage ``(1,3)`` and parameter ``5`` in the range
                ``(-3,3)`` one should set ``pars = [[1,1,3],[5,-3,3]]``. 

                    - **type**: ``list``
                    - **shape**: ``[[par,par_max,par_min],...]``
                    - **default**: ``[[0,0,1]]``

            - **npoints**
            
                Number of points in which the ``(par_min,par_max)`` range
                is divided to compute the logpdf and make the plot.

                    - **type**: ``int``
                    - **default**: ``100``

            - **pars_init**
            
                Central point in the parameter space from which ``par`` is varied and all other parameters are 
                kept fixed. When its value is the default ``None``, the attribute ``Lik.pars_central`` is used.

                    - **type**: ``numpy.ndarray`` or ``None``
                    - **shape**: ``(n_pars,)``
                    - **default**: ``None``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Lik.__set_pars_labels <DNNLikelihood.Lik._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plot.
                    
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"auto"``
                    - **default**: ``"original"``

            - **title_fontsize**
            
                Font size of the figure 
                title.
                    
                    - **type**: ``int``
                    - **default**: ``12``
            
            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **save**
            
                If ``True`` the object is saved after the calculation.
                    
                    - **type**: ``bool``
                    - **default**: ``True``

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>` ``+ "_par_" + str(par[0]) + ".pdf"`` for each ``par`` in ``pars``
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nMaking plot of logpdf as function of paramerers.\n",show=verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        plt.style.use(mplstyle_path)
        pars_labels = self.parameters.__set_pars_labels(pars_labels)
        if pars_init is None:
            pars_init = self.parameters.pars_central
        for par in pars:
            start = timer()
            par_number = par[0]
            par_min = par[1]
            par_max = par[2]
            vals = np.linspace(par_min, par_max, npoints)
            points = np.array(np.broadcast_to(pars_init,(npoints,len(pars_init))),dtype="float")
            points[:, par_number] = vals
            logpdf_vals = [self.logpdf_fn(point, *self.logpdf.args,*self.logpdf.kwargs) for point in points]
            plt.plot(vals, logpdf_vals)
            plt.title(r"%s" % self.name, fontsize=title_fontsize)
            plt.xlabel(r"%s" % pars_labels[par_number])
            plt.ylabel(r"logpdf")
            plt.tight_layout()
            figure_file_name = self.figures.update_figures(figure_file=Path(str(self.file_manager.output_figures_base_file_path) + "_par_"+str(par[0])+".pdf"),
                                                           file_manager=self.file_manager,
                                                           log=self.log,
                                                           timestamp=timestamp,
                                                           overwrite=overwrite) 
            Plots.savefig(self.file_manager.output_figures_folder.joinpath(figure_file_name))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            print(header_string+"\nFigure file\n\t",r"%s" % (path.join(self.output_figures_folder, figure_file_name)), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)
        
    def plot_tmu_1d(self,
                    timestamp_tmu=None,
                    pars_labels="original",
                    title_fontsize=12,
                    show_plot=False,
                    timestamp=None,
                    save=True,
                    overwrite=True,
                    verbose=None):
        """
        Plots the 1-dimensional :math:`t_{\\mu}` stored in 
        :attr:`Lik.predictions["logpdf_profiled_max"][timestamp]["tmu"] <DNNLikelihood.Lik.predictions>`.

        - **Arguments**

            - **timestamp_tmu**
            
                Timestamp idendifying the :math:`t_{\mu}` predictions to be plotted, stored in the 
                :attr:`Lik.predictions["logpdf_profiled_max"][timestamp]["tmu"] <DNNLikelihood.Lik.predictions>`
                attribute.
            
            - **pars_labels**
            
                Argument that is passed to the :meth:`Lik.__set_pars_labels <DNNLikelihood.Lik._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plot.
                    
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"auto"``
                    - **default**: ``"original"``

            - **title_fontsize**
            
                Font size of the figure 
                title.
                    
                    - **type**: ``int``
                    - **default**: ``12``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **save**
            
                If ``True`` the object is saved after the calculation.
                    
                    - **type**: ``bool``
                    - **default**: ``True``

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>` ``+ "_tmu_" + str(par) + ".pdf"``
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nMaking plot of t_mu test statistics as function of paramerers.\n",show=verbose)
        if timestamp_tmu is None:
            raise Exception("You need to specify the \"timestamp_tmu\" argument corresponding to the tmu prediction to be plotted.")
        start = timer()
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        plt.style.use(mplstyle_path)
        pars_labels = self.__set_pars_labels(pars_labels)
        pars_list = self.predictions.logpdf_profiled_max[timestamp_tmu]["pars"]
        tmu_list = self.predictions.logpdf_profiled_max[timestamp_tmu]["tmu"]
        if len(pars_list) == 1:
            par = pars_list[0]
        else:
            raise Exception("Parameters should be should be the same for the different tmu lists.")
        plt.plot(tmu_list[:, 0], tmu_list[:,-1], label="Likelihood")
        plt.title(r"%s" % self.name, fontsize=title_fontsize)
        plt.xlabel(r"$t_{\mu}$(%s)" % (self.pars_labels[par]))
        plt.ylabel(r"%s" % (self.pars_labels[par]))
        plt.legend()
        plt.tight_layout()
        figure_file_name = self.update_figures(self.output_figures_base_file_name + "_tmu_"+str(par) + ".pdf",timestamp=timestamp,overwrite=overwrite) 
        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print(header_string+"\nFigure file\n\t",r"%s" % (path.join(self.output_figures_folder, figure_file_name)), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)