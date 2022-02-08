__all__ = ["Lik"]

import builtins
import codecs
import h5py
import json
import time
from copy import copy
from datetime import datetime
from os import path, sep, stat
from timeit import default_timer as timer

import cloudpickle
import deepdish as dd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from . import inference, utils
from .show_prints import Verbosity, print
from .utils import _FunctionWrapper

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

header_string = "=============================="
footer_string = "------------------------------"

class Lik(Verbosity):
    """
    This class is a container for the :mod:`Likelihood <likelihood>` object, storing all information of the likelihood function.
    The object can be directly created or obtained from an ATLAS histfactory workspace through the 
    :class:`DNNLikelihood.Histfactory` object (see the :mod:`Histfactory <histfactory>` object documentation).
    """
    def __init__(self,
                 name = None,
                 logpdf = None,
                 logpdf_args = None,
                 logpdf_kwargs = None,
                 pars_central = None,
                 pars_pos_poi = None,
                 pars_pos_nuis = None,
                 pars_labels = None,
                 pars_bounds = None,
                 output_folder = None,
                 input_file = None,
                 verbose = True):
        """
        The :class:`Lik <DNNLikelihood.Lik>` object can be initialized in two different ways, depending on the value of 
        the :argument:`input_file` argument.

        - :argument:`input_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no :argument:`name` is given, 
            then one is created. The object is saved upon creation through the :meth:`Lik.save <DNNLikelihood.Lik.save>` method. 
        
        - :argument:`input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Lik.__load <DNNLikelihood.Lik._Lik__load>`
            Depending on the value of the input argument :argument:`output_folder` the 
            :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` method behaves as follows:

                - If :argument:`output_folder` is ``None`` (default) or is equal to :argument:`input_folder`
                    
                    The attribute :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`
                    is set equal to the :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>` one.

                - If :argument:`output_folder` is not ``None`` and is different than :argument:`input_folder`

                    The new :argument:`output_folder` is saved in the 
                    :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` attribute and files present in the 
                    input folder are copied to the new output folder, so that all previous results are preserved in the new path.

            The object is saved upon import through the :meth:`Lik.save <DNNLikelihood.Lik.save>` method.
        
        - **Arguments**

            See class :ref:`Arguments documentation <likelihood_arguments>`.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <Lik.Histfactory.output_h5_file>`
            - :attr:`Lik.output_json_file <Lik.Histfactory.output_json_file>`
            - :attr:`Lik.output_log_file <Lik.Histfactory.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <Lik.Histfactory.output_predictions_json_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        print(header_string,"\nInitialize Likelihood object.\n",show=verbose)
        self.output_folder = output_folder
        self.input_file = input_file
        self.__check_define_input_files()
        if self.input_file == None:
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            self.logpdf = _FunctionWrapper(logpdf, logpdf_args, logpdf_kwargs)
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_central = pars_central
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
            self.__check_define_pars()
            self.predictions = {"logpdf_max": {}, 
                                "logpdf_profiled_max": {},
                                "Figures": {}}
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.__load(verbose=verbose_sub)
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            try:
                self.predictions["logpdf_max"]
                self.predictions["logpdf_profiled_max"]
                self.predictions["Figures"]
            except:
                self.reset_predictions(delete_figures=True, save=False, verbose=verbose_sub)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            self.save(overwrite=True, verbose=verbose_sub)

    def __check_define_input_files(self,verbose=False):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to set the attributes corresponding to input files and folders
        
            - :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`
            - :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>`

        depending on the initial value of the 
        :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_file == None:
            self.input_h5_file = None
            self.input_log_file = None
            self.input_folder = None
            print(header_string,"\nNo Lik input files and folders specified.\n", show=verbose)
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_h5_file = self.input_file+".h5"
            self.input_log_file = self.input_file+".log"
            self.input_folder = path.split(self.input_file)[0]
            print(header_string,"\nLik input folder set to\n\t", self.input_folder,".\n",show=verbose)

    def __check_define_output_files(self,timestamp=None,verbose=False):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to set the attributes corresponding to output files and folders
        
            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>`
            - :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>`
            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`
            - :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <DNNLikelihood.Lik.output_predictions_json_file>`

        depending on the initial values of the 
        :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>` and
        :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` attributes.
        It also creates the output folder if it does not exist.

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.output_folder is not None:
            self.output_folder = path.abspath(self.output_folder)
            if self.input_folder is not None and self.output_folder != self.input_folder:
                utils.copy_and_save_folder(self.input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if self.input_folder is not None:
                self.output_folder = self.input_folder
            else:
                self.output_folder = path.abspath("")
        self.output_folder = utils.check_create_folder(self.output_folder)
        self.output_figures_folder =  utils.check_create_folder(path.join(self.output_folder, "figures"))
        self.output_h5_file = path.join(self.output_folder, self.name+".h5")
        self.output_json_file = path.join(self.output_folder, self.name+".json")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        self.output_predictions_json_file = path.join(self.output_folder, self.name+"_predictions.json")
        self.script_file = path.join(self.output_folder, self.name+"_script.py")
        self.output_figures_base_file_name = self.name+"_figure"
        self.output_figures_base_file_path = path.join(self.output_figures_folder, self.output_figures_base_file_name)
        print(header_string,"\nLik output folder set to\n\t", self.output_folder,".\n",show=verbose)

    def __check_define_name(self):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to define the :attr:`Lik.name <DNNLikelihood.Lik.name>` attribute.
        If the latter attribute is ``None``, then it is replaced with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_likelihood"``, otherwise 
        the suffix "_likelihood" is appended (preventing duplication if it is already present).  
        """
        if self.name == None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_likelihood"
        else:
            self.name = utils.check_add_suffix(self.name, "_likelihood")

    def __check_define_ndims(self):
        """
        Private method used by the :meth:`Lik.__check_define_pars <DNNLikelihood.Lik._Lik__check_define_pars>` one
        to define the :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>` attribute.
        To determine the number of dimensions it computes the logpdf, by calling the
        :meth:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` method on a vector of growing size
        until it does not generate an error.
        """
        check = True
        i = 1
        while check:
            try:
                self.logpdf(np.ones(i))
                check = False
            except:
                i = i+1
        self.ndims = i

    def __check_define_pars(self, verbose=None):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to check parameters consistency and set the attributes

            - :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>` 
            - :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` (converted into ``numpy.ndarray``)
            - :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>` (converted into ``numpy.ndarray``)
            - :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>`
            - :attr:`Lik.pars_labels_auto <DNNLikelihood.Lik.pars_labels_auto>`
            - :attr:`Lik.pars_pos_nuis <DNNLikelihood.Lik.pars_pos_nuis>` (converted into ``numpy.ndarray``)
            - :attr:`Lik.pars_pos_poi <DNNLikelihood.Lik.pars_pos_poi>` (converted into ``numpy.ndarray``)        

        If the attribute :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>` is ``None``,
        the method calls the :meth:`Lik.__check_define_ndims <DNNLikelihood.Lik._Lik__check_define_ndims>`
        method to determine the number of dimensions :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>`
        and sets :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`
        to a vector of zeros with length :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>`, warning the user that
        dimensions and parameters central values have been automatically determined.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>`
        is set to the value of :attr:`Lik.pars_labels_auto <DNNLikelihood.Lik.pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.pars_central is not None:
            self.pars_central = np.array(self.pars_central)
            self.ndims = len(self.pars_central)
        else:
            try:
                self.__check_define_ndims()
                self.pars_central = np.zeros(self.ndims)
                print(header_string,"\nNo central values for the parameters 'pars_central' has been specified. The number of dimensions \
                    have been automatically determined from 'logpdf' and the central values have been set to zero for all \
                    parameters. If they are known it is better to build the object providing parameters central values.\n", show=verbose)
            except:
                raise Exception("Impossible to determine the number of parameters/dimensions and the parameters central values. \
                    Please specify the input parameter 'pars_central'.")
        if self.pars_pos_nuis is not None and self.pars_pos_poi is not None:
            if len(self.pars_pos_poi)+len(self.pars_pos_nuis) == self.ndims:
                self.pars_pos_nuis = np.array(self.pars_pos_nuis)
                self.pars_pos_poi = np.array(self.pars_pos_poi)
            else:
                raise Exception("The number of parameters positions do not match the number of dimensions.")
        elif self.pars_pos_nuis is None and self.pars_pos_poi is None:
            print(header_string,"\nThe positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified.\
                Assuming all parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_nuis = np.array([])
            self.pars_pos_poi = np.array(list(range(self.ndims)))
        elif self.pars_pos_nuis is not None and self.pars_pos_poi is None:
            print(header_string,"\nOnly the positions of the nuisance parameters have been specified.\
                Assuming all other parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_poi = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_nuis))
        elif self.pars_pos_nuis is None and self.pars_pos_poi is not None:
            print(header_string,"\nOnly the positions of the parameters of interest.\
                Assuming all other parameters are nuisance parameters.\n", show=verbose)
            self.pars_pos_nuis = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_poi))
        self.pars_labels_auto = utils.define_pars_labels_auto(self.pars_pos_poi, self.pars_pos_nuis)
        if self.pars_labels is None:
            self.pars_labels = self.pars_labels_auto
        elif len(self.pars_labels) != self.ndims:
            raise Exception("The number of parameters labels do not match the number of dimensions.")
        if self.pars_bounds is not None:
            self.pars_bounds = np.array(self.pars_bounds)
        else:
            self.pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        if len(self.pars_bounds) != self.ndims:
            raise Exception("The length of the parameters bounds array does not match the number of dimensions.")

    def __load(self, verbose=None):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one 
        to load a previously saved
        :class:`Lik <DNNLikelihood.Lik>` object from the files 
        
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`

        The method loads, with the |deepdish_link| package, the content od the 
        :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
        file into a temporary dictionary, subsequently used to update the 
        :attr:`Lik.__dict__ <DNNLikelihood.Lik.__dict__>` attribute.
        The method also loads the content of the :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`
        file, assigning it to the :attr:`Lik.log <DNNLikelihood.Lik.log>` attribute.

        The ``"logpdf_dump"`` item of the loaded dictionary is a ``numpy.void`` object containing the a dump of the callable
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute produced by the |cloudpickle_link| package
        (see the documentation of the :attr:`Lik.save_h5<DNNLikelihood.Lik.save_h5>` method). This item is first
        converted into a (binary) string, then loaded with |cloudpickle_link| to reconstruct the
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` callable attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        dictionary = dd.io.load(self.input_h5_file)
        tmp = dictionary["logpdf_dump"]
        self.logpdf = cloudpickle.loads(tmp.tostring())
        dictionary.pop("logpdf_dump")
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_h5_file)[-1],
                                                path.split(self.input_log_file)[-1]]}
        print(header_string,"\nLikelihood object loaded in", str(end-start), ".\n",show=verbose)
        time.sleep(3)  # Removing this line prevents multiprocessing to work properly on Windows

    def __set_pars_labels(self, pars_labels):
        """
        Private method that returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"auto"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="auto"``
                the function returns the value of :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>`
                or :attr:`Lik.pars_labels_auto <DNNLikelihood.Lik.pars_labels_auto>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"auto"``
        """
        if pars_labels == "original":
            return self.pars_labels
        elif pars_labels == "auto":
            return self.pars_labels_auto
        else:
            return pars_labels

    def save_log(self, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Lik.log <DNNLikelihood.Lik.log>` attribute in the json file
        :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`.

        This method is called by the methods
        
        - :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` with ``overwrite=True`` and ``verbose=verbose_sub`` if :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>` is not ``None``, and with ``overwrite=True`` and ``verbose=verbose_sub`` otherwise.
        - :meth:`Lik.compute_maximum_logpdf <DNNLikelihood.Lik.compute_maximum_logpdf>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Lik.compute_profiled_maxima_logpdf <DNNLikelihood.Lik.compute_profiled_maxima_logpdf>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Lik.plot_logpdf_par <DNNLikelihood.Lik.plot_logpdf_par>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Lik.plot_tmu_1d <DNNLikelihood.Lik.plot_tmu_1d>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Lik.save <DNNLikelihood.Lik.save>` with ``overwrite=overwrite`` and ``verbose=verbose``
        - :meth:`Lik.save_script <DNNLikelihood.Lik.save_script>` with ``overwrite=True`` and ``verbose=verbose_sub``

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.
            
            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates file**

            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            if timestamp is None:
                timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            output_log_file = utils.generate_dump_file_name(self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nLikelihood log file\n\t", output_log_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nLikelihood log file\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nLikelihood log file dump\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions_json(self, timestamp=None,overwrite=False, verbose=None):
        """ Save predictions json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_json_file = self.output_predictions_json_file
            if not overwrite:
                utils.check_rename_file(output_predictions_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_json_file = utils.generate_dump_file_name(self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1]}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nPredictions json file dump\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_h5(self, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the :class:`Lik <DNNLikelihood.Lik>` object to the HDF5 file
        :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`.
        The object is saved by storing the content of the 
        :attr:``Lik.__dict__ <DNNLikelihood.Lik.__dict__>`` 
        attribute in a .h5 file using the |deepdish_link| package. 
        The following attributes are excluded from the saved dictionary:

            - :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`
            - :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>`
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`
            - :attr:`Lik.output_figures_base_file_name <DNNLikelihood.Lik.output_figures_base_file_name>`
            - :attr:`Lik.output_figures_base_file_path <DNNLikelihood.Lik.output_figures_base_file_path>`
            - :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>`
            - :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <DNNLikelihood.Lik.output_predictions_json_file>`
            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`
            - :attr:`Lik.log <DNNLikelihood.Lik.log>` (saved to the file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`)
            - :attr:`Lik.verbose <DNNLikelihood.Lik.verbose>`

        In order to save the callable :attr:`Lik.logpdf<DNNLikelihood.Lik.logpdf>` attribute, an item with key ``"logpdf_dump"``
        is added to the :attr:`Lik.__dict__ < DNNLikelihood.Lik.__dict__>` class dictionary. Such item contains as value 
        a ``numpy.void`` object created from a dump of the callable function produced with the |cloudpickle_link| package.

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_h5_file = self.output_h5_file
            if not overwrite:
                utils.check_rename_file(output_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_h5_file = utils.generate_dump_file_name(self.output_h5_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__, ["input_file", 
                                                          "input_folder",
                                                          "input_h5_file",
                                                          "input_log_file", 
                                                          "output_figures_base_file_name",
                                                          "output_figures_base_file_path",
                                                          "output_figures_folder",
                                                          "output_folder",
                                                          "output_h5_file",
                                                          "output_json_file",
                                                          "output_log_file",
                                                          "output_predictions_json_file",
                                                          "script_file",
                                                          "logpdf",
                                                          "log", 
                                                          "verbose"])
        dump = np.void(cloudpickle.dumps(self.logpdf))
        dictionary = {**dictionary, **{"logpdf_dump": dump}}
        dd.io.save(output_h5_file, dictionary)
        end = timer()
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(output_h5_file)[-1]}
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nLikelihood h5 file\n\t", output_h5_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nLikelihood h5 file\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nLikelihood h5 file dump\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_json(self, timestamp=None, overwrite=False, verbose=None):
        """
        Part of the :class:`Lik <DNNLikelihood.Lik>` object is also saved to the human
        readable json file :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`.

        The object is saved by storing all json serializable attributes obtained from the
        :attr:``Lik.__dict__ <DNNLikelihood.Lik.__dict__>`` 
        attribute. The following attributes are excluded from the saved dictionary:

            - :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`
            - :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>`
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`
            - :attr:`Lik.output_figures_base_file_name <DNNLikelihood.Lik.output_figures_base_file_name>`
            - :attr:`Lik.output_figures_base_file_path <DNNLikelihood.Lik.output_figures_base_file_path>`
            - :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>`
            - :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <DNNLikelihood.Lik.output_predictions_json_file>`
            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`
            - :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>`
            - :attr:`Lik.log <DNNLikelihood.Lik.log>` (saved to the file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`)
            - :attr:`Lik.verbose <DNNLikelihood.Lik.verbose>`

        - **Arguments**

            Same arguments of the :meth:`Lik.save_h5 <DNNLikelihood.Lik.save_h5>` method.

        - **Creates/updates files**

            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_json_file = self.output_json_file
            if not overwrite:
                utils.check_rename_file(output_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_json_file = utils.generate_dump_file_name(
                self.output_json_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__, ["input_file", 
                                                          "input_folder",
                                                          "input_h5_file",
                                                          "input_log_file", 
                                                          "output_figures_base_file_name",
                                                          "output_figures_base_file_path",
                                                          "output_figures_folder",
                                                          "output_folder",
                                                          "output_h5_file",
                                                          "output_json_file",
                                                          "output_log_file",
                                                          "output_predictions_json_file",
                                                          "script_file",
                                                          "logpdf", 
                                                          "log", 
                                                          "verbose"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved object json",
                               "file name": path.split(output_json_file)[-1]}
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nLikelihood json file\n\t", output_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nLikelihood json file\n\t", output_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nLikelihood json file dump\n\t", output_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save(self, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the :class:`Lik <DNNLikelihood.Lik>` object by calling, in the following order,
        the four methods:
        
            - :meth:`Lik.save_json <DNNLikelihood.Lik.save_json>`
            - :meth:`Lik.save_h5 <DNNLikelihood.Lik.save_h5>`
            - :meth:`Lik.save_predictions_json <DNNLikelihood.Lik.save_json>`
            - :meth:`Lik.save_log <DNNLikelihood.Lik.save_log>`

        The :class:`Lik <DNNLikelihood.Lik>` object is saved to three files: an HDF5 compressed file
        used to import back the object, a human-readable json file including json serializable
        attributes, and a log file including the content of the :attr:`Lik.log <DNNLikelihood.Lik.log>`
        attribute.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <Lik.Histfactory.output_h5_file>`
            - :attr:`Lik.output_json_file <Lik.Histfactory.output_json_file>`
            - :attr:`Lik.output_log_file <Lik.Histfactory.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <Lik.Histfactory.output_predictions_json_file>`
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_json(timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_h5(timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_predictions_json(timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_log(timestamp=timestamp, overwrite=overwrite, verbose=verbose)

    def save_script(self, timestamp=None, overwrite=False, verbose=True):
        """
        Saves the file :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`. 

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.
                    
            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates file**

            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_script_file = self.script_file
            if not overwrite:
                utils.check_rename_file(output_script_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_script_file = utils.generate_dump_file_name(self.script_file, timestamp=timestamp)
        with open(output_script_file, "w") as out_file:
            out_file.write("import DNNLikelihood\n"+
                           "import numpy as np\n" + "\n" +
                           "lik = DNNLikelihood.Lik(name=None,\n" +
                           "\tinput_file="+r"'" + r"%s" % ((self.output_h5_file).replace(sep, '/'))+"', \n"+
                           "verbose = "+str(self.verbose)+")"+"\n"+"\n" +
                           "name = lik.name\n" +
                           "def logpdf(x_pars,*args,**kwargs):\n" +
                           "\treturn lik.logpdf_fn(x_pars,*args,**kwargs)\n" +
                           "logpdf_args = lik.logpdf.args\n" +
                           "logpdf_kwargs = lik.logpdf.kwargs\n" +
                           "pars_pos_poi = lik.pars_pos_poi\n" +
                           "pars_pos_nuis = lik.pars_pos_nuis\n" +
                           "pars_central = lik.pars_central\n" +
                           "try:\n" +
                           "\tpars_init_vec = lik.predictions['logpdf_profiled_max']['%s']['X']\n"%timestamp +
                           "except:\n" +
                           "\tpars_init_vec = None\n"
                           "pars_labels = lik.pars_labels\n" +
                           "pars_bounds = lik.pars_bounds\n" +
                           "ndims = lik.ndims\n" +
                           "output_folder = lik.output_folder"
                           )
        end = timer()
        self.log[timestamp] = {"action": "saved", 
                               "file name": path.split(output_script_file)[-1]}
        self.save_log(overwrite=True, verbose=verbose_sub)
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nLikelihood script file\n\t", output_script_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nLikelihood script file\n\t", output_script_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nLikelihood script file dump\n\t", output_script_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def logpdf_fn(self, x_pars, *args, **kwargs):
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
        logpdf.args = args
        logpdf.kwargs = kwargs
        if len(np.shape(x_pars)) == 1:
            if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                return -np.inf
            tmp = logpdf(x_pars)
            if type(tmp) == np.ndarray or type(tmp) == list:
                tmp = tmp[0]
            if np.isnan(tmp):
                tmp = -np.inf
            return tmp
        else:
            x_pars_list = x_pars
            tmp = logpdf(x_pars)
            for i in range(len(x_pars_list)):
                x_pars = x_pars_list[i]
                if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                    tmp[i] = -np.inf
            tmp = np.where(np.isnan(tmp), -np.inf, tmp)
            return tmp

    def reset_predictions(self, 
                          delete_figures=False, 
                          save=False,
                          overwrite=True,
                          verbose=None):
        """
        Re-initializes the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary to

        .. code-block:: python

            predictions = {"logpdf_max": {},
                           "logpdf_profiled_max": {},
                           "Figures": figs}

        Where ``figs`` may be either an empty dictionary or the present value of the corresponding one,
        depending on the value of the ``delete_figures`` argument.

        - **Arguments**

            - **delete_figures**
            
                If ``True`` all files in the :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>` 
                folder are deleted and the ``"Figures"`` item is reset to an empty dictionary.
                    
                    - **type**: ``bool``
                    - **default**: ``True``

            - **save**
            
                If ``True`` the object is saved after resetting the dictionary.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

         - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_predictions_json_file>` (only if ``save=True``)
            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_predictions_json_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <DNNLikelihood.Lik.output_predictions_json_file>` (only if ``save=True``)

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nResetting predictions.\n",show=verbose)
        start = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if delete_figures:
            utils.check_delete_all_files_in_path(self.output_figures_folder)
            figs = {}
            print(header_string,"\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n")
        else:
            figs = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            print(header_string,"\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n")
        self.predictions = {"logpdf_max": {},
                            "logpdf_profiled_max": {},
                            "Figures": figs}
        end = timer()
        self.log[timestamp] = {"action": "reset predictions"}
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nPredictions reset in", end-start, "s.\n")
        
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
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy",[],{"method": "Powell"}],verbose=verbose_sub)
        if pars_init is None:
            pars_init = np.array(self.pars_central)
        else:
            pars_init = np.array(pars_init)
        utils.check_set_dict_keys(self.predictions["logpdf_max"],
                                  [timestamp],
                                  [{}], verbose=False)
        res = inference.compute_maximum_logpdf(logpdf=lambda x: self.logpdf_fn(x,*self.logpdf.args, *self.logpdf.kwargs), 
                                               pars_init=pars_init,
                                               pars_bounds=pars_bounds,
                                               optimizer=optimizer,
                                               minimization_options=minimization_options,
                                               verbose=verbose_sub)
        self.predictions["logpdf_max"][timestamp]["x"], self.predictions["logpdf_max"][timestamp]["y"] = res
        end = timer()
        self.predictions["logpdf_max"][timestamp]["pars_init"] = pars_init
        if pars_bounds is None:
            self.predictions["logpdf_max"][timestamp]["pars_bounds"] = self.pars_bounds
        else:
            self.predictions["logpdf_max"][timestamp]["pars_bounds"] = pars_bounds
        self.predictions["logpdf_max"][timestamp]["optimizer"] = optimizer
        self.predictions["logpdf_max"][timestamp]["minimization_options"] = minimization_options
        self.predictions["logpdf_max"][timestamp]["optimization_time"] = end-start
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
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if pars is None:
            raise Exception("The 'pars' input argument cannot be empty.")
        if pars_ranges is None:
            raise Exception("The 'pars_ranges' input argument cannot be empty.")
        if len(pars)!=len(pars_ranges):
            raise Exception("The input arguments 'pars' and 'pars_ranges' should have the same length.")
        pars_string = str(np.array(pars).tolist())
        if progressbar:
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                print("\nIf you want to show a progress bar please install the ipywidgets package.\n",show=True)
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                "width": "500px", "height": "14px",
                "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
        utils.check_set_dict_keys(self.predictions["logpdf_profiled_max"],
                                  [timestamp],
                                  [{}], verbose=verbose_sub)
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy",[],{"method": "Powell"}],verbose=verbose_sub)
        if pars_init is None:
            pars_init = np.array(self.pars_central)
        else:
            pars_init = np.array(pars_init)
        pars_vals = utils.get_sorted_grid(pars_ranges=pars_ranges, spacing=spacing)
        print("Total number of points:", len(pars_vals),".",show=verbose)
        pars_vals_bounded = []
        if pars_bounds is None:
            for i in range(len(pars_vals)):
                if (np.all(pars_vals[i] >= self.pars_bounds[pars, 0]) and np.all(pars_vals[i] <= self.pars_bounds[pars, 1])):
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
            optimization_times = self.predictions["logpdf_profiled_max"][timestamp]["optimization_times"]
        except:
            optimization_times = []
        for pars_val in pars_vals_bounded:
            print("Optimizing for parameters:",pars," - values:",pars_val.tolist(),".",show=verbose)
            start_sub = timer()
            res.append(inference.compute_profiled_maximum_logpdf(logpdf=lambda x: self.logpdf_fn(x, *self.logpdf.args,*self.logpdf.kwargs),
                                                                 pars=pars, 
                                                                 pars_val=pars_val,
                                                                 ndims=self.ndims,
                                                                 pars_init=pars_init,
                                                                 pars_bounds=pars_bounds,
                                                                 optimizer=optimizer,
                                                                 minimization_options=minimization_options,
                                                                 verbose=verbose_sub))
            end_sub = timer()
            optimization_times.append(end_sub-start_sub)
            if progressbar:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(pars_vals_bounded))
        X_tmp = np.array([x[0].tolist() for x in res])
        Y_tmp = np.array(res)[:, 1]
        self.predictions["logpdf_profiled_max"][timestamp]["X"] = X_tmp
        self.predictions["logpdf_profiled_max"][timestamp]["Y"] = Y_tmp
        print("Computing global maximum to estimate tmu test statistics.",show=verbose)
        self.compute_maximum_logpdf(pars_init=pars_init,
                                    optimizer=optimizer,
                                    minimization_options={},
                                    timestamp=timestamp,
                                    save=False,
                                    overwrite=False,
                                    verbose=False)
        self.predictions["logpdf_profiled_max"][timestamp]["tmu"] = np.array(list(zip(X_tmp[:, pars].flatten(), -2*(Y_tmp-self.predictions["logpdf_max"][timestamp]["y"]))))
        self.predictions["logpdf_profiled_max"][timestamp]["pars"] = pars
        self.predictions["logpdf_profiled_max"][timestamp]["pars_ranges"] = pars_ranges
        self.predictions["logpdf_profiled_max"][timestamp]["pars_init"] = pars_init
        if pars_bounds is None:
            self.predictions["logpdf_profiled_max"][timestamp]["pars_bounds"] = self.pars_bounds
        else:
            self.predictions["logpdf_profiled_max"][timestamp]["pars_bounds"] = pars_bounds
        self.predictions["logpdf_profiled_max"][timestamp]["optimizer"] = optimizer
        self.predictions["logpdf_profiled_max"][timestamp]["minimization_options"] = minimization_options
        self.predictions["logpdf_profiled_max"][timestamp]["optimization_times"] = optimization_times
        end = timer()
        self.predictions["logpdf_profiled_max"][timestamp]["total_optimization_time"] = np.array(optimization_times).sum()
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
        print("Log-pdf values lie in the range [", np.min(self.predictions["logpdf_profiled_max"][timestamp]["Y"]), ",", np.max(self.predictions["logpdf_profiled_max"][timestamp]["Y"]), "].\n", show=verbose)
        
    def update_figures(self,
                       figure_file=None,
                       timestamp=None,
                       overwrite=False,
                       verbose=None):
        """
        Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
        When ``overwrite=False`` it calls the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function and, if 
        ``figure_file`` already existed in the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary, then it
        updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
        (that is the key of the :attr:`Lik.predictions["Figures"] <DNNLikelihood.Lik.predictions>` dictionary).
        When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_name <DNNLikelihood.utils.generate_dump_file_name>` function
        to generate the dump file name.
        It returns the new figure file name.

        - **Arguments**

            - **figure_file**

                Figure file path. If the figure already exists in the 
                :meth:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary, then its name is updated with the corresponding timestamp.

            - **overwrite**

                The method updates file names and :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary only if
                ``overwrite=False``. If ``overwrite="dump"`` the method generates and returns the dump file path. 
                If ``overwrite=True`` the method just returns ``figure_file``.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        
        - **Returns**

            - **new_figure_file**
                
                String identical to the input string ``figure_file`` unless ``overwrite="dump"``.

        - **Creates/updates files**

            - Updates ``figure_file`` file.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nChecking and updating figures dictionary,\n",show=verbose)
        if figure_file is None:
            raise Exception("figure_file input argument of update_figures method needs to be specified while it is None.")
        else:
            new_figure_file = figure_file
            if type(overwrite) == bool:
                if not overwrite:
                    # search figure
                    timestamp=None
                    for k, v in self.predictions["Figures"].items():
                        if figure_file in v:
                            timestamp = k
                    old_figure_file = utils.check_rename_file(path.join(self.output_figures_folder,figure_file),timestamp=timestamp,return_value="file_name",verbose=verbose_sub)
                    if timestamp is not None:
                        self.predictions["Figures"][timestamp] = [f.replace(figure_file,old_figure_file) for f in v] 
                        self.save(overwrite=True, verbose=verbose_sub)
            elif overwrite == "dump":
                new_figure_file = utils.generate_dump_file_name(figure_file, timestamp=timestamp)
        self.log[timestamp] = {"action": "checked/updated figures dictionary",
                               "figure_file": figure_file,
                               "new_figure_file": new_figure_file}
        #self.save_log(overwrite=True, verbose=verbose_sub)
        return new_figure_file

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
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars_labels = self.__set_pars_labels(pars_labels)
        if pars_init == None:
            pars_init = self.pars_central
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
            figure_file_name = self.update_figures(self.output_figures_base_file_name + "_par_"+str(par[0])+".pdf",timestamp=timestamp,overwrite=overwrite) 
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
            timestamp = "datetime_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars_labels = self.__set_pars_labels(pars_labels)
        pars_list = self.predictions["logpdf_profiled_max"][timestamp_tmu]["pars"]
        tmu_list = self.predictions["logpdf_profiled_max"][timestamp_tmu]["tmu"]
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