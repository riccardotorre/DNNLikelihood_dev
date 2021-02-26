__all__ = ["Lik"]

import builtins
import codecs
import h5py
import json
import time
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
            then one is created. 
            The object is saved upon creation through the :meth:`Lik.save <DNNLikelihood.Lik.save>` method. 
        
        - :argument:`input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Lik.__load <DNNLikelihood.Lik._Lik__load>`
            Depending on the value of the input argument :argument:`output_folder` the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` method behaves as follows:

                - If :argument:`output_folder` is ``None`` (default)
                    
                    The attribute :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
                    is set from the :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>` one.
                - If :argument:`output_folder` corresponds to a path different from that stored in the loaded :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute
                    
                    - if path stored in the loaded :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute exists, then its content is copied to the new ``output_folder`` (if the new ``output_foler`` already exists it is renamed by adding a timestamp);
                    - if path stored in the loaded :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` does not exists, then the content of the path :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>` is copied to the new ``output_folder``.
                - If :argument:`output_folder` corresponds to the same path stored in the loaded :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute
                    
                    Output folder, files, and path attributes are not updated and everything is read from the loaded object.
        
        - **Arguments**

            See class :ref:`Arguments documentation <likelihood_arguments>`.

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only the first time the object is created, i.e. if :argument:`input_file` is ``None``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.input_file = input_file
        self.__check_define_input_files()
        if self.input_file == None:
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.logpdf = _FunctionWrapper(logpdf, logpdf_args, logpdf_kwargs)
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_central = pars_central
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
            self.__check_define_pars()
            self.output_folder = output_folder
            self.__check_define_output_files()
            self.predictions = {"logpdf_max": {}, 
                                "logpdf_profiled_max": {},
                                "Figures": {}}
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.__load(verbose=verbose_sub)
            if output_folder != None:
                old_output_folder = self.output_folder
                if old_output_folder != path.abspath(output_folder):
                    self.output_folder = path.abspath(output_folder)
                    try:
                        utils.copy_and_save_folder(old_output_folder, self.output_folder, timestamp=timestamp,verbose=verbose)
                        print(header_string,"\nThe output folder was changed from\n\t",old_output_folder,"\nto\n\t",self.output_folder,"\nThe old folder has been found, all its content has been copied to the new output folder. All (output) path attributes have been updated accordingly.\n",show=verbose)
                    except:
                        utils.copy_and_save_folder(self.input_folder, self.output_folder, timestamp=timestamp,verbose=verbose)
                        print(header_string,"\nThe output folder was changed from\n\t",old_output_folder,"\nto\n\t",self.output_folder,"\nThe old folder has not been found. The content of the present input folder has been copied to the new output folder. All (output) path attributes have been updated accordingly.\n",show=verbose)
                    self.__check_define_output_files()
                    self.log[timestamp] = {"action": "changed_output_folder", 
                                           "old folder": old_output_folder,
                                           "new folder": self.output_folder}
                self.save(overwrite=False, verbose=verbose_sub)
            else:
                self.save_log(overwrite=True, verbose=verbose_sub)
            try:
                self.predictions["logpdf_max"]
                self.predictions["logpdf_profiled_max"]
                self.predictions["Figures"]
            except:
                self.reset_predictions(delete_figures=True, verbose=verbose_sub)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)

    def __check_define_input_files(self):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to set the attributes corresponding to input files
        
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`,
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`,
            - :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>`

        depending on the value of the 
        :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>` attribute.
        """
        if self.input_file == None:
            self.input_h5_file = None
            self.input_log_file = None
            self.input_folder = None
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_h5_file = self.input_file+".h5"
            self.input_log_file = self.input_file+".log"
            self.input_folder = path.split(self.input_file)[0]

    def __check_define_output_files(self):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to set the attributes corresponding to output folders
        
            - :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>`
            - :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`

        and output files

            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>`
            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
            - :attr:`Lik.output_predictions_json_file <DNNLikelihood.Lik.output_predictions_json_file>`
            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`

        depending on the value of the
        :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>` attribute.
        It also creates the folders
        :attr:`Lik.output_folder <DNNLikelihood.Lik.output_folder>`
        and :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>` if 
        they do not exist.
        """
        if self.output_folder == None:
            self.output_folder = ""
        self.output_folder = utils.check_create_folder(path.abspath(self.output_folder))
        self.output_figures_folder =  utils.check_create_folder(path.join(self.output_folder, "figures"))
        self.output_h5_file = path.join(self.output_folder, self.name+".h5")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        self.output_predictions_json_file = path.join(self.output_folder, self.name+"_predictions.json")
        self.script_file = path.join(self.output_folder, self.name+"_script.py")
        self.output_figures_base_file = path.join(self.output_figures_folder, self.name+"_figure")

    def __check_define_name(self):
        """
        Private method used by the :meth:`Lik.__init__ <DNNLikelihood.Lik.__init__>` one
        to define the :attr:`Lik.name <DNNLikelihood.Lik.name>` attribute.
        If it is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_likelihood"``,
        otherwise it appends the suffix "_likelihood" (preventing duplication if it is already present).
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
        :meth:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` passing it a vector of growing size
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
        to check the consistency and set the attributes

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
        dimensiona and parameters central values have been automatically determined.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Lik.pars_labels <DNNLikelihood.Lik.pars_labels>`
        is set to the value of :attr:`Lik.pars_labels_auto <DNNLikelihood.Lik.pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Lik.ndims <DNNLikelihood.Lik.ndims>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.pars_central is not None:
            self.pars_central = np.array(self.pars_central)
            self.ndims = len(self.pars_central)
        else:
            try:
                self.__check_define_ndims()
                self.pars_central = np.zeros(self.ndims)
                print("No central values for the parameters 'pars_central' has been specified. The number of dimensions \
                    have been automatically determined from 'logpdf' and the central values have been set to zero for all \
                    parameters. If they are known it is better to build the object providing parameters central values.", show=verbose)
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
            print("The positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified.\
                Assuming all parameters are parameters of interest.", show=verbose)
            self.pars_pos_nuis = np.array([])
            self.pars_pos_poi = np.array(list(range(self.ndims)))
        elif self.pars_pos_nuis is not None and self.pars_pos_poi is None:
            print("Only the positions of the nuisance parameters have been specified.\
                Assuming all other parameters are parameters of interest.", show=verbose)
            self.pars_pos_poi = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_nuis))
        elif self.pars_pos_nuis is None and self.pars_pos_poi is not None:
            print("Only the positions of the parameters of interest.\
                Assuming all other parameters are nuisance parameters.", show=verbose)
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
        :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>` file into a temporary dictionary, subsequently used to update the 
        :attr:`Histfactory.__dict__ <DNNLikelihood.Histfactory.__dict__>` attribute.
        The ``"logpdf_dump"`` item of the loaded dictionary is a ``numpy.void`` object containing the a dump of the callable
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute produced by the |cloudpickle_link| package
        (see the documentation of the :attr:`Lik.save <DNNLikelihood.Lik.save>` method). This item is first
        converted into a (binary) string, then loaded with |cloudpickle_link| to reconstruct the 
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute.
        The method also loads the content of the :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`
        file, assigning it to the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` attribute.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
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
                                                path.split(self.input_log_file)[-1]],
                               "files paths": [self.input_h5_file,
                                               self.input_log_file]}
        print('Likelihood object loaded in', str(end-start), '.',show=verbose)
        time.sleep(3)  # Removing this line prevents multiprocessing to work properly on Windows

    def __set_pars_labels(self, pars_labels):
        """
        Private method that returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"autp"`` or a list of labels
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
        Saves the content of the :attr:`Lik.log <DNNLikelihood.Lik.log>` attribute in the file
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
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is passed to the
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function
                to generate the dump file name.
                    
                    - **type**: ``str``
                    - **default**: ``None``
            
            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates file**

            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_log_file = utils.generate_dump_file_path(self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print("Likelihood log file", output_log_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Likelihood log file", output_log_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Likelihood log file dump", output_log_file, "saved in", str(end-start), "s.", show=verbose)

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
            output_predictions_json_file = utils.generate_dump_file_path(self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1],
                               "file path": output_predictions_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Predictions json file", output_predictions_json_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Predictions json file", output_predictions_json_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Predictions json file dump", output_predictions_json_file, "saved in", str(end-start), "s.", show=verbose)

    def save(self, timestamp=None, overwrite=False, verbose=None):
        """
        The :class:`Lik <DNNLikelihood.Lik>` object is saved to the HDF5 file
        :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` and the object log is saved
        to the json file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`.
        The object is saved by storing the content of the :attr:``Lik.__dict__ <DNNLikelihood.Lik.__dict__>`` 
        attribute in an h5 file using the |deepdish_link| package.
        In order to be able to store the callable :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute, this is first
        serialized to a byte string using the |cloudpickle_link| package, and then encoded into a ``numpy.void`` object.
        This is stored in a dedicated ``"logpdf_dump"`` key of the object dictionary and is used to restore the 
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` attribute when the object is loaded (see the documentation of the 
        :meth:`Lik.__load <DNNLikelihood.Lik._Lik__load>` method). The following attributes are excluded from the saved
        dictionary:

            - :attr:`Lik.input_file <DNNLikelihood.Lik.input_file>`
            - :attr:`Lik.input_h5_file <DNNLikelihood.Lik.input_h5_file>`
            - :attr:`Lik.input_log_file <DNNLikelihood.Lik.input_log_file>`
            - :attr:`Lik.input_folder <DNNLikelihood.Lik.input_folder>`
            - :attr:`Lik.log <DNNLikelihood.Lik.log>` (saved to the file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`)
            - :attr:`Lik.verbose <DNNLikelihood.Lik.verbose>`

        - **Arguments**

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used to update the 
                :attr:`Lik.log <DNNLikelihood.Lik.log>` object and, when ``overwrite="dump"``, is passed to the
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function
                to generate the dump file name.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if type(overwrite) == bool:
            output_h5_file = self.output_h5_file
            if not overwrite:
                utils.check_rename_file(output_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_h5_file = utils.generate_dump_file_path(self.output_h5_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__, ["input_file", "input_h5_file",
                                                          "input_log_file", "input_folder",
                                                          "logpdf", "log", "verbose"])
        dump = np.void(cloudpickle.dumps(self.logpdf))
        dictionary = {**dictionary, **{"logpdf_dump": dump}}
        dd.io.save(output_h5_file, dictionary)
        end = timer()
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(output_h5_file)[-1],
                               "file path": output_h5_file}
        self.save_predictions_json(timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub)
        self.save_log(overwrite=overwrite, verbose=verbose_sub)
        if type(overwrite) == bool:
            if overwrite:
                print("Likelihood h5 file", output_h5_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Likelihood h5 file", output_h5_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Likelihood h5 file dump", output_h5_file, "saved in", str(end-start), "s.", show=verbose)

    def save_script(self, timestamp=None, overwrite=True, verbose=True):
        """
        Saves the file :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`. 

        - **Arguments**

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used to update the 
                :attr:`Lik.log <DNNLikelihood.Lik.log>` object and, when ``overwrite="dump"``, is passed to the
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function
                to generate the dump file name.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``
                    
            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates file**

            - :attr:`Lik.script_file <DNNLikelihood.Lik.script_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if type(overwrite) == bool:
            output_script_file = self.script_file
            if not overwrite:
                utils.check_rename_file(output_script_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_script_file = utils.generate_dump_file_path(self.script_file, timestamp=timestamp)
        with open(output_script_file, "w") as out_file:
            out_file.write("import DNNLikelihood\n"+
                           "import numpy as np\n" + "\n" +
                           "lik = DNNLikelihood.Lik(name=None,\n" +
                           "\tinput_file="+r"'" + r"%s" % ((self.output_h5_file).replace(sep, '/'))+"', \n"+
                           "verbose = "+str(self.verbose)+")"+"\n"+"\n" +
                           "name = lik.name\n" +
                           "def logpdf(x_pars,*args,**kwargs):\n" +
                           "\treturn lik.logpdf_fn(x_pars,*args,**kwargs)\n" +
                           #"logpdf = lik.logpdf_fn\n" +
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
                               "file name": path.split(output_script_file)[-1], 
                               "file path": output_script_file}
        self.save_log(overwrite=True, verbose=verbose_sub)
        if type(overwrite) == bool:
            if overwrite:
                print("Likelihood script file", output_script_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Likelihood script file", output_script_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Likelihood script file dump", output_script_file, "saved in", str(end-start), "s.", show=verbose)

    def logpdf_fn(self, x_pars, *args, **kwargs):
        """
        This function is used to add constraints and standardize input/output of 
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>`.
        It is constructed from the :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` and 
        :attr:`Lik.logpdf_args <DNNLikelihood.Lik.logpdf_args>` attributes. 
        In the case :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` accepts a single array of 
        parameters ``x_pars`` and returns the logpdf value one point at a time, then the function returns a ``float``, 
        while if :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` is vectorized,
        i.e. accepts an array of ``x_pars`` arrays and returns an array of logpdf values, then the function returns an array. 
        Moreover, the :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>` method is constructed to return the
        logpdf value ``-np.inf`` if any of the parameters lies outside 
        :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` or the 
        :attr:`Lik.logpdf <DNNLikelihood.Lik.logpdf>` function returns ``nan``.

        - **Arguments**

            - **x_pars**

                Values of the parameters for which logpdf is computed.
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
        self.logpdf.args = args
        self.logpdf.kwargs = kwargs
        if len(np.shape(x_pars)) == 1:
            if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                return -np.inf
            tmp = self.logpdf(x_pars)
            if type(tmp) == np.ndarray or type(tmp) == list:
                tmp = tmp[0]
            if np.isnan(tmp):
                tmp = -np.inf
            return tmp
        else:
            x_pars_list = x_pars
            tmp = self.logpdf(x_pars)
            for i in range(len(x_pars_list)):
                x_pars = x_pars_list[i]
                if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                    tmp[i] = -np.inf
            tmp = np.where(np.isnan(tmp), -np.inf, tmp)
            return tmp

    def reset_predictions(self, 
                          delete_figures=False, 
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
            
            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if delete_figures:
            utils.check_delete_all_files_in_path(self.output_figures_folder)
            figs = {}
            print("All predictions and figures have been deleted and the 'predictions' attribute has been initialized.")
        else:
            figs = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            print("All predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.")
        self.predictions = {"logpdf_max": {},
                            "logpdf_profiled_max": {},
                            "Figures": figs}

    def compute_maximum_logpdf(self,
                               pars_init=None,
                               optimizer={},
                               timestamp = None,
                               save=False,
                               overwrite=False,
                               verbose=None):
        """
        Computes the maximum of :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. 
        All information on the maximum, including parameters initialization, parameters bounds, and optimizer, 
        are stored in the ``"logpdf_max"`` item of the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary.
        The method uses the function :func:`inference.compute_maximum_logpdf <DNNLikelihood.inference.compute_maximum_logpdf>`
        based on |scipy_optimize_minimize_link| to find the minimum of minus 
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. Since the latter
        method already contains a bounded logpdf, ``pars_bounds`` is set to ``None`` in the 
        :func:`inference.compute_maximum_logpdf <DNNLikelihood.inference.compute_maximum_logpdf>`
        function to optimize speed. See the documentation of the
        :func:`inference.compute_maximum_logpdf <DNNLikelihood.inference.compute_maximum_logpdf>` 
        function for details.

        - **Arguments**

            - **pars_init**
            
                Starting point for the optimization. If it is ``None``, then
                it is set to the parameters central value :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,)``
                    - **default**: ``None`` (automatically modified to :attr:`Lik.pars_central <DNNLikelihood.Lik.pars_central>`)

            - **optimizer**

                Dictionary containing information on the optimizer and its options.
                    
                    - **type**: ``dict`` with the following structure:

                        - *"name"* (value type: ``str``)
                          This is always set to ``"scipy"``, which is by now the only available optimizer for this task. As more
                          optimizers will be supported the ``"name"`` key will indicate the chosen one.
                        - *"method"* (value type: ``str``)
                          Optimizer method. For the |scipy_optimize_minimize_link| optimizer the default method is given by ``"Powell"``.
                        - *"options"* (value type: ``dict``)
                          Dictionary containing additional options to pass to the |scipy_optimize_minimize_link| function.

                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            optimizer={"name": "scipy",
                                       "method": "Powell",
                                       "options": {"maxiter": 10000,
                                                   "ftol": 0.001}}

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used as key in the 
                :attr:`Lik.predictions["logpdf_max"] <DNNLikelihood.Lik.predictions>` dictionary to save the current prediction.
                It is also used to update the :attr:`Lik.log <DNNLikelihood.Lik.log>` dictionary and to save the object when 
                ``overwrite="dump"``.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **save**
            
                If ``True`` the object is saved (by passing the ``overwrite`` argument to the
                :meth:`Lik.save <DNNLikelihood.Lik.save>` method).
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        utils.check_set_dict_keys(optimizer, ["name",
                                              "method",
                                              "options"],
                                             ["scipy", "Powell", {}], verbose=True)
        if pars_init is None:
            pars_init = np.array(self.pars_central)
        else:
            pars_init = np.array(pars_init)
        utils.check_set_dict_keys(self.predictions["logpdf_max"],
                                  [timestamp],
                                  [{}], verbose=False)
        start = timer()
        res = inference.compute_maximum_logpdf(logpdf=lambda x: self.logpdf_fn(x,*self.logpdf.args, *self.logpdf.kwargs), 
                                               pars_init=pars_init,
                                               pars_bounds=None,
                                               optimizer=optimizer,
                                               verbose=verbose_sub)
        self.predictions["logpdf_max"][timestamp]["x"], self.predictions["logpdf_max"][timestamp]["y"] = res
        end = timer()
        self.predictions["logpdf_max"][timestamp]["pars_init"] = pars_init
        self.predictions["logpdf_max"][timestamp]["optimizer"] = optimizer
        self.predictions["logpdf_max"][timestamp]["optimization_time"] = end-start
        print("Maximum logpdf computed in",end-start,"s.")
        self.log[timestamp] = {"action": "computed maximum logpdf"}
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)

    def compute_profiled_maxima_logpdf(self,
                                       pars=None,
                                       pars_ranges=None,
                                       pars_init = None,
                                       pars_bounds = None,
                                       spacing="grid",
                                       optimizer = {},
                                       timestamp = None,
                                       progressbar=False,
                                       save=False,
                                       overwrite=False,
                                       verbose=None):
        """
        Computes logal maxima of the logpdf for different values of the parameter ``pars``.
        For the list of prameters ``pars``, ranges are passed as ``pars_ranges`` in the form ``(par_min,par_max,n_points)``
        and an array of points is generated according to the argument ``spacing`` (either a grid or a random 
        flat distribution) in the interval. The points in the grid falling outside 
        :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>` are automatically removed.
        All information on the maximum, including parameters initialization, parameters bounds, and optimizer, 
        are stored in the ``"logpdf_profiled_max"`` item of the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary.
        The method also automatically computes, with the same optimizer, the absolute maximum and the :math:`t_{\\pmb\\mu}`
        test statistics. The latter is defined, given a vector of parameters of interest :math:`\\pmb\\mu` and a vector of nuisance parameters
        :math:`\\pmb\\delta` as
        
        .. math::

            t_{\\pmb\\mu}=-2\\left(\\sup_{\\pmb\\delta}\\log\\mathcal{L}(\\pmb\\mu,\\pmb\\delta)-\\sup_{\\pmb\\mu,\\pmb\\delta}\\log\\mathcal{L}(\\pmb\\mu,\\pmb\\delta)\\right).


        Profiled maxima could be used both for frequentist inference and as initial condition for
        Markov Chain Monte Carlo through the :class:`Sampler <DNNLikelihood.Sampler>` object
        (see the :mod:`Sampler <sampler>` object documentation). 
        The method uses the function :func:`inference.compute_profiled_maximum_logpdf <DNNLikelihood.inference.compute_profiled_maximum_logpdf>`
        based on |scipy_optimize_minimize_link| to find the (local) minimum of minus
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`. Since the latter
        method already contains a bounded logpdf, ``pars_bounds`` is set to ``None`` in the 
        :func:`inference.compute_profiled_maximum_logpdf <DNNLikelihood.inference.compute_profiled_maximum_logpdf>`
        function to optimize speed. See the documentation of the
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
            
                Bounds on the parameters. If it is ``None``, then
                it is set to the parameters bounds attribute :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>`.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,2)``
                    - **default**: ``None`` (automatically modified to :attr:`Lik.pars_bounds <DNNLikelihood.Lik.pars_bounds>`)

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
                          This is always set to ``"scipy"``, which is by now the only available optimizer for this task. As more
                          optimizers will be supported the ``"name"`` key will indicate the chosen one.
                        - *"method"* (value type: ``str``)
                          Optimizer method. For the |scipy_optimize_minimize_link| optimizer the default method is given by ``"Powell"``.
                        - *"options"* (value type: ``dict``)
                          Dictionary containing additional options to pass to the |scipy_optimize_minimize_link| function.

                    - **default**: {}
                    - **schematic example**:

                        .. code-block:: python
                            
                            optimizer={"name": "scipy",
                                       "method": "Powell",
                                       "options": {"maxiter": 10000,
                                                   "ftol": 0.001}}

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used as key in the 
                :attr:`Lik.predictions["logpdf_max"] <DNNLikelihood.Lik.predictions>` dictionary to save the current prediction.
                It is also used to update the :attr:`Lik.log <DNNLikelihood.Lik.log>` dictionary and to save the object when 
                ``overwrite="dump"``.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **progressbar**
            
                If ``True`` 
                then  a progress bar is shown.
                    
                    - **type**: ``bool``
                    - **default**: ``True`` 

            - **save**
            
                If ``True`` the object is saved (by passing the ``overwrite`` argument to the
                :meth:`Lik.save <DNNLikelihood.Lik.save>` method).
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>` (only if ``save=True``)
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>` (always)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
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
                print("If you want to show a progress bar please install the ipywidgets package.",show=verbose)
        start = timer()
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                "width": "500px", "height": "14px",
                "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
        utils.check_set_dict_keys(self.predictions["logpdf_profiled_max"],
                                  [timestamp],
                                  [{}], verbose=False)
        utils.check_set_dict_keys(optimizer, ["name",
                                              "method",
                                              "options"],
                                             ["scipy","Powell", {}], verbose=verbose_sub)
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
                                                                 pars_bounds=None,
                                                                 optimizer=optimizer,
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
                                    timestamp=timestamp,
                                    save=False,
                                    overwrite=False,
                                    verbose=verbose_sub)
        self.predictions["logpdf_profiled_max"][timestamp]["tmu"] = np.array(list(zip(X_tmp[:, pars].flatten(), -2*(Y_tmp-self.predictions["logpdf_max"][timestamp]["y"]))))
        self.predictions["logpdf_profiled_max"][timestamp]["pars"] = pars
        self.predictions["logpdf_profiled_max"][timestamp]["pars_ranges"] = pars_ranges
        self.predictions["logpdf_profiled_max"][timestamp]["pars_init"] = pars_init
        self.predictions["logpdf_profiled_max"][timestamp]["pars_bounds"] = pars_bounds
        self.predictions["logpdf_profiled_max"][timestamp]["optimizer"] = optimizer
        self.predictions["logpdf_profiled_max"][timestamp]["optimization_times"] = optimization_times
        end = timer()
        self.predictions["logpdf_profiled_max"][timestamp]["global_optimization_time"] = np.array(optimization_times).sum()
        self.log[timestamp] = {"action": "computed profiled maxima", 
                               "pars": pars,
                               "pars_ranges": pars_ranges, 
                               "number of maxima": len(X_tmp)}
        if save:
            self.save(overwrite=overwrite, verbose=verbose_sub)
        else:
            self.save_log(overwrite=True, verbose=verbose_sub)
        print("Log-pdf values lie in the range [", np.min(self.predictions["logpdf_profiled_max"][timestamp]["Y"]), ",", np.max(self.predictions["logpdf_profiled_max"][timestamp]["Y"]), "]", show=verbose)
        print(len(pars_vals_bounded),"local maxima computed in", end-start, "s.",show=verbose)

    def update_figures(self,figure_file=None,timestamp=None,overwrite=False):
        """
        Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
        When ``overwrite=False`` it calls the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function and, if 
        ``figure_file`` already existed in the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary, then it
        updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
        (that is the key of the :attr:`Lik.predictions["Figures"] <DNNLikelihood.Lik.predictions>` dictionary).
        When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function
        to generate the dump file name.
        It returns the new figure_file.

        - **Arguments**

            - **figure_file**

                Figure file path. If the figure already exists in the 
                :meth:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary, then its name is updated with the corresponding timestamp.

            - **overwrite**

                The method updates file names and :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary only if
                ``overwrite=False``. If ``overwrite="dump"`` the method generates and returns the dump file path. 
                If ``overwrite=True`` the method does nothing.
        
        - **Creates/updates files**

            - Updates ``figure_file`` file name.
        """
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
                        old_figure_file = utils.check_rename_file(figure_file,timestamp=timestamp)
                        if old_figure_file is not None:
                            self.predictions["Figures"][k] = [q.replace(figure_file,old_figure_file) for q in v]
            elif overwrite == "dump":
                new_figure_file = utils.generate_dump_file_path(figure_file, timestamp=timestamp)
        return new_figure_file

    def plot_logpdf_par(self, 
                        pars=[[0,0,1]], 
                        npoints=100, 
                        pars_init=None, 
                        pars_labels="original", 
                        title_fontsize=12, 
                        show_plot=False,
                        timestamp=None,
                        overwrite=False, 
                        verbose=None):
        """
        Plots the logpdf as a function of of the parameter ``par`` in the range ``(min,max)``
        using a number ``npoints`` of points. Only the parameter ``par`` is veried, while all other parameters are kept
        fixed to their value given in ``pars_init``. The function used for the plot is provided by the 
        :meth:`Lik.logpdf_fn <DNNLikelihood.Lik.logpdf_fn>`.

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
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``"original"``

            - **title_fontsize**
            
                Font size of the figure 
                title.
                    
                    - **type**: ``int``
                    - **default**: ``12``
            
            - **show_plot**
            
                If ``True`` the plot is shown on the 
                interactive console.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used as key in the 
                :attr:`Lik.predictions["logpdf_max"] <DNNLikelihood.Lik.predictions>` dictionary to save the current prediction.
                It is also used to update the :attr:`Lik.log <DNNLikelihood.Lik.log>` dictionary and to save the object when 
                ``overwrite="dump"``.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates files**

            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>` ``+ "_par_" + str(par[0]) + ".pdf"`` for each ``par`` in ``pars``.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
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
            figure_file = self.update_figures(figure_file=self.output_figures_base_file + "_par_"+str(par[0])+".pdf",timestamp=timestamp,overwrite=overwrite) 
            plt.savefig(figure_file)
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file)
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": path.split(figure_file)[-1], 
                                   "file path": figure_file}
            print(r"%s" % (figure_file), "created and saved in",str(end-start), "s.", show=verbose)
        self.save_log(overwrite=True, verbose=verbose_sub)
        
    def plot_tmu_1d(self,
                    pars_labels="original",
                    title_fontsize=12,
                    show_plot=False,
                    timestamp=None,
                    overwrite=False,
                    verbose=None):
        """
        Plots the 1-dimensional :math:`t_{\\mu}` stored in 
        :attr:`Lik.predictions["logpdf_profiled_max"][timestamp]["tmu"] <DNNLikelihood.Lik.predictions>`.

        - **Arguments**

            - **pars_labels**
            
                Argument that is passed to the :meth:`Lik.__set_pars_labels <DNNLikelihood.Lik._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plot.
                    
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``"original"``

            - **title_fontsize**
            
                Font size of the figure 
                title.
                    
                    - **type**: ``int``
                    - **default**: ``12``

            - **show_plot**
            
                If ``True`` the plot is shown on the 
                interactive console.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **timestamp**
            
                A timestamp string with format ``"datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]``.
                If it is not passed, then it is generated. It is used as key in the 
                :attr:`Lik.predictions["logpdf_max"] <DNNLikelihood.Lik.predictions>` dictionary to save the current prediction.
                It is also used to update the :attr:`Lik.log <DNNLikelihood.Lik.log>` dictionary and to save the object when 
                ``overwrite="dump"``.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. 
                If ``False`` is a file with the same name already exists, then the old file gets renamed with the 
                :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function.
                If ``"dump"``, a dump of the file is saved through the 
                :func:`utils.generate_dump_file_path <DNNLikelihood.utils.generate_dump_file_path>` function.
                    
                    - **type**: ``bool`` or ``str``
                    - **allowed str**: ``"dump"``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates files**

            - :attr:`Lik.output_figures_base_file <DNNLikelihood.Lik.output_figures_base_file>` ``+ "_tmu_" + str(par) + ".pdf"``.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        pars_labels = self.__set_pars_labels(pars_labels)
        pars_list = self.predictions["logpdf_profiled_max"][timestamp]["pars"]
        tmu_list = self.predictions["logpdf_profiled_max"][timestamp]["tmu"]
        if len(pars_list) == 1:
            par = pars_list[0]
        else:
            raise Exception("Parameters should be  should be the same for the different tmu sources.")
        figure_file = self.update_figures(figure_file=self.output_figures_base_file+ "_tmu_"+str(par) + ".pdf",timestamp=timestamp,overwrite=overwrite)
        plt.plot(tmu_list[:, 0], tmu_list[:,-1], label="Likelihood")
        plt.title(r"%s" % self.name, fontsize=title_fontsize)
        plt.xlabel(r"$t_{\mu}$(%s)" % (self.pars_labels[par]))
        plt.ylabel(r"%s" % (self.pars_labels[par]))
        plt.legend()
        plt.tight_layout()
        plt.savefig(r"%s" % figure_file)
        utils.check_set_dict_keys(self.predictions["Figures"], [timestamp], [[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_file)[-1],
                               "file path": figure_file}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(r"%s" % (figure_file), "created and saved in",str(end-start), "s.", show=verbose)

            
