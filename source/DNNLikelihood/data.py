__all__ = ["Data"]

import builtins
import codecs
import json
import time
from datetime import datetime
from os import path
from timeit import default_timer as timer

import deepdish as dd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import utils
from .show_prints import Verbosity, print


class Data(Verbosity):
    """
    This class contains the ``Data`` object representing the dataset used for training, validating and testing
    the DNNLikelihood. It can be creaded both feeding X and Y data or by loading an existing ``Data`` object.
    """
    def __init__(self,
                 name = None,
                 data_X = None,
                 data_Y = None,
                 dtype = None,
                 pars_central=None,
                 pars_pos_poi=None,
                 pars_pos_nuis=None,
                 pars_labels=None,
                 pars_bounds=None,
                 test_fraction = None,
                 load_on_RAM=False,
                 output_folder = None,
                 input_file=None,
                 verbose = True
                 ):
        """
        The :class:`Data <DNNLikelihood.Data>` object can be initialized in two different ways, depending on the value of
        the :argument:`input_file` argument.

        - :argument:`input_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no :argument:`name` is given,
            then one is created.
            The object is saved upon creation through the :meth:`Data.save <DNNLikelihood.Data.save>` method.

        - :argument:`input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Data.__load <DNNLikelihood.Data._Data__load>`.
            If the :argument:`load_on_RAM` argument is ``True``, then all samples are loaded into RAM, otherwise the dataset is
            kept open and data are retrieved on demand.
            If the input argument :argument:`output_folder` is ``None`` (default), the attribute
            :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>`
            is set from the input file, otherwise it is set to the input argument.

        - **Arguments**

            See class :ref:`Arguments documentation <data_arguments>`.

        - **Creates files**

            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
            - :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>` (only the first time the object is created, i.e. if :argument:`input_file` is ``None``)
            - :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>` (only the first time the object is created, i.e. if :argument:`input_file` is ``None``)
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.input_file = input_file
        self.__check_define_input_files()
        if self.input_file is None:
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.dtype = dtype
            self.data_X = data_X
            self.data_Y = data_Y
            self.__check_define_data()
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_central = pars_central
            self.ndims = self.data_X.shape[1]
            self.npoints = self.data_X.shape[0]
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
            self.__check_define_pars(verbose=verbose_sub)
            self.test_fraction = test_fraction
            self.__define_test_fraction()
            self.output_folder = output_folder
            self.__check_define_output_files()
            self.load_on_RAM = load_on_RAM
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.load_on_RAM = load_on_RAM
            if dtype is not None:
                if type(dtype) is str:
                    self.dtype_required = dtype
                elif type(dtype) is list:
                    self.dtype_required = dtype[1]
            else:
                self.dtype_required = "float64"
            self.__load(verbose=verbose_sub)
            self.__define_test_fraction()
            if output_folder is not None:
                self.output_folder = path.abspath(output_folder)
                self.__check_define_output_files()
            self.save_log(overwrite=True, verbose=verbose_sub)

        self.data_dictionary = {"X_train": np.array([[]], dtype=self.dtype_required), "Y_train": np.array([], dtype=self.dtype_required),
                                "X_val": np.array([[]], dtype=self.dtype_required), "Y_val": np.array([], dtype=self.dtype_required),
                                "X_test": np.array([[]], dtype=self.dtype_required), "Y_test": np.array([], dtype=self.dtype_required),
                                "idx_train": np.array([], dtype="int"), "idx_val": np.array([], dtype="int"), "idx_test": np.array([], dtype="int")}

    def __check_define_input_files(self):
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to set the attributes corresponding to input files:

            - :attr:`Data.input_log_file <DNNLikelihood.Data.input_log_file>`
            - :attr:`Data.input_object_h5_file <DNNLikelihood.Data.input_object_h5_file>`
            - :attr:`Data.input_samples_h5_file <DNNLikelihood.Data.input_samples_h5_file>`

        depending on the value of the 
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>` attribute.
        """
        if self.input_file is None:
            self.input_object_h5_file = None
            self.input_samples_h5_file = None
            self.input_log_file = None
        else:
            self.input_file = self.input_file.replace("_object","")
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_object_h5_file = self.input_file+"_object.h5"
            self.input_samples_h5_file = self.input_file+"_samples.h5"
            self.input_log_file = self.input_file+".log"

    def __check_define_output_files(self):
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to set the attributes corresponding to output files

            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
            - :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`
            - :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`

        depending on the values of the 
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` attribute.
        It also creates the folder 
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` if 
        it does not exist.
        """
        if self.output_folder is None:
            self.output_folder = ""
        self.output_folder = path.abspath(self.output_folder)
        self.output_object_h5_file = path.join(self.output_folder, self.name+"_object.h5")
        self.output_samples_h5_file = path.join(self.output_folder, self.name+"_samples.h5")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        utils.check_create_folder(self.output_folder)
        
    def __check_define_name(self):
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to define the :attr:`Data.name <DNNLikelihood.Data.name>` attribute.
        If it is ``None`` it replaces it with
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_data"``,
        otherwise it appends the suffix "_data" (preventing duplication if it is already present).
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.name = "model_"+timestamp+"_data"
        else:
            self.name = utils.check_add_suffix(self.name, "_data")

    def __check_define_data(self):
        """ 
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to check that data_X and data_Y have the same length, to set the data type attributes
        :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>` and
        :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>` (defaults "float64"), and to convert
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`
        into the dtype corresponding to :attr:`Data.dtype_stored <DNNLikelihood.Data.dtype_stored>`.
        """
        if self.dtype is None:
            self.dtype_stored = "float64"
            self.dtype_required = "float64"
        elif type(self.dtype) is str:
            self.dtype_stored = self.dtype
            self.dtype_required = self.dtype
        elif type(self.dtype) is list:
            self.dtype_stored = self.dtype[0]
            self.dtype_required = self.dtype[1]
        del(self.dtype)
        if len(self.data_X) == len(self.data_Y):
            self.data_X = self.data_X.astype(self.dtype_stored)
            self.data_Y = self.data_Y.astype(self.dtype_stored)
        else:
            raise Exception("data_X and data_Y have different length.")

    def __check_define_pars(self, verbose=None):
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to check the consistency of the attributes

            - :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`
            - :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` 

        and to convert the attributes

            - :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` 
        
        to |numpy_link| arrays.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`
        is set to the value of :attr:`Data.pars_labels_auto <DNNLikelihood.Data.pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Data.ndims <DNNLikelihood.Data.ndims>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.pars_central is not None:
            self.pars_central = np.array(self.pars_central)
            if len(self.pars_central) != self.ndims:
                raise Exception("The lenght of the parameters central values array does not match the number of dimensions.")
        else:
            self.pars_central = np.zeros(self.ndims)
            print("No central values for the parameters 'pars_central' has been specified. They have been set to zero for all\
                parameters. If they are known it is better to build the object providing parameters central values.", show=verbose)
        if self.pars_pos_nuis is not None and self.pars_pos_poi is not None:
            if len(self.pars_pos_poi)+len(self.pars_pos_nuis) == self.ndims:
                self.pars_pos_nuis = np.array(self.pars_pos_nuis)
                self.pars_pos_poi = np.array(self.pars_pos_poi)
            else:
                raise Exception("The number of parameters positions do not match the number of dimensions.")
        elif self.pars_pos_nuis is None and self.pars_pos_poi is None:
            print("The positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified. Assuming all parameters are parameters of interest.", show=verbose)
            self.pars_pos_nuis = np.array([])
            self.pars_pos_poi = np.array(list(range(self.ndims)))
        elif self.pars_pos_nuis is not None and self.pars_pos_poi is None:
            print("Only the positions of the nuisance parameters have been specified. Assuming all other parameters are parameters of interest.", show=verbose)
            self.pars_pos_poi = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_nuis))
        elif self.pars_pos_nuis is None and self.pars_pos_poi is not None:
            print("Only the positions of the parameters of interest have been specified. Assuming all other parameters are nuisance parameters.", show=verbose)
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
            raise Exception("The lenght of the parameters bounds array does not match the number of dimensions.")

    def __load(self,verbose=None):
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one 
        to load a previously saved
        :class:`Data <DNNLikelihood.Data>` object from the files 
        
            - :attr:`Data.input_samples_h5_file <DNNLikelihood.Data.input_samples_h5_file>`
            - :attr:`Data.input_object_h5_file <DNNLikelihood.Data.input_object_h5_file>`
            - :attr:`Data.input_log_file <DNNLikelihood.Data.input_log_file>`

        If :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>` is ``True``, then all samples are loaded into RAM, 
        otherwise the dataset is kept open and data are retrieved on demand.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        dictionary = dd.io.load(self.input_object_h5_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        self.opened_dataset = h5py.File(self.input_samples_h5_file, "r")
        data = self.opened_dataset["data"]
        self.data_X = data.get("X")
        self.data_Y = data.get("Y")
        if self.load_on_RAM:
            self.data_X = self.data_X[:].astype(self.dtype_required)
            self.data_Y = self.data_Y[:].astype(self.dtype_required)
            self.opened_dataset.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_samples_h5_file)[-1],
                                               path.split(self.input_log_file)[-1],
                                               path.split(self.input_object_h5_file)[-1]],
                               "files paths": [self.input_samples_h5_file,
                                               self.input_log_file,
                                               self.input_object_h5_file]}
        print("Data object loaded in", str(end-start), ".",show=verbose)
        if self.load_on_RAM:
            print("Samples loaded on RAM.", show=verbose)

    def __define_test_fraction(self):
        """ 
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to set the :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` attribute and, from this, the
        two attributes :attr:`Data.train_range <DNNLikelihood.Data.train_range>` and 
        :attr:`Data.test_range <DNNLikelihood.Data.test_range>` containing the range of indices of
        the train (and validation) and test sets given by
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and
        :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`.
        """
        if self.test_fraction is None:
            self.test_fraction = 0
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)

    def __check_sync_data_dictionary(self,verbose=None):
        """
        Private method used to check that indices and data are synced in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`
        dictionary. It is called by all methods generating or updating data.
        This is needed when the :class:`Data <DNNLikelihood.Data>` object is used by the 
        :class:`DnnLik <DNNLikelihood.DnnLik>` one. In particular, when the latter is imported from files
        only indices are loaded and properly put in the corresponding 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`, while, to gain speed, data are not.
        Whenever the :class:`DnnLik <DNNLikelihood.DnnLik>` needs data, and calls any of its method that
        generate them, the method :meth:`Data.__check_sync_data_dictionary <DNNLikelihood.Data._Data__check_sync_data_dictionary>`
        automatically loads the data corresponding to the existing indices in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` of the corresponding 
        :class:`Data <DNNLikelihood.Data>` object. See the documentation of 
        :ref:`the DNNLikelihood object <DnnLik_object>` for more details.
        """
        if len(self.data_dictionary["idx_train"]) != 0 and len(self.data_dictionary["idx_train"]) != len(self.data_dictionary["X_train"]):
            self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype_required)
            self.data_dictionary["Y_train"] = self.data_Y[self.data_dictionary["idx_train"]].astype(self.dtype_required)
            print("Loaded train data corresponding to existing indices.")
        if len(self.data_dictionary["idx_val"]) != 0 and len(self.data_dictionary["idx_val"]) != len(self.data_dictionary["X_val"]):
            self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype_required)
            self.data_dictionary["Y_val"] = self.data_Y[self.data_dictionary["idx_val"]].astype(self.dtype_required)
            print("Loaded val data corresponding to existing indices.")
        if len(self.data_dictionary["idx_test"]) != 0 and len(self.data_dictionary["idx_test"]) != len(self.data_dictionary["X_test"]):
            self.data_dictionary["X_test"] = self.data_X[self.data_dictionary["idx_test"]].astype(self.dtype_required)
            self.data_dictionary["Y_test"] = self.data_Y[self.data_dictionary["idx_test"]].astype(self.dtype_required)
            print("Loaded test data corresponding to existing indices.")

    def define_test_fraction(self,verbose=None):
        """ 
        Method analog to the private method :meth:`Data.__define_test_fraction <DNNLikelihood.Data._Data__define_test_fraction>`
        useful to change the available fraction of train/valifation vs test data.
        Differently from the analog private method, this one updates the files corresponding to the
        :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>` and
        :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>` attributes.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Updates files**

            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
            - :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`
        """
        _, verbose_sub = self.set_verbosity(verbose)
        if self.test_fraction is None:
            self.test_fraction = 0
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)
        self.save(overwrite=True, verbose=verbose_sub)

    def close_opened_dataset(self,verbose=None):
        """ 
        Closes the opened h5py datasets 
        :attr:`Data.opened_dataset <DNNLikelihood.Data.opened_dataset>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, _ = self.set_verbosity(verbose)
        try:
            self.opened_dataset.close()
            del(self.opened_dataset)
            print("Closed", self.input_file,".",show=verbose)
        except:
            print("No dataset to close.", show=verbose)

    def save_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Data.log <DNNLikelihood.Data.log>` attribute in the file
        :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`

        This method is called by the methods
        
        - :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` with ``overwrite=True`` and ``verbose=verbose_sub``  with ``overwrite=True`` and ``verbose=verbose_sub`` if :attr:`Data.input_file <DNNLikelihood.Data.input_file>` is not ``None``, and with ``overwrite=True`` and ``verbose=verbose_sub`` otherwise
        - :meth:`Data.save <DNNLikelihood.Data.save>` with ``overwrite=overwrite`` and ``verbose=verbose``
        - :meth:`Data.generate_train_indices <DNNLikelihood.Data.generate_train_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_train_data <DNNLikelihood.Data.generate_train_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.update_train_indices <DNNLikelihood.Data.update_train_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.update_train_data <DNNLikelihood.Data.update_train_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_test_indices <DNNLikelihood.Data.generate_test_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_test_data <DNNLikelihood.Data.generate_test_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.compute_sample_weights <DNNLikelihood.Data.compute_sample_weights>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.define_scalers <DNNLikelihood.Data.define_scalers>` with ``overwrite=True`` and ``verbose=verbose_sub``

        This method is called by the
        :meth:`Data.save <DNNLikelihood.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates/updates file**

            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_log_file, verbose=verbose_sub)
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if overwrite:
            print("Data log file", self.output_log_file,"updated in", str(end-start), "s.",show=verbose)
        else:
            print("Data log file", self.output_log_file, "saved in", str(end-start), "s.", show=verbose)

    def save_object_h5(self, overwrite=False, verbose=True):
        """
        The :class:`Lik <DNNLikelihood.Data>` object is saved to the HDF5 file
        :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`, the datasets corresponding to the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attributes
        are saved to the file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`.
        
        This method saves the content of the :attr:``Lik.__dict__ <DNNLikelihood.Lik.__dict__>`` 
        attribute in the h5 file :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`
        using the |deepdish_link| package. The following attributes are excluded from the saved
        dictionary:

            - :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`
            - :attr:`Data.data_X <DNNLikelihood.Data.data_X>` (saved to the file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`)
            - :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` (saved to the file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`)
            - :attr:`Data.dtype_required <DNNLikelihood.Data.dtype_required>`
            - :attr:`Data.input_file <DNNLikelihood.Data.input_file>`
            - :attr:`Data.input_samples_h5_file <DNNLikelihood.Data.input_samples_h5_file>`
            - :attr:`Data.input_object_h5_file <DNNLikelihood.Data.input_object_h5_file>`
            - :attr:`Data.input_log_file <DNNLikelihood.Data.input_log_file>`
            - :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>`
            - :attr:`Data.log <DNNLikelihood.Data.log>` (saved to the file :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`)
            - :attr:`Data.opened_dataset <DNNLikelihood.Data.opened_dataset>`
            - :attr:`Data.test_range <DNNLikelihood.Data.test_range>`
            - :attr:`Data.train_range <DNNLikelihood.Data.train_range>`
            - :attr:`Data.verbose <DNNLikelihood.Data.verbose>`

        This method is called by the
        :meth:`Data.save <DNNLikelihood.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_object_h5_file, verbose=verbose_sub)
        dictionary = utils.dic_minus_keys(self.__dict__, ["data_dictionary","data_X","data_Y",
                                                          "dtype_required","input_file","input_samples_h5_file",
                                                          "input_object_h5_file","input_log_file","load_on_RAM",
                                                          "log","opened_dataset","test_range",
                                                          "train_range","verbose"])
        dd.io.save(self.output_object_h5_file, dictionary)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_object_h5_file)[-1],
                               "file path": self.output_object_h5_file}
        print("Data object saved to file", self.output_object_h5_file,"in", str(end-start), "s.",show=verbose)

    def save_samples_h5(self, overwrite=False, verbose=None):
        """
        The :class:`Lik <DNNLikelihood.Data>` object is saved to the HDF5 file
        :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`, the datasets corresponding to the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attributes
        are saved to the file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`.
        
        This method saves the content of the attributes :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and
        :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`, and an additional entry with the shape of the former
        in the h5 file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`
        using the |h5py_link| package.
        
        This method is called by the :meth:`Data.save <DNNLikelihood.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_samples_h5_file,verbose=verbose_sub)
        start = timer()     
        h5_out = h5py.File(self.output_samples_h5_file, "w")
        data = h5_out.create_group("data")
        data["shape"] = np.shape(self.data_X)
        data["X"] = self.data_X.astype(self.dtype_stored)
        data["Y"] = self.data_Y.astype(self.dtype_stored)
        h5_out.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_samples_h5_file)[-1],
                               "file path": self.output_samples_h5_file}
        print(str(self.npoints), "data samples (data_X, data_Y) saved to file", self.output_samples_h5_file,"in", end-start, "s.",show=verbose)

    def save(self, overwrite=False, verbose=None):
        """
        The :class:`Lik <DNNLikelihood.Data>` object is saved to the HDF5 file
        :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`, the datasets corresponding to the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>` and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` attributes
        are saved to the file :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`.

        This method calls in order the three methods
        
            - :meth:`Data.save_log <DNNLikelihood.Data.save_json>`
            - :meth:`Data.save_object_h5 <DNNLikelihood.Data.save_object_h5>`
            - :meth:`Data.save_samples_h5 <DNNLikelihood.Data.save_samples_h5>`

        to save the full object.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
            - :attr:`Data.output_object_h5_file <DNNLikelihood.Data.output_object_h5_file>`
            - :attr:`Data.output_samples_h5_file <DNNLikelihood.Data.output_samples_h5_file>` 
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_object_h5(overwrite=overwrite, verbose=verbose)
        self.save_samples_h5(overwrite=overwrite, verbose=verbose)
        self.save_log(overwrite=overwrite, verbose=verbose)
        
    def generate_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Method used to generate ``npoints_train`` train and ``npoints_val`` validation data indices 
        (integer positions in the 
        :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` arrays). It updates the 
        ``"idx_train"`` and ``"idx_val"`` items of the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        _, verbose_sub = self.set_verbosity(verbose)
        np.random.seed(seed)
        idx_train = np.random.choice(self.train_range, npoints_train+npoints_val, replace=False)
        idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)]
        self.data_dictionary["idx_train"] = idx_train
        self.data_dictionary["idx_val"] = idx_val
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_train", "idx_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by generate_train_data

    def generate_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Method used to generate ``npoints_train`` train and ``npoints_val`` validation data.
        In order to extract the required number of points from the :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` arrays (which may not be available on RAM),
        it first generates indices with the :meth:`Data.generate_train_indices <DNNLikelihood.Data.generate_train_indices>`
        method and then slices the data with the generated indices.
        It updates the ``"X_train"``, ``"Y_train"``, ``"X_val"`` and ``"Y_val"`` items of the
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        self.generate_train_indices(npoints_train, npoints_val, seed)
        self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype_required)
        self.data_dictionary["Y_train"] = self.data_Y[self.data_dictionary["idx_train"]].astype(self.dtype_required)
        self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype_required)
        self.data_dictionary["Y_val"] = self.data_Y[self.data_dictionary["idx_val"]].astype(self.dtype_required)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_train", "Y_train", "X_val", "Y_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Generated", str(npoints_train), "(X_train, Y_train) samples and ", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.",show=verbose)

    def update_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_train_indices <DNNLikelihood.Data.generate_train_indices>`, but it only updates
        indices adding more points if any are already available in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
        It updates the  ``"idx_train"`` and ``"idx_val"`` items of the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        _, verbose_sub = self.set_verbosity(verbose)
        existing_train = np.sort(np.concatenate((self.data_dictionary["idx_train"], self.data_dictionary["idx_val"])))
        np.random.seed(seed)
        idx_train = np.random.choice(np.setdiff1d(np.array(self.train_range), existing_train), npoints_train+npoints_val, replace=False)
        if np.size(idx_train) != 0:
            idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)]
        else:
            idx_val = idx_train
        self.data_dictionary["idx_train"] = np.sort(np.concatenate((self.data_dictionary["idx_train"],idx_train)))
        self.data_dictionary["idx_val"] = np.sort(np.concatenate((self.data_dictionary["idx_val"],idx_val)))
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_train", "idx_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by update_train_data
        return [idx_train, idx_val]

    def update_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_train_indices <DNNLikelihood.Data.generate_train_data>`, but it only updates
        data adding more points if any are already available in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.
        It updates the  ``"X_train"``, ``"Y_train"``, ``"X_val"`` and ``"Y_val"`` items of the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        [npoints_train, npoints_val] = [(i > 0) * i for i in [npoints_train-len(self.data_dictionary["idx_train"]), 
                                                                            npoints_val-len(self.data_dictionary["idx_val"])]]
        idx_train, idx_val = self.update_train_indices(npoints_train, npoints_val, seed, verbose=verbose)
        if idx_train != []:
            if self.data_dictionary["X_train"].size == 0:
                self.data_dictionary["X_train"] = self.data_X[idx_train].astype(self.dtype_required)
                self.data_dictionary["Y_train"] = self.data_Y[idx_train].astype(self.dtype_required)
            else:
                self.data_dictionary["X_train"] = np.concatenate((self.data_dictionary["X_train"],self.data_X[idx_train])).astype(self.dtype_required)
                self.data_dictionary["Y_train"] = np.concatenate((self.data_dictionary["Y_train"],self.data_Y[idx_train])).astype(self.dtype_required)
        if idx_val != []:
            if self.data_dictionary["X_val"].size == 0:
                self.data_dictionary["X_val"] = self.data_X[idx_val].astype(self.dtype_required)
                self.data_dictionary["Y_val"] = self.data_Y[idx_val].astype(self.dtype_required)
            else:
                self.data_dictionary["X_val"] = np.concatenate((self.data_dictionary["X_val"], self.data_X[idx_val])).astype(self.dtype_required)
                self.data_dictionary["Y_val"] = np.concatenate((self.data_dictionary["Y_val"], self.data_Y[idx_val])).astype(self.dtype_required)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_train", "Y_train", "X_val", "Y_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Added", str(npoints_train), "(X_train, Y_train) samples and", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.",show=verbose)

    def generate_test_indices(self, npoints_test, verbose=None):
        """
        Method used to generate ``npoints_test`` test data indices 
        (integer positions in the :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` arrays). Test indices are always generated
        deterministically and always represent
        the first ``npoints_test`` in the :attr:`Dara.test_range <DNNLikelihood.Data.test_range>` range.
        It updates the ``"idx_test"`` item of the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of 
                test points.

                    - **type**: ``int``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        _, verbose_sub = self.set_verbosity(verbose)
        n_existing_test = len(self.data_dictionary["idx_test"])
        idx_test = np.array(self.test_range)[range(n_existing_test, n_existing_test+npoints_test)]
        self.data_dictionary["idx_test"] = np.concatenate((self.data_dictionary["idx_test"],idx_test))
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_test"],
                               "npoints test": npoints_test}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by generate_test_data
        return idx_test

    def generate_test_data(self, npoints_test, verbose=None):
        """
        Method used to generate ``npoints_test`` test data indices
        (integer positions in the :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` arrays). In order to extract the required 
        number of points from the :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` arrays (which may not be available on RAM),
        it first generates indices with the :meth:`Data.generate_test_indices <DNNLikelihood.Data.generate_test_indices>`
        method and then slices the data with the generated indices.
        Test data are always generated deterministically and always correspond to the first ``npoints_test`` 
        in the :attr:`Dara.test_range <DNNLikelihood.Data.test_range>` range of :attr:`Data.data_X <DNNLikelihood.Data.data_X>`
        and :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`.
        It updates the ``"X_test"`` and ``"Y_test"`` items of the
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of
                test points.

                    - **type**: ``int``

            - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        npoints_test = npoints_test-len(self.data_dictionary["Y_test"])
        idx_test = self.generate_test_indices(npoints_test, verbose=verbose)
        if idx_test != []:
            if self.data_dictionary["X_test"].size == 0:
                self.data_dictionary["X_test"] = self.data_X[idx_test].astype(self.dtype_required)
                self.data_dictionary["Y_test"] = self.data_Y[idx_test].astype(self.dtype_required)
            else:
                self.data_dictionary["X_test"] = np.concatenate((self.data_dictionary["X_test"], self.data_X[idx_test])).astype(self.dtype_required)
                self.data_dictionary["Y_test"] = np.concatenate((self.data_dictionary["Y_test"], self.data_Y[idx_test])).astype(self.dtype_required)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_train", "Y_train", "X_val", "Y_val"],
                               "npoints train": npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Added", str(npoints_test), "(X_test, Y_test) samples in", end-start, "s.",show=verbose)

    def compute_sample_weights(self, sample, nbins=100, power=1, verbose=None):
        """
        Method that computes weights of points given their distribution. Sample weights are used to weigth data as a function of 
        their frequency, obtained by binning the data in ``nbins`` and assigning weight equal to the inverse of the bin count
        to the power ``power``. In order to avoid too large weights for bins with very few counts, all bins with less than 5 counts
        are assigned frequency equal to 1/5 to the power ``power``.
        
        - **Arguments**

            - **sample**

                Distribution of points that 
                need to be weighted.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(len(sample),)``

            - **nbins**

                Number of bins to histogram the 
                sample data

                    - **type**: ``int``
                    - **default**: ``100``

            - **power**

                Exponent of the inverse of the bin count used
                to assign weights.

                    - **type**: ``float``
                    - **default**: ``1``

            - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Returns**

            |Numpy_link| array of the same length of ``sample`` containing
            the required weights.
            
                - **type**: ``numpy.ndarray``
                - **shape**: ``(len(sample),)``
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        hist, edges = np.histogram(sample, bins=nbins)
        hist = np.where(hist < 5, 5, hist)
        tmp = np.digitize(sample, edges, right=True)
        W = 1/np.power(hist[np.where(tmp == nbins, nbins-1, tmp)], power)
        W = W/np.sum(W)*len(sample)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "computed sample weights"}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Sample weights computed in", end-start, "s.",show=verbose)
        return W

    def define_scalers(self, data_X, data_Y, scalerX_bool, scalerY_bool, verbose=None):
        """
        Method that defines |standard_scalers_link| fit to the ``data_X`` and ``data_Y`` data.
        Scalers are defined based in the boolean flags ``scalerX_bool`` and ``scalerY_bool``. When the flags are false
        the scalers are defined with the arguemtns ``with_mean=False`` and ``with_std=False`` which correspond to identity
        transformations.

        - **Arguments**

            - **data_X**

                ``X`` data to fit
                the |standard_scaler_link| ``scalerX``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndim)``

            - **data_Y**

                ``Y`` data to fit
                the |standard_scaler_link| ``scalerY``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,)``

            - **scalerX_bool**

                If true the ``X`` scaler is fit to the ``data_X`` data, otherwise it is set
                to the identity transformation.

                    - **type**: ``bool``

            - **scalerY_bool**

                If true the ``Y`` scaler is fit to the ``data_Y`` data, otherwise it is set
                to the identity transformation.

                    - **type**: ``bool``

            - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Returns**

            List of the form ``[scalerX,scalerY]`` containing 
            the ``X`` and ``Y`` scalers.
            
                - **type**: ``list``
                - **shape**: ``[scalerX,scalerY]``
        """
        _, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if scalerX_bool:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            scalerX.fit(data_X)
        else:
            scalerX = StandardScaler(with_mean=False, with_std=False)
            scalerX.fit(data_X)
        if scalerY_bool:
            scalerY = StandardScaler(with_mean=True, with_std=True)
            scalerY.fit(data_Y.reshape(-1, 1))
        else:
            scalerY = StandardScaler(with_mean=False, with_std=False)
            scalerY.fit(data_Y.reshape(-1, 1))
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "defined scalers",
                               "scaler X": scalerX_bool,
                               "scaler Y": scalerY_bool}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Standard scalers defined in", end-start, "s.", show=verbose)
        return [scalerX, scalerY]
