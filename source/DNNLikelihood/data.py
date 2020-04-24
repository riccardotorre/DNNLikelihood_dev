__all__ = ["Data"]

from os import path
import codecs
import json
import builtins
from datetime import datetime
from timeit import default_timer as timer

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import show_prints, utils
from .show_prints import print


class Data(show_prints.Verbosity):
    """
    This class contains the ``Data`` object representing the dataset used for training, validating and testing
    the DNNLikelihood. It can be creaded both feeding X and Y data or by loading an existing ``Data`` obkect.
    """
    def __init__(self,
                 name = None,
                 data_X = None,
                 data_Y = None,
                 dtype=None,
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
        """Initializes the ``data`` object
        It has either "mode = 0" (create) or "mode = 1" (load) operation depending on the given 
        inputs (see documentation of __check_define_mode and __init_mode methods for more details).
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
            self.ndims = self.data_X.shape[1]
            self.npoints = self.data_X.shape[0]
            self.pars_labels = pars_labels
            self.generic_pars_labels = utils.define_generic_pars_labels(self.pars_pos_poi, self.pars_pos_nuis)
            self.pars_bounds = pars_bounds
            self.__check_define_pars(verbose=verbose_sub)
            self.test_fraction = test_fraction
            self.define_test_fraction()
            self.output_folder = output_folder
            self.__check_define_output_files()
            self.load_on_RAM = load_on_RAM
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.load_on_RAM = load_on_RAM
            if dtype is not None:
                self.dtype = dtype
            self.__load(verbose=verbose_sub)
            self.define_test_fraction()
            if output_folder is not None:
                self.output_folder = path.abspath(output_folder)
                self.__check_define_output_files()
            self.save_json(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

        self.data_dictionary = {"X_train": np.array([[]], dtype=self.dtype), "Y_train": np.array([], dtype=self.dtype),
                                "X_val": np.array([[]],dtype=self.dtype), "Y_val": np.array([],dtype=self.dtype),
                                "X_test": np.array([[]], dtype=self.dtype), "Y_test": np.array([], dtype=self.dtype),
                                "idx_train": np.array([], dtype="int"), "idx_val": np.array([], dtype="int"), "idx_test": np.array([], dtype="int")}

    def __check_define_input_files(self):
        """
        Sets the attributes corresponding to input files
        :attr:`Data.input_h5_file <DNNLikelihood.Data.input_h5_file>`,
        :attr:`Data.input_json_file <DNNLikelihood.Data.input_json_file>`, and
        :attr:`Data.input_log_file <DNNLikelihood.Data.input_log_file>`
        depending on the value of the 
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>` attribute.
        """
        if self.input_file is None:
            self.input_h5_file = self.input_file
            self.input_json_file = self.input_file
            self.input_log_file = self.input_file
        else:
            self.input_file = path.abspath(path.splitext(input_file)[0])
            self.input_h5_file = self.input_file+".h5"
            self.input_json_file = self.input_file+".json"
            self.input_log_file = self.input_file+".log"

    def __check_define_output_files(self):
        """
        Sets the attributes corresponding to output files
        :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>`,
        :attr:`Data.output_json_file <DNNLikelihood.Data.output_json_file>`,
        :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
        depending on the value of the 
        :attr:`Data.input_file <DNNLikelihood.Data.input_file>` and
        :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>` attributes.
        """
        if self.input_file is None:
            if self.output_folder is None:
                self.output_folder = ""
            self.output_folder = path.abspath(self.output_folder)
            self.output_h5_file = path.join(self.output_folder, self.name+".h5")
            self.output_json_file = path.join(self.output_folder, self.name+".json")
            self.output_log_file = path.join(self.output_folder, self.name+".log")
        else:
            self.output_h5_file = path.join(self.output_folder, self.name+".h5")
            self.output_json_file = path.join(self.output_folder, self.name+".json")
            self.output_log_file = path.join(self.output_folder, self.name+".log")
        
    def __check_define_name(self):
        """
        Private method that defines the :attr:`Data.name <DNNLikelihood.Data.name>` attribute.
        If :attr:`Data.name <DNNLikelihood.Data.name>` is ``None`` it replaces it with
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_data"``,
        otherwise it appends the suffix "_data" (preventing duplication if it is already present).
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.name = "model_"+timestamp+"_data"
        else:
            self.name = utils.check_add_suffix(self.name, "_data")

    def __check_define_data(self, verbose=None):
        """ 
        Checks that data_X and data_Y have the same length and sets the data type to
        :attr:`Data.dtype <DNNLikelihood.Data.dtype>` (default "float64").
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.dtype is None:
            self.dtype = "float64"
        if len(self.data_X) == len(self.data_Y):
            self.data_X = self.data_X.astype(self.dtype)
            self.data_Y = self.data_Y.astype(self.dtype)
        else:
            raise Exception("data_X and data_Y have different length.")

    def __check_define_pars(self, verbose=None):
        """
        Private method that checks the consistency of the 
        :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`,
        :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`,
        :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`,
        and :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` 
        attributes and converts :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`,
        :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`, and
        and :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` to |numpy_link| arrays.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`
        is set to the value of :attr:`Data.generic_pars_labels <DNNLikelihood.Data.generic_pars_labels>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Data.ndims <DNNLikelihood.Data.ndims>`.
        """
        verbose, _ = self.set_verbosity(verbose)
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
        if self.pars_labels is None:
            self.pars_labels = self.generic_pars_labels
        elif len(self.pars_labels) != self.ndims:
            raise Exception("The number of parameters labels do not match the number of dimensions.")
        if self.pars_bounds is not None:
            self.pars_bounds = np.array(self.pars_bounds)
        else:
            self.pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        if len(self.pars_bounds) != self.ndims:
            raise Exception("The lenght of the parameters bounds array does not match the number of dimensions.")

    def __load(self,verbose=None):
        """ Loads samples as np.array (on RAM) if load_on_RAM=True or as h5py dataset (on disk) if load_on_RAM=False
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        with open(self.input_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        self.pars_pos_poi = np.array(self.pars_pos_poi)
        self.pars_pos_nuis = np.array(self.pars_pos_nuis)
        self.pars_bounds = np.array(self.pars_bounds)
        self.opened_dataset = h5py.File(self.input_h5_file, "r")
        data = self.opened_dataset["data"]
        self.data_X = data.get("X")
        self.data_Y = data.get("Y")
        if self.load_on_RAM:
            self.data_X = self.data_X[:].astype(self.dtype)
            self.data_Y = self.data_Y[:].astype(self.dtype)
            self.opened_dataset.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_json_file)[-1],
                                               path.split(self.input_log_file)[-1],
                                               path.split(self.input_h5_file)[-1]],
                               "files paths": [self.input_json_file,
                                               self.input_log_file,
                                               self.input_h5_file]}
        print('Loaded likelihood in', str(end-start), '.', show=verbose)
        if self.load_on_RAM:
            print('Samples loaded on RAM.', show=verbose)

    def define_test_fraction(self):
        if self.test_fraction is None:
            self.test_fraction = 0
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)

    def close_samples(self,verbose=None):
        """ Closes opened h5py datasets if there are any.
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
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_log_file, verbose=verbose_sub)
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #self.log[timestamp] = {"action": "saved", "file name": path.split(self.output_log_file)[-1], "file path": self.output_log_file}
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Data log file", self.output_log_file, "saved in", str(end-start), "s.", show=verbose)

    def save_json(self, overwrite=False, verbose=True):
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_json_file, verbose=verbose_sub)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved", 
                               "file name": path.split(self.output_json_file)[-1],
                               "file path": self.output_json_file}
        dictionary = utils.dic_minus_keys(self.__dict__, ["dataX", "dataY", "log", "input_file",
                                                          "train_range", "test_range", "data_dictionary",
                                                          "load_on_RAM", "verbose"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Data json file", self.output_json_file,"saved in", str(end-start), "s.",show=verbose)

    def save_h5(self, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_h5_file,verbose=verbose_sub)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_h5_file)[-1],
                               "file path": self.output_h5_file}
        start = timer()     
        h5_out = h5py.File(self.output_h5_file, "w")
        data = h5_out.create_group("data")
        data["shape"] = np.shape(self.data_X)
        data["X"] = self.data_X.astype(self.dtype)
        data["Y"] = self.data_Y.astype(self.dtype)
        h5_out.close()
        end = timer()
        self.save_log(overwrite=overwrite, verbose=verbose)
        print("Saved", str(self.npoints), "(data_X, data_Y) samples in data h5 file", self.output_h5_file,"in", end-start, "s.",show=verbose)

    def save(self, overwrite=False, verbose=None):
        """

        :class:`Data <DNNLikelihood.Data>` objects are saved according to the following table.

        +-------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        | Saved | Atrributes                                                                                                       | Method                                                                                           |
        +=======+==================================================================================================================+==================================================================================================+
        |   X   | :attr:`Data.input_file <DNNLikelihood.Data.input_file>`                                                |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.train_range <DNNLikelihood.Data.train_range>`                                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.test_range <DNNLikelihood.Data.test_range>`                                                          |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>`                                                |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.load_on_RAM <DNNLikelihood.Data.load_on_RAM>`                                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.verbose <DNNLikelihood.Data.verbose>`                                                                |                                                                                                  |
        +-------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        |   ✔   | :attr:`Data.name <DNNLikelihood.Data.name>`                                                                      | :meth:`Data.save_json <DNNLikelihood.Data.save_json>`                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.ndims <DNNLikelihood.Data.ndims>`                                                                      |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.npoints <DNNLikelihood.Data.npoints>`                                                                |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.dtype <DNNLikelihood.Data.dtype>`                                                                |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.generic_pars_labels <DNNLikelihood.Data.generic_pars_labels>`                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.output_folder <DNNLikelihood.Data.output_folder>`                                                    |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>`                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.output_json_file <DNNLikelihood.Data.output_json_file>`                                    |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`                                      |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`                                                      |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`                                                    |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`                                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>`                                                        |                                                                                                  |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>`                                                    |                                                                                                  |
        +-------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        |   ✔   | :attr:`Data.log <DNNLikelihood.Data.log>`                                                                        | :meth:`Data.save_log <DNNLikelihood.Data.save_log>`                                    |
        +-------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+
        |   ✔   | :attr:`Data.data_X <DNNLikelihood.Data.data_X>`                                                                  | :meth:`Data.save_h5 <DNNLikelihood.Data.save_h5>`                                      |
        |       |                                                                                                                  |                                                                                                  |
        |       | :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>`                                                                  |                                                                                                  |
        +-------+------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------+

        This methos calls in order the :meth:`Likelihood.save_json <DNNLikelihood.Likelihood.save_json>`, :meth:`Likelihood.save_h5 <DNNLikelihood.Likelihood.save_h5>` and
        :meth:`Likelihood.save_log <DNNLikelihood.Likelihood.save_log>` methods.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Data.output_json_file <DNNLikelihood.Data.output_json_file>`
            - :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>`
            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_json(overwrite=overwrite, verbose=verbose)
        self.save_h5(overwrite=overwrite, verbose=verbose)
        self.save_log(overwrite=overwrite, verbose=verbose)

    def generate_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        show_prints.verbose = verbose
        #Generate new indices
        np.random.seed(seed)
        idx_train = np.random.choice(self.train_range, npoints_train+npoints_val, replace=False)
        idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)]
        #Update indices in data_dictionary
        self.data_dictionary["idx_train"] = idx_train
        self.data_dictionary["idx_val"] = idx_val

    def generate_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        show_prints.verbose = verbose
        start = timer()
        # Generate new indices
        self.generate_train_indices(npoints_train, npoints_val, seed, verbose=verbose)
        # Update data dictionary
        self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype)
        self.data_dictionary["Y_train"] = self.data_Y[self.data_dictionary["idx_train"]].astype(self.dtype)
        self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype)
        self.data_dictionary["Y_val"] = self.data_Y[self.data_dictionary["idx_val"]].astype(self.dtype)
        end = timer()
        show_prints.verbose = verbose
        print("Generated", str(npoints_train), "(X_train, Y_train) samples and ", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.")

    def update_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        show_prints.verbose = verbose
        #Check existing indices
        existing_train = np.sort(np.concatenate((self.data_dictionary["idx_train"], self.data_dictionary["idx_val"])))
        #Generate new indices
        np.random.seed(seed)
        idx_train = np.random.choice(np.setdiff1d(np.array(self.train_range), existing_train), npoints_train+npoints_val, replace=False)
        if np.size(idx_train) != 0:
            idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)]
        else:
            idx_val = idx_train
        #Update indices in data_dictionary
        self.data_dictionary["idx_train"] = np.sort(np.concatenate((self.data_dictionary["idx_train"],idx_train)))
        self.data_dictionary["idx_val"] = np.sort(np.concatenate((self.data_dictionary["idx_val"],idx_val)))
        #Return new indices
        return [idx_train, idx_val]

    def update_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        show_prints.verbose = verbose
        start = timer()
        #Update number of points to generate
        [npoints_train, npoints_val] = [(i > 0) * i for i in [npoints_train-len(self.data_dictionary["Y_train"]), 
                                                                            npoints_val-len(self.data_dictionary["Y_val"])]]
        # Generate new indices
        idx_train, idx_val = self.update_train_indices(npoints_train, npoints_val, seed, verbose=verbose)
        # Update data dictionary
        if idx_train != []:
            if self.data_dictionary["X_train"].size == 0:
                self.data_dictionary["X_train"] = self.data_X[idx_train].astype(self.dtype)
                self.data_dictionary["Y_train"] = self.data_Y[idx_train].astype(self.dtype)
            else:
                self.data_dictionary["X_train"] = np.concatenate((self.data_dictionary["X_train"],self.data_X[idx_train])).astype(self.dtype)
                self.data_dictionary["Y_train"] = np.concatenate((self.data_dictionary["Y_train"],self.data_Y[idx_train])).astype(self.dtype)
        if idx_val != []:
            if self.data_dictionary["X_val"].size == 0:
                self.data_dictionary["X_val"] = self.data_X[idx_val].astype(self.dtype)
                self.data_dictionary["Y_val"] = self.data_Y[idx_val].astype(self.dtype)
            else:
                self.data_dictionary["X_val"] = np.concatenate((self.data_dictionary["X_val"], self.data_X[idx_val])).astype(self.dtype)
                self.data_dictionary["Y_val"] = np.concatenate((self.data_dictionary["Y_val"], self.data_Y[idx_val])).astype(self.dtype)
        end = timer()
        show_prints.verbose = verbose
        print("Added", str(npoints_train), "(X_train, Y_train) samples and", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.")

    def generate_test_indices(self, npoints_test, verbose=None):
        show_prints.verbose = verbose
        #Check existing indices
        n_existing_test = len(self.data_dictionary["idx_test"])
        #Generate new indices
        idx_test = np.array(self.test_range)[range(
            n_existing_test, n_existing_test+npoints_test)]
        #Update indices in data_dictionary
        self.data_dictionary["idx_test"] = np.concatenate((self.data_dictionary["idx_test"],idx_test))
        #Return new indices
        return idx_test

    def generate_test_data(self, npoints_test, verbose=None):
        show_prints.verbose = verbose
        start = timer()
        #Update number of points to generate
        npoints_test = npoints_test-len(self.data_dictionary["Y_test"])
        # Generate new indices
        idx_test = self.generate_test_indices(npoints_test, verbose=verbose)
        # Update data dictionary
        if idx_test != []:
            if self.data_dictionary["X_test"].size == 0:
                self.data_dictionary["X_test"] = self.data_X[idx_test].astype(self.dtype)
                self.data_dictionary["Y_test"] = self.data_Y[idx_test].astype(self.dtype)
            else:
                self.data_dictionary["X_test"] = np.concatenate((self.data_dictionary["X_test"], self.data_X[idx_test])).astype(self.dtype)
                self.data_dictionary["Y_test"] = np.concatenate((self.data_dictionary["Y_test"], self.data_Y[idx_test])).astype(self.dtype)
        end = timer()
        show_prints.verbose = verbose
        print("Added", str(npoints_test), "(X_test, Y_test) samples in", end-start, "s.")

    def compute_sample_weights(self, sample, bins=100, power=1):
        hist, edges = np.histogram(sample, bins=bins)
        hist = np.where(hist < 5, 5, hist)
        tmp = np.digitize(sample, edges, right=True)
        W = 1/np.power(hist[np.where(tmp == bins, bins-1, tmp)], power)
        W = W/np.sum(W)*len(sample)
        return W

    def define_scalers(self, X_train, Y_train, scalerX_bool, scalerY_bool, verbose=None):
        show_prints.verbose = verbose
        if scalerX_bool:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            scalerX.fit(X_train)
        else:
            scalerX = StandardScaler(with_mean=False, with_std=False)
            scalerX.fit(X_train)
        if scalerY_bool:
            scalerY = StandardScaler(with_mean=True, with_std=True)
            scalerY.fit(Y_train.reshape(-1, 1))
        else:
            scalerY = StandardScaler(with_mean=False, with_std=False)
            scalerY.fit(Y_train.reshape(-1, 1))
        return [scalerX, scalerY]

    #def data_to_generate(self, npoints_train, npoints_val, npoints_test):
    #    map = {"idx_train": npoints_train, "idx_val": npoints_val, "idx_test": npoints_test}
    #    result = []
    #    for key,value in map.items():
    #        result.append(value-len(self.data_dictionary[key]))
    #    result = [(i > 0) * i for i in result]
    #    return result
