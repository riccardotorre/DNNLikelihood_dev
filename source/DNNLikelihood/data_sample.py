__all__ = ["Data_sample"]

import numpy as np
import h5py
from datetime import datetime
from timeit import default_timer as timer
import builtins

from . import utility

ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

#class MyError(Exception):
#    def __init__(self, obj, method):
#        print('Debug info:', repr(obj.data), method.__name__)
#        raise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Data_sample(object):
    """Class to initialize the Data_sample object
    """
    def __init__(self,
                 data_X = None,
                 data_Y = None,
                 dtype="float64",
                 pars_pos_poi=None,
                 pars_pos_nuis=None,
                 pars_labels=None,
                 test_fraction = None,
                 name = None,
                 data_sample_input_filename=None,
                 data_sample_output_filename=None,
                 load_on_RAM = False
                 ):
        """Initializes the Data_sample object
        It has either "mode = 0" (create) or "mode = 1" (load) operation depending on the given 
        inputs (see documentation of __check_define_mode and __init_mode methods for more details).
        """
        self.dtype = dtype
        self.data_X = data_X
        self.data_Y = data_Y
        self.pars_pos_poi = np.array(pars_pos_poi)
        self.pars_pos_nuis = np.array(pars_pos_nuis)
        self.pars_labels = pars_labels
        if test_fraction is None:
            self.test_fraction = 0
        else:
            self.test_fraction = test_fraction
        self.name = name
        self.data_sample_input_filename = data_sample_input_filename
        self.data_sample_output_filename = data_sample_output_filename
        self.load_on_RAM = load_on_RAM

        self.__check_define_mode()
        self.__init_mode()
        self.__check_data()
        self.__check_define_name()
        self.__check_define_pars()
        self.define_test_fraction()
        self.data_dictionary = {"X_train": np.array([[]], dtype=self.dtype), "Y_train": np.array([], dtype=self.dtype),
                                "X_val": np.array([[]],dtype=self.dtype), "Y_val": np.array([],dtype=self.dtype),
                                "X_test": np.array([[]], dtype=self.dtype), "Y_test": np.array([], dtype=self.dtype),
                                "idx_train": np.array([], dtype="int"), "idx_val": np.array([], dtype="int"), "idx_test": np.array([], dtype="int")}
    
    def __check_define_name(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.name is None:
            self.name = "data_sample_"+timestamp

    def __check_define_pars(self):
        if self.pars_pos_nuis is None and self.pars_pos_poi is None:
            print(
                "The positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified.\
                Assuming all parameters are parameters of interest.")
            self.pars_pos_nuis = []
            self.pars_pos_poi = list(range(self.ndim))
        elif self.pars_pos_nuis is not None and self.pars_pos_poi is None:
            print(
                "Only the positions of the nuisance parameters have been specified.\
                Assuming all other parameters are parameters of interest.")
            self.pars_pos_poi = np.setdiff1d(np.array(range(self.ndim)), np.array(self.pars_pos_nuis)).tolist()
        elif self.pars_pos_nuis is None and self.pars_pos_poi is not None:
            print(
                "Only the positions of the parameters of interest.\
                Assuming all other parameters are nuisance parameters.")
            self.pars_pos_nuis = np.setdiff1d(np.array(range(self.ndim)), np.array(self.pars_pos_poi)).tolist()
        else:
            if len(self.pars_pos_poi)+len(self.pars_pos_nuis) != self.ndim:
                raise Exception("The number of parameters positions do not match the number of dimensions.")
        if self.pars_labels is not None:
            if len(self.pars_labels) != self.ndim:
                raise Exception("The number of parameters labels do not match the number of dimensions.")
        else:
            self.pars_labels = []
            i_poi = 1
            i_nuis = 1
            for i in range(len(self.pars_pos_poi)+len(self.pars_pos_nuis)):
                if i in self.pars_pos_poi:
                    self.pars_labels.append(r"$\theta_{%d}$" % i_poi)
                    i_poi = i_poi+1
                else:
                    self.pars_labels.append(r"$\nu_{%d}$" % i_nuis)
                    i_nuis = i_nuis+1

    def __check_define_mode(self):
        """Checks which parameters have been passed and defines the working mode. Modes are:
        0 = create (see documentation of __init_0__)
        1 = load (see documentation of __init_1__)
        """
        # Checks
        if (self.data_X is not None and self.data_Y is None) or (self.data_X is None and self.data_Y is not None):
            raise Exception("You should specify either both or none of data_X and data_Y.\nPlease change parameters and execute again.")
            # MyError(self, self.__check_define_mode)
        if self.data_X is not None:
            if self.load_on_RAM:
                print("When providing data load_on_RAM is automatically set to False (default value).")
            if self.data_sample_input_filename is not None:
                print("When providing data data_sample_input_filename is automatically set to None (default value).")
            self.mode = 0
            #print("Working in 'create' mode (self.mode = 0)")
        if self.data_X is None:
            if self.data_sample_input_filename is None and self.load_on_RAM is False:
                raise Exception("No data neither file available. Please input either data or file and execute again.")
                #MyError(self, self.__check_define_mode)
            elif self.data_sample_input_filename is None and self.load_on_RAM:
                raise Exception("To import samples please specify data_sample_input_filename.\nPlease change parameters and execute again.")
                #MyError(self, self.__check_define_mode)
            elif self.data_sample_input_filename is not None:
                self.mode = 1
                #print("Working in 'load' mode (self.mode = 1).\nDepending on the value of load_on_RAM samples are either loaded into RAM as np.arrays or read from file as h5py dataset.")
        try:
            self.mode
        except:
            raise Exception("Unable to determine working mode. Please check input parameters.")
            #MyError(self, self.__check_define_mode)
            #return

    def __init_mode(self):
        """ Initializes according to self.mode (determined by the method __check_define_mode)
        Mode 0 = create: data_X and data_Y are given as input.
            Data_sample object is created, defining npoints and ndim.
        Mode 1 = load as np.array: data_sample_input_filename is given as input. and load_on_RAM is True.
            If load_on_RAM=True Data_sample object is created, name, data_X, data_Y, npoints and ndim are loaded from file as np.arrays (loaded into RAM).
            If load_on_RAM=False Data_sample object is created, name, data_X, data_Y, npoints and ndim are read from file as h5py dataset (on disk).
        """
        if self.mode == 0:
            self.__create_samples()
        elif self.mode == 1:
            self.__load_samples()
    
    def __create_samples(self):
        self.data_X = self.data_X.astype(self.dtype)
        self.data_Y = self.data_Y.astype(self.dtype)
        self.npoints = self.data_X.shape[0]
        self.ndim = self.data_X.shape[1]

    def __load_samples(self):
        """ Loads samples as np.array (on RAM) if load_on_RAM=True or as h5py dataset (on disk) if load_on_RAM=False
        """
        start = timer()
        self.opened_dataset = h5py.File(self.data_sample_input_filename, "r")
        ds_list = list(self.opened_dataset.keys())
        ds_list.remove("data")
        self.name = ds_list[0]
        parameters = self.opened_dataset["parameters"]
        self.pars_pos_poi = parameters["pars_pos_poi"][:]
        self.pars_pos_nuis = parameters["pars_pos_nuis"][:]
        self.pars_labels = [str(i, 'utf-8') for i in parameters["pars_labels"]]
        data = self.opened_dataset["data"]
        self.npoints = data.get("shape")[0]
        self.ndim = data.get("shape")[1]
        self.data_X = data.get("X")
        self.data_Y = data.get("Y")
        self.test_fraction = data.get("test_fraction")[0]
        if self.load_on_RAM:
            self.data_X = self.data_X[:].astype(self.dtype)
            self.data_Y = self.data_Y[:].astype(self.dtype)
            self.opened_dataset.close()
            end = timer()
            print("Loaded into RAM", str(self.npoints), "(data_X, data_Y) samples as np.arrays from file", self.data_sample_output_filename, "in", end-start, "s.")
        else:
            end = timer()
            print("Opened h5py dataset with", str(self.npoints), "(data_X, data_Y) samples from file", self.data_sample_output_filename, "in", end-start, "s.")

    def define_test_fraction(self):
        #print(self.npoints)
        #print(self.test_fraction)
        #print(round(self.npoints*(1-self.test_fraction)))
        #print(round(self.npoints*(1-self.test_fraction))))
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)

    def close_samples(self,verbose=True):
        """ Closes opened h5py datasets if there are any.
        """
        global ShowPrints
        ShowPrints = verbose
        try:
            self.opened_dataset.close()
            del(self.opened_dataset)
            print("Closed", self.data_sample_input_filename)
        except:
            print("No dataset to close.")

    def __check_data(self):
        """ Checks that data_X and data_Y have the same length
        """
        if not (len(self.data_X) == len(self.data_Y)):
            print("data_X and data_Y have different length.")

    def save_samples(self,verbose=True):
        """ Save samples to data_sample_output_filename as h5 file
        """
        global ShowPrints
        ShowPrints = verbose
        data_sample_filename = utility.check_rename_file(self.data_sample_output_filename)
        start = timer()     
        h5_out = h5py.File(data_sample_filename,"w")
        h5_out.create_group(self.name)
        parameters = h5_out.create_group("parameters")
        parameters["pars_pos_poi"] = self.pars_pos_poi
        parameters["pars_pos_nuis"] = self.pars_pos_nuis
        parameters["pars_labels"] = np.string_(self.pars_labels)
        data = h5_out.create_group("data")
        data["shape"] = np.shape(self.data_X)
        data["X"] = self.data_X.astype(self.dtype)
        data["Y"] = self.data_Y.astype(self.dtype)
        data["test_fraction"] = np.array([self.test_fraction])
        h5_out.close()
        end = timer()
        print("Saved", str(self.npoints), "(data_X, data_Y) samples in file", data_sample_filename,"in", end-start, "s.")

    def generate_train_indices(self, npoints_train, npoints_val, seed, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        #Generate new indices
        np.random.seed(seed)
        idx_train = np.random.choice(self.train_range, npoints_train+npoints_val, replace=False)
        idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)]
        #Update indices in data_dictionary
        self.data_dictionary["idx_train"] = idx_train
        self.data_dictionary["idx_val"] = idx_val

    def generate_train_data(self, npoints_train, npoints_val, seed, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        # Generate new indices
        self.generate_train_indices(npoints_train, npoints_val, seed, verbose=verbose)
        # Update data dictionary
        self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype)
        self.data_dictionary["Y_train"] = self.data_Y[self.data_dictionary["idx_train"]].astype(self.dtype)
        self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype)
        self.data_dictionary["Y_val"] = self.data_Y[self.data_dictionary["idx_val"]].astype(self.dtype)
        end = timer()
        ShowPrints = verbose
        print("Generated", str(npoints_train), "(X_train, Y_train) samples and ", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.")

    def update_train_indices(self, npoints_train, npoints_val, seed, verbose=False):
        global ShowPrints
        ShowPrints = verbose
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

    def update_train_data(self, npoints_train, npoints_val, seed, verbose=True):
        global ShowPrints
        ShowPrints = verbose
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
        ShowPrints = verbose
        print("Added", str(npoints_train), "(X_train, Y_train) samples and", str(npoints_val),"(X_val, Y_val) samples in", end-start,"s.")

    def generate_test_indices(self, npoints_test, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        #Check existing indices
        n_existing_test = len(self.data_dictionary["idx_test"])
        #Generate new indices
        idx_test = np.array(self.test_range)[range(
            n_existing_test, n_existing_test+npoints_test)]
        #Update indices in data_dictionary
        self.data_dictionary["idx_test"] = np.concatenate((self.data_dictionary["idx_test"],idx_test))
        #Return new indices
        return idx_test

    def generate_test_data(self, npoints_test, verbose=True):
        global ShowPrints
        ShowPrints = verbose
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
        ShowPrints = verbose
        print("Added", str(npoints_test), "(X_test, Y_test) samples in", end-start, "s.")

    def compute_sample_weights(self, sample, bins=100, power=1):
        hist, edges = np.histogram(sample, bins=bins)
        hist = np.where(hist < 5, 5, hist)
        tmp = np.digitize(sample, edges, right=True)
        W = 1/np.power(hist[np.where(tmp == bins, bins-1, tmp)], power)
        W = W/np.sum(W)*len(sample)
        return W

    def define_scalers(self, X_train, Y_train, scalerX_bool, scalerY_bool, verbose=False):
        global ShowPrints
        ShowPrints = verbose
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
    
