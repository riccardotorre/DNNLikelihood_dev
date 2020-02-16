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
                 test_fraction = 1,
                 name = None,
                 data_sample_input_filename=None,
                 data_sample_output_filename=None,
                 load_on_RAM = False
                 ):
        """Initializes the Data_sample object
        It has either "mode = 0" (create) or "mode = 1" (load) operation depending on the given 
        inputs (see documentation of __check_define_mode__ and __init_mode__ methods for more details).
        """
        self.data_X = data_X
        self.data_Y = data_Y
        self.test_fraction = test_fraction
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if name is None:
            self.name = "data_sample_"+timestamp
        else:
            self.name = name
        self.data_sample_input_filename = data_sample_input_filename
        self.data_sample_output_filename = data_sample_output_filename
        self.load_on_RAM = load_on_RAM

        self.__check_define_mode__()
        self.__init_mode__()
        self.__check_data__()
        self.define_test_fraction()
        self.data_dictionary = {"X_train": np.array([[]]), "Y_train": np.array([]),
                                "X_val": np.array([[]]), "Y_val": np.array([]),
                                "X_test": np.array([[]]), "Y_test": np.array([]),
                                "idx_train": [], "idx_val": [], "idx_test": []}
    
    def __check_define_mode__(self):
        """Checks which parameters have been passed and defines the working mode. Modes are:
        0 = create (see documentation of __init_0__)
        1 = load (see documentation of __init_1__)
        """
        # Checks
        if (self.data_X is not None and self.data_Y is None) or (self.data_X is None and self.data_Y is not None):
            raise utility.DataError("You should specify either both or none of data_X and data_Y.\nPlease change parameters and execute again.")
            # MyError(self, self.__check_define_mode__)
        if self.data_X is not None:
            if self.load_on_RAM:
                print("When providing data load_on_RAM is automatically set to False (default value).")
            if self.data_sample_input_filename is not None:
                print("When providing data data_sample_input_filename is automatically set to None (default value).")
            self.mode = 0
            #print("Working in 'create' mode (self.mode = 0)")
        if self.data_X is None:
            if self.data_sample_input_filename is None and self.load_on_RAM is False:
                raise utility.DataError("No data neither file available. Please input either data or file and execute again.")
                #MyError(self, self.__check_define_mode__)
            elif self.data_sample_input_filename is None and self.load_on_RAM:
                raise utility.DataError("To import samples please specify data_sample_input_filename.\nPlease change parameters and execute again.")
                #MyError(self, self.__check_define_mode__)
            elif self.data_sample_input_filename is not None:
                self.mode = 1
                #print("Working in 'load' mode (self.mode = 1).\nDepending on the value of load_on_RAM samples are either loaded into RAM as np.arrays or read from file as h5py dataset.")
        try:
            self.mode
        except:
            raise utility.DataError("Unable to determine working mode. Please check input parameters.")
            #MyError(self, self.__check_define_mode__)
            #return

    def __init_mode__(self):
        """ Initializes according to self.mode (determined by the method __check_define_mode__)
        Mode 0 = create: data_X and data_Y are given as input.
            Data_sample object is created, defining npoints and ndim.
        Mode 1 = load as np.array: data_sample_input_filename is given as input. and load_on_RAM is True.
            If load_on_RAM=True Data_sample object is created, name, data_X, data_Y, npoints and ndim are loaded from file as np.arrays (loaded into RAM).
            If load_on_RAM=False Data_sample object is created, name, data_X, data_Y, npoints and ndim are read from file as h5py dataset (on disk).
        """
        if self.mode == 0:
            self.__create_samples__()
        elif self.mode == 1:
            self.__load_samples__()
    
    def __create_samples__(self):
        self.npoints = self.data_X.shape[0]
        self.ndim = self.data_X.shape[1]

    def __load_samples__(self):
        """ Loads samples as np.array (on RAM) if load_on_RAM=True or as h5py dataset (on disk) if load_on_RAM=False
        """
        start = timer()
        self.opened_dataset = h5py.File(self.data_sample_input_filename, "r")
        ds_list = list(self.opened_dataset.keys())
        ds_list.remove("data")
        self.name = ds_list[0]
        data = self.opened_dataset["data"]
        self.npoints = data.get("shape")[0]
        self.ndim = data.get("shape")[1]
        self.data_X = data.get("X")
        self.data_Y = data.get("Y")
        self.test_fraction = data.get("test_fraction")[0]
        if self.load_on_RAM:
            self.data_X = self.data_X[:]
            self.data_Y = self.data_Y[:]
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

    def __check_data__(self):
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
        h5_out = h5py.File(data_sample_filename)
        h5_out.create_group(self.name)
        data = h5_out.create_group("data")
        data["shape"] = np.shape(self.data_X)
        data["X"] = self.data_X
        data["Y"] = self.data_Y
        data["test_fraction"] = np.array([self.test_fraction])
        h5_out.close()
        end = timer()
        print("Saved", str(self.npoints), "(data_X, data_Y) samples in file", data_sample_filename,"in", end-start, "s.")

    def generate_indices(self, npoints_train, npoints_val, npoints_test, seed, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        #Generate new indices
        np.random.seed(seed)
        idx_train = np.random.choice(self.train_range, npoints_train+npoints_val, replace=False)
        idx_train, idx_val = train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)
        idx_test = np.array(self.test_range)[range(npoints_test)]
        idx_train, idx_val, idx_test = [np.sort(idx_train).tolist(), np.sort(idx_val).tolist(), np.sort(idx_test).tolist()]
        #Update indices in data_dictionary
        self.data_dictionary["idx_train"] = np.sort(idx_train).tolist()
        self.data_dictionary["idx_val"] = np.sort(idx_val).tolist()
        self.data_dictionary["idx_test"] = np.sort(idx_test).tolist()

    def generate_data(self, npoints_train, npoints_val, npoints_test, seed, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        # Generate new indices
        self.generate_indices(npoints_train, npoints_val, npoints_test, seed, verbose=verbose)
        # Update data dictionary
        self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]]
        self.data_dictionary["Y_train"] = self.data_Y[self.data_dictionary["idx_train"]]
        self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]]
        self.data_dictionary["Y_val"] = self.data_Y[self.data_dictionary["idx_val"]]
        self.data_dictionary["X_test"] = self.data_X[self.data_dictionary["idx_test"]]
        self.data_dictionary["Y_test"] = self.data_Y[self.data_dictionary["idx_test"]]
        end = timer()
        ShowPrints = verbose
        print("Generated", str(npoints_train), "(X_train, Y_train) samples,", str(npoints_val),
              "(X_val, Y_val) samples, and", str(npoints_test), "(X_test, Y_test) samples in", end-start,"s.")

    def update_indices(self, npoints_train, npoints_val, npoints_test, seed, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        #Check existing indices
        existing_train = np.sort(np.concatenate((np.array(self.data_dictionary["idx_train"]), np.array(self.data_dictionary["idx_val"]))))
        n_existing_test = len(np.array(self.data_dictionary["idx_test"]))
        #Generate new indices
        np.random.seed(seed)
        idx_train = np.random.choice(np.setdiff1d(np.array(self.train_range), existing_train), npoints_train+npoints_val, replace=False)
        if np.size(idx_train) != 0:
            idx_train, idx_val = train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val)
        else:
            idx_val = idx_train
        idx_test = np.array(self.test_range)[range(n_existing_test,n_existing_test+npoints_test)]
        idx_train, idx_val, idx_test = [np.sort(idx_train).tolist(), np.sort(idx_val).tolist(), np.sort(idx_test).tolist()]
        #Update indices in data_dictionary
        self.data_dictionary["idx_train"] = np.sort(self.data_dictionary["idx_train"]+idx_train).tolist()
        self.data_dictionary["idx_val"] = np.sort(self.data_dictionary["idx_val"]+idx_val).tolist()
        self.data_dictionary["idx_test"] = np.sort(self.data_dictionary["idx_test"]+idx_test).tolist()
        #Return new indices
        return [idx_train, idx_val, idx_test]

    def update_data(self, npoints_train, npoints_val, npoints_test, seed, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        #Update number of points to generate
        [npoints_train, npoints_val, npoints_test] = [(i > 0) * i for i in [npoints_train-len(self.data_dictionary["Y_train"]), 
                                                                            npoints_val-len(self.data_dictionary["Y_val"]), 
                                                                            npoints_test-len(self.data_dictionary["Y_test"])]]
        # Generate new indices
        idx_train, idx_val, idx_test = self.update_indices(npoints_train, npoints_val, npoints_test, seed, verbose=verbose)
        # Update data dictionary
        if idx_train != []:
            if self.data_dictionary["X_train"].size == 0:
                self.data_dictionary["X_train"] = self.data_X[idx_train]
                self.data_dictionary["Y_train"] = self.data_Y[idx_train]
            else:
                self.data_dictionary["X_train"] = np.concatenate((self.data_dictionary["X_train"],self.data_X[idx_train]))
                self.data_dictionary["Y_train"] = np.concatenate((self.data_dictionary["Y_train"],self.data_Y[idx_train]))
        if idx_val != []:
            if self.data_dictionary["X_val"].size == 0:
                self.data_dictionary["X_val"] = self.data_X[idx_val]
                self.data_dictionary["Y_val"] = self.data_Y[idx_val]
            else:
                self.data_dictionary["X_val"] = np.concatenate((self.data_dictionary["X_val"], self.data_X[idx_val]))
                self.data_dictionary["Y_val"] = np.concatenate((self.data_dictionary["Y_val"], self.data_Y[idx_val]))
        if idx_test != []:
            if self.data_dictionary["X_test"].size == 0:
                self.data_dictionary["X_test"] = self.data_X[idx_test]
                self.data_dictionary["Y_test"] = self.data_Y[idx_test]
            else:
                self.data_dictionary["X_test"] = np.concatenate((self.data_dictionary["X_test"],self.data_X[idx_test]))
                self.data_dictionary["Y_test"] = np.concatenate((self.data_dictionary["Y_test"],self.data_Y[idx_test]))
        end = timer()
        ShowPrints = verbose
        print("Added", str(npoints_train), "(X_train, Y_train) samples,", str(npoints_val),
              "(X_val, Y_val) samples, and", str(npoints_test), "(X_test, Y_test) samples in", end-start,"s.")

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
    
