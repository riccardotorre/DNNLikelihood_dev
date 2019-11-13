__all__ = ["Data_sample"]

import builtins
ShowPrints = True


def print(*args, **kwargs):
    global ShowPrints
    if ShowPrints:
        return builtins.print(*args, **kwargs)
    else:
        return None
        
import numpy as np
from . import utility

class Data_sample(object):
    """Class to initialize the Sampling object
    """
    def __init__(self,
                 data_X = None,
                 data_logprob = None,
                 name = None,
                 data_sample_input_filename = None,
                 data_sample_output_filename=None,
                 import_from_file = False,
                 npoints = None,
                 shuffle = False
                 ):
        """Initializes the sampling object from a table. 
        The table should be a numpy array nxm with n the number of points and m the number of parameters +2 
        Each row should contain the values of the parameters (both physics and nuisance parameters)
        and in the last two columns the values of logprob and loglikelihood.
        By default the object is initialized with an empty matrix.
        kwargs = {"loglik": None, "logprior": None, "ndim": None, "nwalkers": None, "nsteps": None, "backend_filename": "None", "chains_name": "None", "new_sampler": False, "parallel_CPU": False, "vectorize": False, "emcee_args": []}
        """
        self.data_X = data_X
        self.data_logprob = data_logprob
        if self.name == None:
            self.name = "data_sample"
        else:
            self.name = name
        self.data_sample_input_filename = data_sample_input_filename
        self.data_sample_output_filename = data_sample_output_filename
        self.import_from_file = import_from_file
        self.npoints = npoints
        self.shuffle = shuffle
        if self.npoints is None:
            self.npoints = len(data_X)

        self.__check_data__()

        if self.data_X is None:
            if self.data_sample_input_filename is None and self.import_from_file is False:
                print("No data neither file available. Please input data or file to load data.")
                return None
            elif self.data_sample_input_filename is None and self.import_from_file:
                print("To import samples please specify a data sample file.")
                return None
            elif self.data_sample_input_filename is not None and self.import_from_file:
                self.load_samples()
        if self.dataX is not None:
            if self.import_from_file:
                print("When providing data you should set import_from_file=False.\nContinuing with import_from_file=False.")
                self.import_from_file = False
        
    def __set_param__(self, par_name, par_val):
        if par_val is None:
            par_val = eval("self."+par_name)
            print("No parameter"+par_val+"specified. Its value has been set to",
                  par_val, ".")
        else:
            setattr(self, par_name, par_val)

    def __check_data__(self):
        if self.data_X is not None:
            if len(self.data_logprob) != len(self.data_X):
                print("Provided X and Y data have different length.")
                return None
        if self.data_logprob is not None:
            if len(self.data_logprob) != len(self.data_X):
                print("Provided X and Y data have different length.")
                return None

    def set_samples(self, data_X, data_logprob):
        self.data_X = data_X
        self.data_logprob = data_logprob
        if self.npoints is None:
            self.npoints = len(data_X)
        self.__check_data__()
    
    def load_samples(self, data_sample_input_filename=None, npoints=None, shuffle=None):
        self.__set_param__("data_sample_input_filename",data_sample_input_filename)
        self.__set_param__("shuffle", shuffle)
        self.__set_param__("npoints", npoints)
        start = timer()
        pickle_in = open(data_sample_input_filename, 'rb')
        self.name = pickle.load(pickle_in)
        self.ntotpoints = pickle.load(pickle_in)
        if npoints is "all":
            self.data_X = pickle.load(pickle_in)
            if shuffle:
                indices = np.random.choice(np.arange(len(self.data_X)), size=len(self.data_X), replace=False)
            else:
                indices = np.arange(len(self.data_X))
            self.data_X = self.data_X[indices]
            self.data_logprob = pickle.load(pickle_in)[indices]
            self.npoints = len(self.data_X)
            end = timer()
            print("Imported", str(npoints), "(data_X, data_logprob) samples in",end-start,"s.")
        else:
            self.data_X = pickle.load(pickle_in)
            if shuffle:
                indices = np.random.choice(np.arange(len(self.data_X)), size=int(npoints), replace=False)
            else:
                indices = np.arange(npoints)
            self.data_X = self.data_X[indices]
            self.data_logprob = pickle.load(pickle_in)[indices]
            end = timer()
            print("Imported", str(len(indices)), "(data_X, data_logprob) samples in",end-start,"s.")
        pickle_in.close()

    def save_samples(self, data_sample_output_filename=None, name=None):
        self.__set_param__("name", name)
        self.__set_param__("data_sample_output_filename",data_sample_output_filename)
        data_sample_output_filename = utility.check_rename_file(data_sample_output_filename)
        utility.save_samples(allsamples, logprob_values, data_sample_output_filename,name)
