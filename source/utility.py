import inspect
import os
import sys
import pickle
from timeit import default_timer as timer

def check_rename_file(filename):
    if os.path.exists(filename):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("The file", filename, "already exists.")
        file, extension = os.path.splitext(filename)
        filename = file+"_"+now+extension
        print("In order not to overwrite saving to the file", filename)
    else:
        print("Saving to the file", filename)
    return filename

def save_samples(allsamples, logprob_values, data_sample_filename, name):
    data_sample_filename = check_rename_file(data_sample_filename)
    data_sample_shape = np.shape(allsamples)
    data_sample_name = name+"_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pickle_out = open(data_sample_filename, "wb")
    start = timer()
    pickle.dump(data_sample_timestamp, pickle_out, protocol=4)
    pickle.dump(data_sample_shape, pickle_out, protocol=4)
    pickle.dump(allsamples, pickle_out, protocol=4)
    pickle.dump(logprob_values, pickle_out, protocol=4)
    end = timer()
    pickle_out.close()
    statinfo = os.stat(data_sample_filename)
    print("File saved in", end-start,"seconds.\nFile size is", statinfo.st_size, ".")

def set_param(obj_name, par_name):
    if eval(par_name) is None:
        exec("%s = %s" % (par_name, obj_name+"."+par_name))
    else:
        setattr(eval(obj_name), par_name, eval(par_name))
    return eval(par_name)
#
#class BlockPrints:
#    def __init__(self):
#        self.initial_stout = sys.stdout
#        self.status = "unblocked"
#    def blockPrint(self):
#        self.initial_stout = sys.stdout
#        sys.stdout = open(os.devnull, 'w')
#        self.status = "blocked"
#    def resetPrint(self):
#        sys.stdout.close()
#        sys.stdout = self.initial_stout
#        self.status = "unblocked"
#    def __enter__(self):
#        self.blockPrint()
#    def __exit__(self, exc_type, exc_val, exc_tb):
#        self.resetPrint()
#    def setprints(self, verbose):
#        if verbose:
#            if self.status == "blocked":
#                self.resetPrint()
#        else:
#            self.blockPrint()

#class BlockPrints:
#    def __enter__(self):
#        self._original_stdout = sys.stdout
#        sys.stdout = open(os.devnull, "w")
#    def __exit__(self, exc_type, exc_val, exc_tb):
#        sys.stdout.close()
#        sys.stdout = self._original_stdout
#
#class HandlePrint():
#    def __init__(self):
#        self.initial_stout = sys.stdout
#    def blockPrint(self):
#        sys.stdout = open(os.devnull, 'w')
#    # Restore
#    def resetPrint(self):
#        sys.stdout = self.initial_stout
