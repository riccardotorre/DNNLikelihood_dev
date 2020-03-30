__all__ = ["Lik"]

from os import path, stat
import builtins
from datetime import datetime
import pickle
import cloudpickle
import ipywidgets as widgets
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
try:
    from jupyterthemes import jtplot
except:
    print("No module named 'jupyterthemes'. Continuing without.\nIf you wish to customize jupyter notebooks please install 'jupyterthemes'.")

from . import inference

ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

class Lik(object):
    """Container class for the original likelihood"""
    def __init__(self,
                 lik_name = None,
                 logpdf = None,
                 pars_pos_poi = None,
                 pars_pos_nuis = None,
                 pars_init = None,
                 pars_labels = None,
                 pars_bounds = None,
                 lik_input_file=None):
        if lik_input_file is None:
            self.lik_input_file = lik_input_file
        else:
            self.lik_input_file = path.abspath(lik_input_file)
        if self.lik_input_file is None:
            if lik_name is None:
                self.lik_name = "likelihood_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self.lik_name = lik_name
            self.output_file_base_name = path.abspath("likelihood_"+lik_name)
            self.logpdf = logpdf
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_init = pars_init
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
        else:
            self.load_lik()
            self.lik_input_file = path.abspath(lik_input_file)

    def plot_lik_par(self,par,start,end,steps):
        jtplot.reset()
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        vals = np.linspace(start, end, steps)
        points = np.asarray([self.pars_init]*steps)
        points[:, par] = vals
        logpdf_vals = [self.logpdf(point) for point in points]
        plt.plot(vals, logpdf_vals)
        plt.title(r"%s" % self.lik_name.replace("_","\_"))
        plt.xlabel(r"%s" % self.pars_labels[par].replace("_", "\_"))
        plt.ylabel(r"logpdf")
        plt.show()
        plt.close()

    def compute_lik_max(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        res = inference.maximum_loglik(self.logpdf,pars_init=self.pars_init,pars_bounds=self.pars_bounds)
        self.X_lik_max = res[0]
        self.Y_lik_max = res[1]
        end = timer()
        print("Maximum likelihood computed in",end-start,"s.")

    def compute_profiled_maxima(self,par_pos,par_low=0,par_high=2,npoints=10,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px',
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
        if verbose:
            display(overall_progress)
        iterator = 0
        start = timer()
        par_vals = np.random.uniform(par_low, par_high, npoints)
        res = []
        for par in par_vals:
            res.append(inference.maximum_prof_loglik(self.logpdf,
                                                     pars_init=self.pars_init,
                                                     pars_bounds=self.pars_bounds, 
                                                     pars_fixed_pos=[par_pos], 
                                                     pars_fixed_val=[par]))
            iterator = iterator + 1
            overall_progress.value = float(iterator)/(npoints)
        self.X_prof_lik_max = np.array([x[0].tolist() for x in res])
        self.Y_prof_lik_max = np.array(res)[:,1]
        end = timer()
        print("Log-pdf values lie in the range [",np.min(self.Y_prof_lik_max),",",np.max(self.Y_prof_lik_max),"]")
        print("Parameter initialization computed in",end-start,"s.")

    def save_lik(self, out_file=None, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if out_file is None:
            out_file = self.output_file_base_name+"_"+timestamp+".pickle"
        else:
            out_file = out_file.replace(".pickle", "")+".pickle"
        pickle_out = open(out_file, 'wb')
        cloudpickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihoods saved in file', out_file,"in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def load_lik(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        in_file = self.lik_input_file
        start = timer()
        pickle_in = open(in_file, 'rb')
        in_object = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(in_file)
        self.__dict__.update(in_object.__dict__)
        end = timer()
        print('Likelihoods loaded in', str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')
