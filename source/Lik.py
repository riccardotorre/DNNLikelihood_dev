__all__ = ["Lik"]

import os
import builtins
from datetime import datetime
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

mplstyle_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],"matplotlib.mplstyle")

class Lik(object):
    """Container class for the original likelihood"""
    def __init__(self,
                 lik_name = None,
                 logpdf = None,
                 pars_pos_poi = None,
                 pars_pos_nuis = None,
                 pars_init = None,
                 pars_labels = None,
                 pars_bounds = None):
        if lik_name is None:
            self.lik_name = "likelihood_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            self.lik_name = lik_name
        self.logpdf = logpdf
        self.pars_pos_poi = pars_pos_poi
        self.pars_pos_nuis = pars_pos_nuis
        self.pars_init = pars_init
        self.pars_labels = pars_labels
        self.pars_bounds = pars_bounds

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

    def compute_lik_max(self):
        start = timer()
        res = inference.maximum_loglik(self.logpdf,pars_init=self.pars_init,pars_bounds=self.pars_bounds)
        self.X_lik_max = res[0]
        self.Y_lik_max = res[1]
        end = timer()
        print("Maximum likelihood computed in",end-start,"s.")

    def compute_profiled_maxima(self,par_pos,par_low=0,par_high=2,npoints=10):
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px',
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
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

