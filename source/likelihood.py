__all__ = ["Likelihood"]

from os import path, stat, sep
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

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

class Likelihood(object):
    """Container class for the original likelihood
    .. _likelihood_class:
    """
    def __init__(self,
                 name = None,
                 logpdf = None,
                 logpdf_args = None,
                 pars_pos_poi = None,
                 pars_pos_nuis = None,
                 pars_init = None,
                 pars_labels = None,
                 pars_bounds = None,
                 out_folder = None,
                 lik_input_file=None):
        if lik_input_file is None:
            self.lik_input_file = lik_input_file
        else:
            self.lik_input_file = path.abspath(lik_input_file)
        if self.lik_input_file is None:
            if name is None:
                self.name = "likelihood_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self.name = name
            self.output_file_base_name = name.rstrip("_likelihood")+"_likelihood"
            self.logpdf = logpdf
            self.logpdf_args = logpdf_args
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_init = pars_init
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
            if out_folder is None:
                out_folder = ""
            self.out_folder = path.abspath(out_folder)
            self.X_lik_max = None
            self.Y_lik_max = None
            self.X_prof_lik_max = np.array([[]])
            self.Y_prof_lik_max = np.array([[]])
            self.define_logpdf_file = ""
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
        logpdf_vals = [self.logpdf(point,*self.logpdf_args) for point in points]
        plt.plot(vals, logpdf_vals)
        plt.title(r"%s" % self.name.replace("_","\_"),fontsize=10)
        plt.xlabel(r"%s" % self.pars_labels[par].replace("_", "\_"))
        plt.ylabel(r"logpdf")
        plt.show()
        plt.close()

    def compute_lik_max(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        if self.X_lik_max is None:
            start = timer()
            res = inference.maximum_loglik(lambda x: self.logpdf(x,*self.logpdf_args),pars_init=self.pars_init,pars_bounds=self.pars_bounds)
            self.X_lik_max = res[0]
            self.Y_lik_max = res[1]
            end = timer()
            print("Maximum likelihood computed in",end-start,"s.")
        else:
            print("Maximum likelihood already stored in self.X_lik_max and self.Y_lik_max")

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
            res.append(inference.maximum_prof_loglik(lambda x: self.logpdf(x, *self.logpdf_args),
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

    def save_lik(self, overwrite=False, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.out_folder is None:
            if overwrite:
                out_file = path.abspath(self.output_file_base_name+".pickle")
            else:
                out_file = utility.check_rename_file(path.abspath(self.output_file_base_name+".pickle"))
        else:
            if overwrite:
                out_file = path.join(self.out_folder,self.output_file_base_name+".pickle")
            else:
                out_file = utility.check_rename_file(path.join(self.out_folder,self.output_file_base_name+".pickle"))
        pickle_out = open(out_file, 'wb')
        cloudpickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihood saved in file', out_file, "in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

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
        print('Likelihood loaded in', str(end-start),'.')

    def logpdf_fn(self,x):
        for i in range(len(x)):
            if not (x[i] > self.pars_bounds[i][0] and x[i] < self.pars_bounds[i][1]):
                return -np.inf
        tmp = self.logpdf(x, *self.logpdf_args)
        if np.isnan(tmp):
            tmp = -np.inf
        return tmp

    def generate_define_logpdf_file(self):
        filename = self.output_file_base_name+"_define_logpdf"+".py"
        self.define_logpdf_file = path.join(self.out_folder,filename)
        with open(self.define_logpdf_file, 'w') as out_file:
            out_file.write("import sys\n"+
                   "sys.path.append('../DNNLikelihood_dev')\n"+
                   "import source\n"+"\n"+
                   "lik = source.likelihood(name=None,\n"+
                   "\tlik_input_file="+r"'"+ r"%s" % (path.join(self.out_folder,self.output_file_base_name+".pickle").replace(sep,'/'))+r"')"+"\n"+"\n"+
                   "pars_pos_poi = lik.pars_pos_poi\n"+
                   "pars_pos_nuis = lik.pars_pos_nuis\n"+
                   "pars_init_vec = lik.X_prof_lik_max.tolist()\n"+
                   "pars_labels = lik.pars_labels\n"+
                   "nwalkers = len(lik.X_prof_lik_max)\n"+
                   "chains_name = lik.name\n"+"\n"+
                   "def logpdf(x):\n"+
                   "\treturn lik.logpdf_fn(x)\n"+"\n"+
                   "logpdf_args = None")
        print("File", self.define_logpdf_file, "correctly generated.")
