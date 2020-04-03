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
from . import utils

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
    """
    .. _likelihood_class:
    This class contains the ``Likelihood`` object, storing all information of the original likelihood function.
    The object can be manually created or obtained from an ATLAS histfactory workspace through the ``histfactory`` 
    module (see :ref:`The Histfactory object <histfactory_class>`) 
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
                 output_folder = None,
                 likelihood_input_file=None):
        """
        Instantiate the ``Likelihood`` object. 
        If ``likelihood_input_file`` has the default value ``None``, the other arguments are parsed, otherwise all other arguments
        are ignored and the object is entirely reconstructed from the input file. The input file should be a .pickle file exported 
        through the ``Likelihood.save_likelihood()`` method.
        
        - **Arguments**

        See Class arguments.

        """
        if likelihood_input_file is None:
            self.likelihood_input_file = likelihood_input_file
        else:
            self.likelihood_input_file = path.abspath(likelihood_input_file)
        if self.likelihood_input_file is None:
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
            if output_folder is None:
                output_folder = ""
            self.output_folder = path.abspath(output_folder)
            self.X_logpdf_max = None
            self.Y_logpdf_max = None
            self.X_prof_logpdf_max = None
            self.Y_prof_logpdf_max = None
            self.define_logpdf_file = ""
        else:
            self.load_likelihood()
            self.likelihood_input_file = path.abspath(likelihood_input_file)

    def plot_logpdf_par(self,par,par_min,par_max,npoints=100,pars_init=None):
        """
        Method that produces a plot of the logpdf as a function of of the parameter ``par`` in the range ``(min,max)``
        using a number ``npoints`` of points. Only the parameter ``par`` is veried, while all other parameters are kept
        fixed to their value given ib ``pars_init``. 

        - **Arguments**

            - **par**
            
                Position of the parameter to plot 
                in the parameters list.
                    - **type**: ``int``

            - **par_min, par_max**
            
                Minimum and maximum values of the 
                parameter ``par`` that define its range.
                    - **type**: ``int`` or ``float``

            - **npoints**
            
                Number of points in which the ``(par_min,par_max)`` range
                is divided to compute the logpdf and make the plot
                    - **type**: ``int``
                    - **default**: ``100``

            - **pars_init**
            
                Initial point in the parameter space from which ``par`` is varied and all other parameters are 
                kept fixed. When its value is the default ``None``, the attribute ``Likelihood.pars_init`` is used.
                    - **type**: ``numpy.ndarray`` or ``None``
                    - **shape**: ``(n_pars,)``
                    - **default**: ``None``
        
        """
        jtplot.reset()
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        if pars_init is None:
            pars_init = self.pars_init
        vals = np.linspace(par_min, par_max, npoints)
        points = np.asarray([pars_init]*npoints)
        points[:, par] = vals
        logpdf_vals = [self.logpdf(point,*self.logpdf_args) for point in points]
        plt.plot(vals, logpdf_vals)
        plt.title(r"%s" % self.name.replace("_","\_"),fontsize=10)
        plt.xlabel(r"%s" % self.pars_labels[par].replace("_", "\_"))
        plt.ylabel(r"logpdf")
        plt.show()
        plt.close()

    def compute_maximum_logpdf(self,verbose=True):
        """
        Method that computes the maximum of logpdf. 
        The values of the parameters and of logpdf at the global maximum are stored in the attributes
        ``Likelihood.X_logpdf_max`` and ``Likelihood.Y_logpdf_max``, respectively.
        The method uses the function ``inference.find_maximum``
        based on scipy.optimize.minimize. See the doc of ``inference.find_maximum`` for more details.

        - **Arguments**

            - **verbose**
            
                Verbose mode. 
                See :ref:`_verbose_implementation`.
                    - **type**: ``bool``
                    - **default**: ``True`` 
        
        """
        global ShowPrints
        ShowPrints = verbose
        if self.X_logpdf_max is None:
            start = timer()
            res = inference.find_maximum(lambda x: self.logpdf(x,*self.logpdf_args),pars_init=self.pars_init,pars_bounds=self.pars_bounds)
            self.X_logpdf_max = res[0]
            self.Y_logpdf_max = res[1]
            end = timer()
            print("Maximum likelihood computed in",end-start,"s.")
        else:
            print("Maximum likelihood already stored in self.X_logpdf_max and self.Y_logpdf_max")

    def compute_profiled_maxima(self,par,par_min=0,par_max=2,npoints=10,verbose=True):
        """
        Method that computes logal maxima of the logpdf for different values of the parameter ``par``.
        A number ``npoints`` of different values of ``par`` are set randomly (with a flat distribution) 
        in the interval ``(par_min,par_max)``.
        The values of the parameters and of logpdf at the local maxima are stored in the attributes
        ``Likelihood.X_prof_logpdf_max`` and ``Likelihood.Y_prof_logpdf_max``, respectively.
        The method uses the function ``inference.find_maximum``
        based on scipy.optimize.minimize. See the doc of ``inference.find_maximum`` for more details.

        - **Arguments**

            - **par**
            
                Position of the parameter under which logpdf 
                is not profiled.
                    - **type**: ``int``

            - **par_min, par_max**
            
                Minimum and maximum values of the 
                parameter ``par``.
                    - **type**: ``int`` or ``float``
                    - **defaule**: ``0``, ``2``

            - **npoints**
            
                Number of points in which the profiled maxima
                are computed
                    - **type**: ``int``
                    - **default**: ``100``

            - **verbose**
            
                Verbose mode. 
                See :ref:`_verbose_implementation`.
                    - **type**: ``bool``
                    - **default**: ``True`` 
        
        """
        global ShowPrints
        ShowPrints = verbose
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px',
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
        if verbose:
            display(overall_progress)
        iterator = 0
        start = timer()
        par_vals = np.random.uniform(par_min, par_max, npoints)
        res = []
        for par in par_vals:
            res.append(inference.find_prof_maximum(lambda x: self.logpdf(x, *self.logpdf_args),
                                                     pars_init=self.pars_init,
                                                     pars_bounds=self.pars_bounds, 
                                                     pars_fixed_pos=[par], 
                                                     pars_fixed_val=[par]))
            iterator = iterator + 1
            overall_progress.value = float(iterator)/(npoints)
        self.X_prof_logpdf_max = np.array([x[0].tolist() for x in res])
        self.Y_prof_logpdf_max = np.array(res)[:,1]
        end = timer()
        print("Log-pdf values lie in the range [",np.min(self.Y_prof_logpdf_max),",",np.max(self.Y_prof_logpdf_max),"]")
        print("Parameter initialization computed in",end-start,"s.")

    def save_likelihood(self, overwrite=False, verbose=True):
        """
        Saves the ``Likelihood`` object in the file ``path.join(Histfactory.output_folder,self.output_file_base_name+".pickle")`` using pickle.
        In particular it does a picle.dump of the full object.

        - **Arguments**

            - **overwrite**
            
                Flag that determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the ``utils.check_rename_file`` function (see :ref:`_utils_check_rename_file`) is used  
                to append a time-stamp to the file name.
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbose mode. 
                See :ref:`_verbose_implementation`.
                    - **type**: ``bool``
                    - **default**: ``True``
        """
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if overwrite:
            out_file = path.join(self.output_folder,self.output_file_base_name+".pickle")
        else:
            out_file = utils.check_rename_file(path.join(self.output_folder,self.output_file_base_name+".pickle"))
        pickle_out = open(out_file, 'wb')
        cloudpickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(self, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihood saved in file', out_file, "in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def load_likelihood(self, verbose=True):
        """
        Loads the ``Likelihood`` object from the file ``Likelihood.likelihood_input_file``. The object is assigned to 
        a temporary variable which is then used to update the ``Likelihood.__dict__`` attribute.
        In particular it does a picle.dump of the full object.

        - **Arguments**

            - **overwrite**
            
                Flag that determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the ``utils.check_rename_file`` function (see :ref:`_utils_check_rename_file`) is used  
                to append a time-stamp to the file name.
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbose mode. 
                See :ref:`_verbose_implementation`.
                    - **type**: ``bool``
                    - **default**: ``True``
        """
        global ShowPrints
        ShowPrints = verbose
        in_file = self.likelihood_input_file
        start = timer()
        pickle_in = open(in_file, 'rb')
        in_object = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(in_file)
        self.__dict__.update(in_object.__dict__)
        end = timer()
        print('Likelihood loaded in', str(end-start),'.')

    def logpdf_fn(self,x_pars):
        """
        .. _likelihood.logpdf_fn:
        Callable method returning the logpdf function given parameters values. It is constructed from the
        attributed ``Likelihood.logpdf`` and ``Likelihood.logpdf_args`` so that it gives the logpdf value if
        parameters values are within ``Likelihood.pars_bounds`` and ``-np.inf`` otherwise. It also checks
        if the value of logpdf gives ``Nan`` and in such case it returns ``-np.inf``.

        - **Arguments**

            - **x_pars**
                Values of the parameters for which 
                logpdf is computed.
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(n_pars,)``

         - **Returns**

            The value of 
            logpdf.
                - **type**: ``float`` or ``-np.inf``

        """
        for i in range(len(x_pars)):
            if not (x_pars[i] > self.pars_bounds[i][0] and x_pars[i] < self.pars_bounds[i][1]):
                return -np.inf
        if self.logpdf_args is None:
            tmp = self.logpdf(x_pars)
        else:
            tmp = self.logpdf(x_pars, *self.logpdf_args)
        if type(tmp) is np.ndarray or type(tmp) is list:
            tmp = tmp[0]
        if np.isnan(tmp):
            tmp = -np.inf
        return tmp

    def generate_define_logpdf_file(self):
        """
        .. _likelihood_generate_define_logpdf_file:
        Generates and saves the file ``Likelihood.define_logpdf_file`` containing the code to instantiate the
        ``Likelihood`` object sometimes needed to properly run Markov Chain Monte Carlo in parallel 
        (using ``Multiprocessing``) through the ``Sampler`` object inside Jupyter notebooks on the Windows platform.
        """
        filename = self.output_file_base_name+"_define_logpdf"+".py"
        self.define_logpdf_file = path.join(self.output_folder,filename)
        with open(self.define_logpdf_file, 'w') as out_file:
            out_file.write("import sys\n"+
                   "sys.path.append('../DNNLikelihood_dev')\n"+
                   "import DNNLikelihood\n"+"\n"+
                   "lik = DNNLikelihood.Likelihood(name=None,\n"+
                   "\tlikelihood_input_file="+r"'"+ r"%s" % (path.join(self.output_folder,self.output_file_base_name+".pickle").replace(sep,'/'))+r"')"+"\n"+"\n"+
                   "pars_pos_poi = lik.pars_pos_poi\n"+
                   "pars_pos_nuis = lik.pars_pos_nuis\n"+
                   "pars_init_vec = lik.X_prof_logpdf_max.tolist()\n"+
                   "pars_labels = lik.pars_labels\n"+
                   "nwalkers = len(lik.X_prof_logpdf_max)\n"+
                   "chains_name = lik.name\n"+"\n"+
                   "def logpdf(x_pars):\n"+
                   "\treturn lik.logpdf_fn(x_pars)\n"+"\n"+
                   "logpdf_args = None")
        print("File", self.define_logpdf_file, "correctly generated.")
