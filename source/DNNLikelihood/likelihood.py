__all__ = ["Likelihood"]

import builtins
import pickle
from datetime import datetime
from os import path, sep, stat
from timeit import default_timer as timer

import cloudpickle
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from . import inference, utils
from . import show_prints
from .show_prints import print

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")


class Likelihood(object):
    """
    This class contains the ``Likelihood`` object, storing all information of the original likelihood function.
    The object can be manually created or obtained from an ATLAS histfactory workspace through the 
    :class:`DNNLikelihood.Histfactory` module.
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
                 likelihood_input_file = None,
                 verbose = True):
        """
        Instantiates the ``Likelihood`` object. 
        If ``likelihood_input_file`` has the default value ``None``, the other arguments are parsed, otherwise all other arguments
        are ignored and the object is entirely reconstructed from the input file. The input file should be a .pickle file exported 
        through the ``Likelihood.save_likelihood()`` method.
        
        - **Arguments**

            See Class arguments.
        """
        show_prints.verbose = verbose
        self.likelihood_input_file = likelihood_input_file
        if self.likelihood_input_file is None:
            self.name = name
            self.check_define_name()
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
            self.output_base_filename = path.join(self.output_folder,self.name)
            self.X_logpdf_max = None
            self.Y_logpdf_max = None
            self.X_prof_logpdf_max = None
            self.Y_prof_logpdf_max = None
            self.X_prof_logpdf_max_tmp = None
            self.Y_prof_logpdf_max_tmp = None
            self.likelihood_script_file = path.join(self.output_folder, self.output_base_filename+"_script.py")
        else:
            self.likelihood_input_file = path.abspath(utils.check_add_suffix(likelihood_input_file,".pickle"))
            self.__load_likelihood(verbose=verbose)
        self.generic_pars_labels = utils.define_generic_pars_labels(self.pars_pos_poi, self.pars_pos_nuis)

    def __check_define_name(self):
        """
        If :attr:`Likelihood.name <DNNLikelihood.Likelihood.name>` is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_likelihood"``
        otherwise it appends the suffix "_likelihood" (preventing duplication if it is already present).
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.name = "model_"+timestamp+"_likelihood"
        else:
            self.name = utils.check_add_suffix(name, "_likelihood")

    #def set_verbose(self, verbose=True):
    #    show_prints.verbose = verbose
    #    print(show_prints.verbose)

    #def hello_world(self):
    #    print("hello world")

    def __load_likelihood(self, verbose=True):
        """
        Private method used by the ``__init__`` one to load the ``Likelihood`` object from the file ``Likelihood.likelihood_input_file``.
        """
        show_prints.verbose = verbose
        in_file = self.likelihood_input_file
        start = timer()
        pickle_in = open(in_file, 'rb')
        self.name = pickle.load(pickle_in)
        self.logpdf = pickle.load(pickle_in)
        self.logpdf_args = pickle.load(pickle_in)
        self.pars_pos_poi = pickle.load(pickle_in)
        self.pars_pos_nuis = pickle.load(pickle_in)
        self.pars_init = pickle.load(pickle_in)
        self.pars_labels = pickle.load(pickle_in)
        self.pars_bounds = pickle.load(pickle_in)
        self.output_folder = pickle.load(pickle_in)
        self.output_base_filename = pickle.load(pickle_in)
        self.X_logpdf_max = pickle.load(pickle_in)
        self.Y_logpdf_max = pickle.load(pickle_in)
        self.X_prof_logpdf_max = pickle.load(pickle_in)
        self.Y_prof_logpdf_max = pickle.load(pickle_in)
        self.likelihood_script_file = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(in_file)
        end = timer()
        print('Likelihood loaded in', str(end-start), '.')

    def __set_pars_labels(self, pars_labels):
        """
        Returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns :attr:`Likelihood.pars_labels <DNNLikelihood.Likelihood.pars_labels>`
                and :attr:`Likelihood.generic_pars_labels <DNNLikelihood.Likelihood.generic_pars_labels>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``
        """
        if pars_labels is "original":
            return self.pars_labels
        elif pars_labels is "generic":
            return self.generic_pars_labels
        else:
            return pars_labels

    def plot_logpdf_par(self,pars=0,npoints=100,pars_init=None,pars_labels="original",overwrite=False,verbose=True):
        """
        Method that produces a plot of the logpdf as a function of of the parameter ``par`` in the range ``(min,max)``
        using a number ``npoints`` of points. Only the parameter ``par`` is veried, while all other parameters are kept
        fixed to their value given ib ``pars_init``. 

        - **Arguments**

            - **pars**
            
                List of lists containing the position of the parametes in the parameters vector, 
                and their minimum value and maximum for the plot.
                For example, to plot parameters ``1`` in the rage ``(1,3)`` and parameter ``5`` in the range
                ``(-3,3)`` one should set ``pars = [[1,1,3],[5,-3,3]]``. 

                    - **type**: ``list``
                    - **shape**: ``[[ ]]``

            - **npoints**
            
                Number of points in which the ``(par_min,par_max)`` range
                is divided to compute the logpdf and make the plot.

                    - **type**: ``int``
                    - **default**: ``100``

            - **pars_init**
            
                Initial point in the parameter space from which ``par`` is varied and all other parameters are 
                kept fixed. When its value is the default ``None``, the attribute ``Likelihood.pars_init`` is used.

                    - **type**: ``numpy.ndarray`` or ``None``
                    - **shape**: ``(n_pars,)``
                    - **default**: ``None``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Likelihood.__set_pars_labels <DNNLikelihood.Likelihood._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plot.
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        """
        show_prints.verbose = verbose
        plt.style.use(mplstyle_path)
        pars_labels = self.__set_pars_labels(pars_labels)
        if pars_init is None:
            pars_init = self.pars_init
        for par in pars:
            par_number = par[0]
            par_min = par[1]
            par_max = par[2]
            vals = np.linspace(par_min, par_max, npoints)
            points = np.asarray([pars_init]*npoints)
            points[:, par_number] = vals
            logpdf_vals = [self.logpdf(point,*self.logpdf_args) for point in points]
            plt.plot(vals, logpdf_vals)
            plt.title(r"%s" % self.name.replace("_","\_"),fontsize=10)
            plt.xlabel(r"%s" % pars_labels[par_number].replace("_", "\_"))
            plt.ylabel(r"logpdf")
            plt.tight_layout()
            figure_filename = self.output_base_filename+"_par_"+str(par[0])+".pdf"
            if not overwrite:
                figure_filename = utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()

    def compute_maximum_logpdf(self,verbose=True):
        """
        Method that computes the maximum of logpdf. The values of the parameters and of logpdf at the 
        global maximum are stored in the attributes ``Likelihood.X_logpdf_max`` and ``Likelihood.Y_logpdf_max``, 
        respectively. The method uses the function ``inference.find_maximum``
        based on scipy.optimize.minimize to find the maximum of ``Likelihood.logpdf_fn``. Since this 
        method already contains a bounded logpdf,  ``pars_bounds`` is set to ``None`` in the 
        :func:`inference.find_maximum <DNNLikelihood.inference.find_maximum>`
        function to optimizes speed. See the doc of :func:`inference.find_maximum <DNNLikelihood.inference.find_maximum>` for more details.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        
        """
        show_prints.verbose = verbose
        if self.X_logpdf_max is None:
            start = timer()
            res = inference.find_maximum(lambda x: self.logpdf_fn(x,*self.logpdf_args), pars_init=self.pars_init, pars_bounds=None)
            self.X_logpdf_max = res[0]
            self.Y_logpdf_max = res[1]
            end = timer()
            print("Maximum likelihood computed in",end-start,"s.")
        else:
            print("Maximum likelihood already stored in self.X_logpdf_max and self.Y_logpdf_max")

    def compute_profiled_maxima(self,pars,pars_ranges,spacing="grid",append=False,verbose=True):
        """
        Computes logal maxima of the logpdf for different values of the parameter ``pars``.
        For a list of prameters ``pars`` ranges are passed as ``pars_ranges`` in the form ``(par_min,par_max,n_points)``
        and an array of points is generated according to the argument ``spacing`` (either a grid or a random 
        flat distribution) in the interval. The points in the grid falling outside ``Likelihood.pars_bounds`` are
        automatically removed.
        The values of the parameters and of logpdf at the local maxima are stored in the attributes
        ``Likelihood.X_prof_logpdf_max`` and ``Likelihood.Y_prof_logpdf_max``, respectively.
        They could be used both for frequentist maximum profiled likelihood inference or as initial condition for
        Markov Chain Monte Carlo through the :class:`DNNLikelihood.Sampler` object. 
        The method uses the function ``inference.find_prof_maximum`` based on scipy.optimize.minimize
        to find the maximum of ``Likelihood.logpdf_fn`` function. Since the latter method already contains a 
        bounded logpdf, ``pars_bounds`` is set to ``None`` in the ``inference.find_prof_maximum`` function 
        to maximize speed. See the doc of ``inference.find_prof_maximum`` for more details.

        - **Arguments**

            - **pars**
            
                List of position of the parameter under which logpdf 
                is not profiled.

                    - **type**: ``list``
                    - **shape**: ``[ ]``
                    - **example**: ``[1,5,8]``

            - **pars_ranges**
            
                Ranges of the parameters ``pars``
                containing ``(min,max,n_points)``.

                    - **type**: ``list``
                    - **shape**: ``[[ ]]``
                    - **example**: ``[[0,1,5],[-1,1,5],[0,5,3]]``

            - **spacing**
            
                It can be either ``"grid"`` or ``"random"``. Depending on its 
                value the ``n_points`` for each parameter are taken on an 
                equally spaced grid or are generated randomly in the interval.

                    - **type**: ``str``
                    - **accepted**: ``"grid"`` or ``"random"``
                    - **default**: ``grid``

            - **append**
            
                If ``append=False`` the values of ``X_prof_logpdf_max`` and ``Y_prof_logpdf_max``
                are replaced, otherwise, newly computed values are appended to the existing ones.
                If dimension ot the newly computed ones is incompatible with the existing ones,
                new values are saved in the temporary attributes ``X_prof_logpdf_max_tmp`` and 
                ``Y_prof_logpdf_max_tmp`` and a warning message is generated. Notice that the attributes
                ``X_prof_logpdf_max_tmp`` and ``Y_prof_logpdf_max_tmp`` are not saved and always get
                initialized to ``None``.

                    - **type**: ``str``
                    - **accepted**: ``"grid"`` or ``"random"``
                    - **default**: ``grid``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        
        """
        show_prints.verbose = verbose
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px',
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
        if verbose:
            display(overall_progress)
        iterator = 0
        start = timer()
        pars_vals = utils.get_sorted_grid(pars_ranges=pars_ranges, spacing=spacing)
        print("Total number of points:", len(pars_vals))
        pars_vals_bounded = []
        for i in range(len(pars_vals)):
            if (np.all(pars_vals[i] >= self.pars_bounds[pars, 0]) and np.all(pars_vals[i] <= self.pars_bounds[pars, 1])):
                pars_vals_bounded.append(pars_vals[i])
        if len(pars_vals) != len(pars_vals_bounded):
            print("Deleted", str(len(pars_vals)-len(pars_vals_bounded)),"points outside the parameters allowed range.")
        res = []
        for pars_val in pars_vals_bounded:
            res.append(inference.find_prof_maximum(lambda x: self.logpdf_fn(x, *self.logpdf_args),
                                                   pars_init=self.pars_init,
                                                   pars_bounds=None, 
                                                   pars_fixed_pos=pars, 
                                                   pars_fixed_val=pars_val))
            iterator = iterator + 1
            overall_progress.value = float(iterator)/(len(pars_vals_bounded))
        X_tmp = np.array([x[0].tolist() for x in res])
        Y_tmp = np.array(res)[:, 1]
        if self.X_prof_logpdf_max is None:
            self.X_prof_logpdf_max = X_tmp
            self.Y_prof_logpdf_max = Y_tmp
        else:
            if append:
                if np.shape(self.X_prof_logpdf_max)[1] == np.shape(X_tmp)[1]:
                    self.X_prof_logpdf_max = np.concatenate((self.X_prof_logpdf_max, X_tmp))
                    self.Y_prof_logpdf_max = np.concatenate((self.Y_prof_logpdf_max, Y_tmp))
                    print("New values have been appended to the existing ones.")
                else:
                    self.X_prof_logpdf_max_tmp = X_tmp
                    self.Y_prof_logpdf_max_tmp = Y_tmp
                    print("New values and existing ones have different shape and cannot be concatenated. New values stored in the temporary attributes 'X_prof_logpdf_max_tmp' and 'Y_prof_logpdf_max_tmp'.")
            else:
                self.X_prof_logpdf_max = X_tmp
                self.Y_prof_logpdf_max = Y_tmp
        end = timer()
        print("Log-pdf values lie in the range [",np.min(self.Y_prof_logpdf_max),",",np.max(self.Y_prof_logpdf_max),"]")
        print(len(pars_vals_bounded),"local maxima computed in", end-start, "s.")

    def save_likelihood(self, overwrite=False, verbose=True):
        """
        Saves the ``Likelihood`` object in the file ``Likelihood.output_base_filename+".pickle"``.
        In particular it does a dump of each of the attribuses ``name``, ``logpdf``, ``logpdf_args``, 
        ``pars_pos_poi``, ``pars_pos_nuis``, ``pars_init``, ``pars_labels``, ``pars_bounds``, ``output_folder``,
        ``output_base_filename``, ``X_logpdf_max``, ``Y_logpdf_max``, ``X_prof_logpdf_max``, ``Y_prof_logpdf_max``,
        ``Y_prof_logpdf_max`` in this order. All attributes are saved with ``pickle``, but ``logpdf``, which is saved
        using ``cloudpickle`` to avoid the "Can't pickle..."  error.

        - **Arguments**

            - **overwrite**
            
                Flag that determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function 
                is used to append a time-stamp to the file name.

                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        show_prints.verbose = verbose
        start = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if overwrite:
            out_file = self.output_base_filename+".pickle"
        else:
            out_file = utils.check_rename_file(self.output_base_filename+".pickle")
        pickle_out = open(out_file, 'wb')
        pickle.dump(self.name, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.logpdf, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.logpdf_args, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.pars_pos_poi, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.pars_pos_nuis, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.pars_init, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.pars_labels, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.pars_bounds, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.output_folder, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.output_base_filename, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Y_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X_prof_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Y_prof_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.likelihood_script_file, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihood saved in file', out_file, "in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def logpdf_fn(self, x_pars, *logpdf_args):
        """s
        This function is used to add constraints and standardize input/output of ``Likelihood.logpdf``.
        It is constructed from ``Likelihood.logpdf`` and ``Likelihood.logpdf_args``. 
        In the case ``Likelihood.logpdf`` accepts a single array of parameters ``x_pars`` and computes the lofpdf
        one point at a time, the function returns a ``float``, while if ``Likelihood.logpdf`` is vectorized 
        (i.e. accepts an array of ``x_pars`` arrays and returns an array of logpdf values), it returns an array. 
        The function is constructed to return lofpdf value ``-np.inf`` if any of the parameters lies outside
        ``Likelihood.pars_bounds`` or the value of ``Likelihood.logpdf`` output is ``nan``.

        - **Arguments**

            - **x_pars**

                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``) 
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``, depending on the 
                equivalent argument accepted by ``Likelihood.logpdf``.
                Values of the parameters for which 
                logpdf is computed.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(n_pars,)``

            - **args**

                List containing additional inputs needed by the logpdf function. 
                See :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>`.

                    - **type**: ``list`` or None
                    - **shape of list**: ``[]``

        - **Returns**

            Value or array of values 
            of the logpdf.
            
                - **type**: ``float`` or ``numpy.ndarray``
                - **shape for numpy.ndarray**: ``(n_points,)``
        """
        if len(np.shape(x_pars)) == 1:
            if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                return -np.inf
            if logpdf_args is None:
                tmp = self.logpdf(x_pars)
            else:
                tmp = self.logpdf(x_pars, *logpdf_args)
            if type(tmp) is np.ndarray or type(tmp) is list:
                tmp = tmp[0]
            if np.isnan(tmp):
                tmp = -np.inf
            return tmp
        else:
            x_pars_list = x_pars
            if logpdf_args is None:
                tmp = self.logpdf(x_pars_list)
            else:
                tmp = self.logpdf(x_pars_list, *logpdf_args)
            for i in range(len(x_pars_list)):
                x_pars = x_pars_list[i]
                if not (np.all(x_pars >= self.pars_bounds[:, 0]) and np.all(x_pars <= self.pars_bounds[:, 1])):
                    np.put(tmp,i,-np.inf)
            tmp = np.where(np.isnan(tmp), -np.inf, tmp)
            return tmp
            

        #else:
        #    x_pars_list = x_pars
        #    pars_out_of_range = []
        #    for i in len(x_pars_list):
        #        for j in range(len(x_pars_list[i])):
        #            if not (x_pars_list[i,j] >= self.pars_bounds[i][0] and x_pars[i] <= self.pars_bounds[i][1]):
        #            return -np.inf
        #    for x_pars in x_pars_list:
        #        for i in range(len(x_pars)):
        #            if not (x_pars[i] >= self.pars_bounds[i][0] and x_pars[i] <= self.pars_bounds[i][1]):
        #            return -np.inf
        #        if self.logpdf_args is None:
        #            tmp = self.logpdf(x_pars)
        #        else:
        #            tmp = self.logpdf(x_pars, *self.logpdf_args)
        #        if type(tmp) is np.ndarray or type(tmp) is list:
        #            tmp = tmp[0]
        #        if np.isnan(tmp):
        #            tmp = -np.inf
        #        return tmp

    def generate_likelihood_script_file(self):
        """
        Generates and saves the file ``Likelihood.likelihood_script_file`` containing the code to instantiate the
        ``Likelihood`` object sometimes needed to properly run Markov Chain Monte Carlo in parallel 
        (using ``Multiprocessing``) through the ``Sampler`` object inside Jupyter notebooks on the Windows platform.
        """
        with open(self.likelihood_script_file, 'w') as out_file:
            out_file.write("import sys\n"+
                   "sys.path.append('../DNNLikelihood_dev/source')\n"+
                   "import DNNLikelihood\n"+"\n"+
                   "lik = DNNLikelihood.Likelihood(name=None,\n"+
                   "\tlikelihood_input_file="+r"'"+ r"%s" % ((self.output_base_filename+".pickle").replace(sep,'/'))+r"')"+"\n"+"\n"+
                   "name = lik.name\n"+
                   "def logpdf(x_pars,*args):\n"+
                   "\tif args is None:\n"+
                   "\t\treturn lik.logpdf_fn(x_pars)\n"+
                   "\telse:\n"+
                   "\t\treturn lik.logpdf_fn(x_pars,*args)\n"+
                   "logpdf_args = lik.logpdf_args\n"+
                   "pars_pos_poi = lik.pars_pos_poi\n"+
                   "pars_pos_nuis = lik.pars_pos_nuis\n"+
                   "pars_init_vec = lik.X_prof_logpdf_max.tolist()\n"+
                   "pars_labels = lik.pars_labels\n"+
                   "output_folder = lik.output_folder"
                   )
        print("File", self.likelihood_script_file, "correctly generated.")
