__all__ = ["Likelihood"]

import builtins
import codecs
import json
import pickle
import time
from datetime import datetime
from os import path, sep, stat
from timeit import default_timer as timer

import cloudpickle
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from . import inference, show_prints, utils
from .show_prints import print

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")


class Likelihood(show_prints.Verbosity):
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
        self.verbose = verbose
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.likelihood_input_file = likelihood_input_file
        if self.likelihood_input_file is None:
            self.likelihood_input_json_file = self.likelihood_input_file
            self.likelihood_input_log_file = self.likelihood_input_file
            self.likelihood_input_pickle_file = self.likelihood_input_file
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.logpdf = logpdf
            self.logpdf_args = logpdf_args
            self.pars_pos_poi = np.array(pars_pos_poi)
            self.pars_pos_nuis = np.array(pars_pos_nuis)
            self.pars_init = np.array(pars_init)
            self.pars_labels = pars_labels
            self.generic_pars_labels = utils.define_generic_pars_labels(self.pars_pos_poi, self.pars_pos_nuis)
            self.pars_bounds = np.array(pars_bounds)
            if output_folder is None:
                output_folder = ""
            self.output_folder = path.abspath(output_folder)
            self.output_files_base_path = path.join(self.output_folder,self.name)
            self.likelihood_output_json_file = self.output_files_base_path+".json"
            self.likelihood_output_log_file = self.output_files_base_path+".log"
            self.likelihood_output_pickle_file = self.output_files_base_path+".pickle"
            self.likelihood_script_file = self.output_files_base_path+"_script.py"
            self.figure_files_base_path = self.output_files_base_path+"_figure"
            self.X_logpdf_max = None
            self.Y_logpdf_max = None
            self.X_prof_logpdf_max = None
            self.Y_prof_logpdf_max = None
            self.X_prof_logpdf_max_tmp = None
            self.Y_prof_logpdf_max_tmp = None
            self.figures_list = []
            self.save_likelihood_json(overwrite=False, verbose=verbose)
        else:
            self.likelihood_input_file = path.abspath(path.splitext(likelihood_input_file)[0])
            self.likelihood_input_json_file = self.likelihood_input_file+".json"
            self.likelihood_input_log_file = self.likelihood_input_file+".log"
            self.likelihood_input_pickle_file = self.likelihood_input_file+".pickle"
            self.__load_likelihood(verbose=verbose)
            if output_folder is not None:
                self.output_folder = path.abspath(output_folder)
                self.output_files_base_path = path.join(self.output_folder,self.name)
                self.likelihood_output_json_file = self.output_files_base_path+".json"
                self.likelihood_output_log_file = self.output_files_base_path+".log"
                self.likelihood_output_pickle_file = self.output_files_base_path+".pickle"
                self.likelihood_script_file = self.output_files_base_path+"_script.py"
                self.figure_files_base_path = self.output_files_base_path+"_figure"
            self.verbose = verbose

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
            self.name = utils.check_add_suffix(self.name, "_likelihood")

    #def set_verbose(self, verbose=True):
    #    show_prints.verbose = verbose
    #    print(show_prints.verbose)

    #def hello_world(self):s
    #    print("hello world")

    def __load_likelihood(self, verbose=None):
        """
        Private method used by the ``__init__`` one to load the ``Likelihood`` object from the files 
        :attr:`Histfactory.likelihood_input_json_file <DNNLikelihood.Histfactory.likelihood_input_json_file>`
        and :attr:`Histfactory.likelihood_input_pickle_file <DNNLikelihood.Histfactory.likelihood_input_pickle_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        self.set_verbosity(verbose)
        start = timer()
        with open(self.likelihood_input_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.likelihood_input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        self.pars_pos_poi = np.array(self.pars_pos_poi)
        self.pars_pos_nuis = np.array(self.pars_pos_nuis)
        self.pars_init = np.array(self.pars_init)
        self.pars_bounds = np.array(self.pars_bounds)
        pickle_in = open(self.likelihood_input_pickle_file, "rb")
        self.logpdf = pickle.load(pickle_in)
        self.X_logpdf_max = pickle.load(pickle_in)
        self.Y_logpdf_max = pickle.load(pickle_in)
        self.X_prof_logpdf_max = pickle.load(pickle_in)
        self.Y_prof_logpdf_max = pickle.load(pickle_in)
        pickle_in.close()
        self.X_prof_logpdf_max_tmp = None
        self.Y_prof_logpdf_max_tmp = None
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {
            "action": "loaded", "file name": path.split(self.likelihood_input_json_file)[-1], "file path": self.likelihood_input_json_file}
        print('Likelihood loaded in', str(end-start), '.')
        self.save_likelihood_log(overwrite=True, verbose=False)

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

    def plot_logpdf_par(self,pars=0,npoints=100,pars_init=None,pars_labels="original",overwrite=False,verbose=None):
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
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, _ = self.set_verbosity(verbose)
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
            figure_filename = self.figure_files_base_path+"_par_"+str(par[0])+".pdf"
            utils.append_without_duplicate(self.figures_list,figure_filename)
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {
                "action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            self.save_likelihood_log(overwrite=True, verbose=False)

    def compute_maximum_logpdf(self,verbose=None):
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
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        
        """
        self.set_verbosity(verbose)
        if self.X_logpdf_max is None:
            start = timer()
            res = inference.find_maximum(lambda x: self.logpdf_fn(x,*self.logpdf_args), pars_init=self.pars_init, pars_bounds=None)
            self.X_logpdf_max = res[0]
            self.Y_logpdf_max = res[1]
            end = timer()
            print("Maximum likelihood computed in",end-start,"s.")
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "computed maximum logpdf"}
            self.save_likelihood_log(overwrite=True, verbose=False)
        else:
            print("Maximum likelihood already stored in self.X_logpdf_max and self.Y_logpdf_max")

    def compute_profiled_maxima(self,pars,pars_ranges,spacing="grid",append=False,verbose=None):
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

        When using interactive python in Jupyter notebooks if ``verbose=2`` the import process shows a progress bar through 
        the widgets module.

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
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        
        """
        verbose, _ = self.set_verbosity(verbose)
        if verbose == 2:
            progressbar = True
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                show_prints.verbose = True
                print("If you want to show a progress bar please install the ipywidgets package.")
                self.set_verbosity(verbose)
        else:
            progressbar = False
        start = timer()
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                "width": "500px", "height": "14px",
                "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
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
            if progressbar:
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "computed profiled maxima", "number of maxima": len(X_tmp)}
        self.save_likelihood_log(overwrite=True, verbose=False)

    def save_likelihood_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Likelihood.log <DNNLikelihood.Likelihood.log>` attribute in the file
        :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`

        This method is called with ``overwrite=False`` and ``verbose=False`` when the object is created from input arguments
        and with ``overwrite=True`` and ``verbose=False`` each time the 
        :attr:`Likelihood.log <DNNLikelihood.Likelihood.log>` attribute is updated.

        - **Arguments**

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        self.set_verbosity(verbose)
        time.sleep(1)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_log_file)
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #self.log[timestamp] = {"action": "saved", "file name": path.split(self.likelihood_output_log_file)[-1], "file path": self.likelihood_output_log_file}
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.likelihood_output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Likelihood log file", self.likelihood_output_log_file,"saved in", str(end-start), "s.")

    def save_likelihood_json(self, overwrite=False, verbose=None):
        """
        ``Likelihood`` objects are saved to two files: a .json and a .pickle, corresponding to the two attributes
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`
         and :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Likelihood.likelihood_input_pickle_file>`.
        This method saves the .json file containing all class attributes but the
        :attr:`Likelihood.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`,
        :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Likelihood.likelihood_input_file>`,
        :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Likelihood.likelihood_input_pickle_file>`, and
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>` attributes.

        This method is called with ``overwrite=False`` and ``verbose=False`` when the object is created from input arguments
        and with ``overwrite=True`` and ``verbose=False`` each time an attribute different from ``"log"``, ``"logpdf"``,
        ``"likelihood_input_file"``,``likelihood_input_json_file"``,``"likelihood_input_pickle_file"`` is updated.

        - **Arguments**

            - **overwrite**

                It determines whether an existing file gets overwritten or if a new file is created.
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used
                to append a time-stamp to the file name.

                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**

                Verbosity mode.
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`Likelihood.likelihood_output_json_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`

        - **Updates file**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_json_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {
            "action": "saved", "file name": path.split(self.likelihood_output_json_file)[-1], "file path": self.likelihood_output_json_file}
        dictionary = utils.dic_minus_keys(self.__dict__, ["log", "logpdf", 
                                                          "likelihood_input_file", "likelihood_input_json_file",
                                                          "likelihood_input_log_file","likelihood_input_pickle_file",
                                                          "X_logpdf_max", "Y_logpdf_max"
                                                          "X_prof_logpdf_max", "Y_prof_logpdf_max"
                                                          "X_prof_logpdf_max_tmp", "Y_prof_logpdf_max_tmp"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.likelihood_output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Likelihood json file", self.likelihood_output_json_file,"saved in", str(end-start), "s.")
        self.save_likelihood_log(overwrite=overwrite, verbose=verbose)

    def save_likelihood_pickle(self, overwrite=False, verbose=None):
        """
        ``Likelihood`` objects are saved to two files: a .json and a .pickle, corresponding to the two attributes
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`
         and :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Likelihood.likelihood_input_pickle_file>`.
        This method saves the .pickle file containing a dump of the 
        :attr:``Likelihood.likelihood_dict <DNNLikelihood.Likelihood.likelihood_dict>`` attribute.
        
        In order to save space, in the likelihood_dict the members corresponding to the keys 
        ``lik_numbers_list`` are saved in the ``model_loaded=True`` mode (so with full model included), 
        while the others are saved in ``model_loaded=False`` mode.

        - **Arguments**

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_pickle_file>`

        - **Updates file**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        verbose, _ =self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_pickle_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {
            "action": "saved", "file name": path.split(self.likelihood_output_pickle_file)[-1], "file path": self.likelihood_output_pickle_file}
        pickle_out = open(self.likelihood_output_pickle_file, "wb")
        cloudpickle.dump(self.logpdf, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Y_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.X_prof_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Y_prof_logpdf_max, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        end = timer()
        print("Likelihood pickle file", self.likelihood_output_pickle_file, "saved in", str(end-start), "s.")
        self.save_likelihood_log(overwrite=overwrite, verbose=verbose)

    def save_likelihood(self, overwrite=False, verbose=True):
        """
        Calls in order the :meth:`Likelihood.save_likelihood_pickle <DNNLikelihood.Likelihood.save_likelihood_pickle>` and
        :meth:`Likelihood.save_likelihood_json <DNNLikelihood.Likelihood.save_likelihood_json>` methods.
        Notice that each of them also calls the 
        :meth:`Likelihood.save_likelihood_json <DNNLikelihood.Likelihood.save_likelihood_json>` method, which update the log
        file :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_pickle_file>`
            - :attr:`Likelihood.likelihood_output_json_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`

        - **Updates file**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        verbose, _ =self.set_verbosity(verbose)
        self.save_likelihood_pickle(overwrite=overwrite, verbose=verbose)
        self.save_likelihood_json(overwrite=overwrite, verbose=verbose)

    def save_likelihood_script(self, verbose=True):
        """
        Saves the file :attr:`Likelihood.likelihood_script_file <DNNLikelihood.Likelihood.likelihood_script_file>` 
        containing the code to instantiate the ``Likelihood`` object. This is used to initialize the 
        :class:`Sampler <DNNLikelihood.Sampler>` object. This procedure makes ensures that Markov Chain Monte Carlo properly 
        runs in parallel (using ``Multiprocessing``) inside Jupyter notebooks also on the Windows platform.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file*

            - :attr:`Likelihood.likelihood_script_file <DNNLikelihood.Likelihood.likelihood_script_file>`
        """
        show_prints.verbose = verbose
        with open(self.likelihood_script_file, 'w') as out_file:
            out_file.write("import sys\n" +
                           "sys.path.append('../DNNLikelihood_dev/source')\n" +
                           "import DNNLikelihood\n"+"\n" +
                           "lik = DNNLikelihood.Likelihood(name=None,\n" +
                           "\tlikelihood_input_file="+r"'" + r"%s" % ((self.likelihood_output_json_file).replace(sep, '/'))+"', \n"+
                           "verbose = "+str(self.verbose)+")"+"\n"+"\n" +
                           "name = lik.name\n" +
                           "def logpdf(x_pars,*args):\n" +
                           "\tif args is None:\n" +
                           "\t\treturn lik.logpdf_fn(x_pars)\n" +
                           "\telse:\n" +
                           "\t\treturn lik.logpdf_fn(x_pars,*args)\n" +
                           "logpdf_args = lik.logpdf_args\n" +
                           "pars_pos_poi = lik.pars_pos_poi\n" +
                           "pars_pos_nuis = lik.pars_pos_nuis\n" +
                           "pars_init_vec = lik.X_prof_logpdf_max.tolist()\n" +
                           "pars_labels = lik.pars_labels\n" +
                           "output_folder = lik.output_folder"
                           )
        print("File", self.likelihood_script_file, "correctly generated.")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {
            "action": "saved", "file name": path.split(self.likelihood_script_file)[-1], "file path": self.likelihood_script_file}
        self.save_likelihood_log(overwrite=True, verbose=False)

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
