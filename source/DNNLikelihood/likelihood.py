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
    This class is a container for the ``Likelihood`` object, storing all information of the likelihood function.
    The object can be directly created or obtained from an ATLAS histfactory workspace through the 
    :class:`DNNLikelihood.Histfactory` object (see :ref:`the Histfactory object <histfactory_object>`).
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
        The :class:`Likelihood <DNNLikelihood.Likelihood>` object can be initialized in two different ways, depending on the value of 
        the :option:`likelihood_input_file` argument.

        - :option:`likelihood_input_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no name is available, then one is created. 
            This method also saves the object through the
            :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` method. 
        
        - :option:`likelihood_input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Likelihood.__load_likelihood <DNNLikelihood.Likelihood._Likelihood__load_likelihood>`
            If the input argument :option:`output_folder` is ``None`` (default), the attribute 
            :attr:`Likelihood.output_folder <DNNLikelihood.Likelihood.output_folder>`
            is set from the input file, otherwise it is set to the input argument.
        
        - **Arguments**

            See class :ref:`Arguments documentation <likelihood_arguments>`.

        - **Produces file**

            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`
            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_pickle_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
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
            if self.pars_labels is None:
                self.pars_labels = self.generic_pars_labels
            if pars_bounds is not None:
                self.pars_bounds = np.array(pars_bounds)
            else:
                self.pars_bounds = np.vstack([np.full(len(self.pars_init),-np.inf),np.full(len(self.pars_init),np.inf)]).T
            if output_folder is None:
                output_folder = ""
            self.output_folder = utils.check_create_folder(path.abspath(output_folder))
            self.likelihood_output_json_file = path.join(self.output_folder, self.name+".json")
            self.likelihood_output_log_file = path.join(self.output_folder, self.name+".log")
            self.likelihood_output_pickle_file = path.join(self.output_folder, self.name+".pickle")
            self.likelihood_script_file = path.join(self.output_folder, self.name+"_script.py")
            self.figure_files_base_path = path.join(self.output_folder, self.name+"_figure")
            self.X_logpdf_max = None
            self.Y_logpdf_max = None
            self.X_prof_logpdf_max = None
            self.Y_prof_logpdf_max = None
            self.X_prof_logpdf_max_tmp = None
            self.Y_prof_logpdf_max_tmp = None
            self.figures_list = []
            self.save_likelihood(overwrite=False, verbose=verbose_sub)
        else:
            self.likelihood_input_file = path.abspath(path.splitext(likelihood_input_file)[0])
            self.likelihood_input_json_file = self.likelihood_input_file+".json"
            self.likelihood_input_log_file = self.likelihood_input_file+".log"
            self.likelihood_input_pickle_file = self.likelihood_input_file+".pickle"
            self.__load_likelihood(verbose=verbose_sub)
            if output_folder is not None:
                self.output_folder = path.abspath(output_folder)
                self.likelihood_output_json_file = path.join(self.output_folder, self.name+".json")
                self.likelihood_output_log_file = path.join(self.output_folder, self.name+".log")
                self.likelihood_output_pickle_file = path.join(self.output_folder, self.name+".pickle")
                self.likelihood_script_file = path.join(self.output_folder, self.name+"_script.py")
                self.figure_files_base_path = path.join(self.output_folder, self.name+"_figure")

    def __check_define_name(self):
        """
        If :attr:`Likelihood.name <DNNLikelihood.Likelihood.name>` is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_likelihood"``,
        otherwise it appends the suffix "_likelihood" (preventing duplication if it is already present).
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
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
        Private method used by the :meth:`Likelihood.__init__ <DNNLikelihood.Likelihood.__init__>` one to import a previously saved
        :class:`Likelihood <DNNLikelihood.Likelihood>` object from the files 
        :attr:`Likelihood.histfactory_input_json_file <DNNLikelihood.Likelihood.histfactory_input_json_file>`,
        :attr:`Likelihood.histfactory_input_json_file <DNNLikelihood.Likelihood.histfactory_input_log_file>`
        and :attr:`Likelihood.histfactory_input_pickle_file <DNNLikelihood.Likelihood.histfactory_input_pickle_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {
            "action": "loaded", "file name": path.split(self.likelihood_input_json_file)[-1], "file path": self.likelihood_input_json_file}
        print('Likelihood loaded in', str(end-start), '.',show=verbose)
        self.save_likelihood_log(overwrite=True, verbose=verbose_sub)

    def __set_pars_labels(self, pars_labels):
        """
        Returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns the value of :attr:`Likelihood.pars_labels <DNNLikelihood.Likelihood.pars_labels>`
                or :attr:`Likelihood.generic_pars_labels <DNNLikelihood.Likelihood.generic_pars_labels>`, respectively,
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

    def save_likelihood_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Likelihood.log <DNNLikelihood.Likelihood.log>` attribute in the file
        :attr:`Likelihood.likelihood_input_log_file <DNNLikelihood.Likelihood.likelihood_input_log_file>`

        This method is called by the methods
        
        - :meth:`Likelihood.__load_likelihood <DNNLikelihood.Likelihood._Likelihood__load_likelihood>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Likelihood.compute_maximum_logpdf <DNNLikelihood.Likelihood.compute_maximum_logpdf>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Likelihood.compute_profiled_maxima <DNNLikelihood.Likelihood.compute_profiled_maxima>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Likelihood.plot_logpdf_par <DNNLikelihood.Likelihood.plot_logpdf_par>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` with ``overwrite=overwrite`` and ``verbose=verbose``
        - :meth:`Likelihood.save_likelihood_script <DNNLikelihood.Likelihood.save_likelihood_script>` with ``overwrite=True`` and ``verbose=verbose_sub``

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``


            - **verbose**
            
                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        verbose, _ = self.set_verbosity(verbose)
        time.sleep(1)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_log_file)
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #self.log[timestamp] = {"action": "saved", "file name": path.split(self.likelihood_output_log_file)[-1], "file path": self.likelihood_output_log_file}
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.likelihood_output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Likelihood log file", self.likelihood_output_log_file,"saved in", str(end-start), "s.",show=verbose)

    def save_likelihood_json(self, overwrite=False, verbose=None):
        """
        ``Likelihood`` objects are saved to three files: a .json, a .log, and a .pickle, corresponding to the three attributes
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`,
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`,
        and :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Likelihood.likelihood_input_pickle_file>`.
        This method saves the .json file containing all class attributes but the attributes

            - :attr:`Likelihood.log <DNNLikelihood.Histfactory.log>`
            - :attr:`Likelihood.logpdf <DNNLikelihood.Histfactory.logpdf>`
            - :attr:`Likelihood.verbose <DNNLikelihood.Histfactory.verbose>`
            - :attr:`Likelihood.likelihood_input_file <DNNLikelihood.Histfactory.likelihood_input_file>`
            - :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Histfactory.likelihood_input_json_file>`
            - :attr:`Likelihood.likelihood_input_log_file <DNNLikelihood.Histfactory.likelihood_input_log_file>`
            - :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Histfactory.likelihood_input_pickle_file>`
            - :attr:`Likelihood.X_logpdf_max <DNNLikelihood.Histfactory.X_logpdf_max>`
            - :attr:`Likelihood.Y_logpdf_max <DNNLikelihood.Histfactory.Y_logpdf_max>`
            - :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Histfactory.X_prof_logpdf_max>`
            - :attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Histfactory.Y_prof_logpdf_max>`
            - :attr:`Likelihood.X_prof_logpdf_max_tmp <DNNLikelihood.Histfactory.X_prof_logpdf_max_tmp>`
            - :attr:`Likelihood.Y_prof_logpdf_max_tmp <DNNLikelihood.Histfactory.Y_prof_logpdf_max_tmp>`

        This method is called by the
        :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` method to save the entire object.

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
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_json_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {
            "action": "saved", "file name": path.split(self.likelihood_output_json_file)[-1], "file path": self.likelihood_output_json_file}
        dictionary = utils.dic_minus_keys(self.__dict__, ["log", "logpdf", "verbose",
                                                          "likelihood_input_file", "likelihood_input_json_file",
                                                          "likelihood_input_log_file","likelihood_input_pickle_file",
                                                          "X_logpdf_max", "Y_logpdf_max"
                                                          "X_prof_logpdf_max", "Y_prof_logpdf_max"
                                                          "X_prof_logpdf_max_tmp", "Y_prof_logpdf_max_tmp"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.likelihood_output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Likelihood json file", self.likelihood_output_json_file,"saved in", str(end-start), "s.",show=verbose)

    def save_likelihood_pickle(self, overwrite=False, verbose=None):
        """
        ``Likelihood`` objects are saved to three files: a .json, a .log, and a .pickle, corresponding to the three attributes
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`,
        :attr:`Likelihood.likelihood_input_json_file <DNNLikelihood.Likelihood.likelihood_input_json_file>`,
        and :attr:`Likelihood.likelihood_input_pickle_file <DNNLikelihood.Likelihood.likelihood_input_pickle_file>`.
        This method saves the .pickle file making a dump of the following attributes (in order)

            - :attr:`Likelihood.logpdf <DNNLikelihood.Histfactory.logpdf>`
            - :attr:`Likelihood.X_logpdf_max <DNNLikelihood.Histfactory.X_logpdf_max>`
            - :attr:`Likelihood.Y_logpdf_max <DNNLikelihood.Histfactory.Y_logpdf_max>`
            - :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Histfactory.X_prof_logpdf_max>`
            - :attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Histfactory.Y_prof_logpdf_max>`

        The :attr:`Likelihood.logpdf <DNNLikelihood.Histfactory.logpdf>` attribute is not directly pickable, for this reason
        it is dump using the |cloudpickle_link| package, which offers mote flexible python objects serialization.

        This method is called by the
        :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>` method to save the entire object.

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

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_pickles_file>`

.. |cloudpickle_link| raw:: html
    
    <a href="https://pypi.org/project/cloudpickle/1.3.0/"  target="_blank"> cloudpickle</a>
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.likelihood_output_pickle_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
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
        print("Likelihood pickle file", self.likelihood_output_pickle_file, "saved in", str(end-start), "s.",show=verbose)

    def save_likelihood(self, overwrite=False, verbose=True):
        """
        This methos calls in order the :meth:`Likelihood.save_likelihood_pickle <DNNLikelihood.Likelihood.save_likelihood_pickle>`,
        :meth:`Likelihood.save_likelihood_json <DNNLikelihood.Likelihood.save_likelihood_json>`, and
        :meth:`Likelihood.save_likelihood_log <DNNLikelihood.Likelihood.save_likelihood_log>` methods to save the entire object.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_pickle_file>`
            - :attr:`Likelihood.likelihood_output_json_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`
            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_likelihood_pickle(overwrite=overwrite, verbose=verbose)
        self.save_likelihood_json(overwrite=overwrite, verbose=verbose)
        self.save_likelihood_log(overwrite=overwrite, verbose=verbose)

    def save_likelihood_script(self, verbose=True):
        """
        Saves the file :attr:`Likelihood.likelihood_script_file <DNNLikelihood.Likelihood.likelihood_script_file>`. 

        - **Arguments**

            - **verbose**

                Verbosity mode.
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`Likelihood.likelihood_script_file <DNNLikelihood.Likelihood.likelihood_script_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
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
        print("File", self.likelihood_script_file, "correctly generated.",show=verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved", "file name": path.split(self.likelihood_script_file)[-1], "file path": self.likelihood_script_file}
        self.save_likelihood_log(overwrite=True, verbose=verbose_sub)

    def logpdf_fn(self, x_pars, *logpdf_args):
        """
        This function is used to add constraints and standardize input/output of 
        :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>`.
        It is constructed from the :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` and 
        :attr:`Likelihood.logpdf_args <DNNLikelihood.Likelihood.logpdf_args>` attributes. 
        In the case :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` accepts a single array of 
        parameters ``x_pars`` and returns the logpdf value one point at a time, then the function returns a ``float``, 
        while if :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` is vectorized,
        i.e. accepts an array of ``x_pars`` arrays and returns an array of logpdf values, then the function returns an array. 
        Moreover, the :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>` method is constructed to return the
        logpdf value ``-np.inf`` if any of the parameters lies outside 
        :attr:`Likelihood.pars_bounds <DNNLikelihood.Likelihood.pars_bounds>` or the 
        :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` function returns ``nan``.

        - **Arguments**

            - **x_pars**

                Value (values) of the parameters for which logpdf is computed.
                It could be a single point in parameter space corresponding to an array with shape ``(n_pars,)``) 
                or a list of points corresponding to an array with shape ``(n_points,n_pars)``, depending on the 
                equivalent argument accepted by ``Likelihood.logpdf``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(n_pars,)`` or ``(n_points,n_pars)``

            - **args**

                List of additional inputs needed by the :attr:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` function. 

                    - **type**: ``list`` or None
                    - **shape of list**: ``[]``

        - **Returns**

            Value (values)
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

    def compute_maximum_logpdf(self,verbose=None):
        """
        Computes the maximum of :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>`. 
        The values of the parameters and of logpdf at the 
        global maximum are stored in the attributes 
        :attr:`Likelihood.X_logpdf_max <DNNLikelihood.Likelihood.X_logpdf_max>` and 
        :attr:`Likelihood.Y_logpdf_max <DNNLikelihood.Likelihood.Y_logpdf_max>`, respectively. 
        The method uses the function :func:`inference.find_maximum <DNNLikelihood.inference.find_maximum>`
        based on |scipy_optimize_minimize_link| to find the minimum of minus 
        :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>`. Since the latter
        method already contains a bounded logpdf, ``pars_bounds`` is set to ``None`` in the 
        :func:`inference.find_maximum <DNNLikelihood.inference.find_maximum>`
        function to optimize speed. See the documentation of the
        :func:`inference.find_maximum <DNNLikelihood.inference.find_maximum>` 
        function for details.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

.. |scipy_optimize_minimize_link| raw:: html
    
    <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html"  target="_blank"> scipy.optimize.minimize</a>
        
        
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.X_logpdf_max is None:
            start = timer()
            res = inference.find_maximum(lambda x: self.logpdf_fn(x,*self.logpdf_args), pars_init=self.pars_init, pars_bounds=None)
            self.X_logpdf_max = res[0]
            self.Y_logpdf_max = res[1]
            end = timer()
            print("Maximum likelihood computed in",end-start,"s.")
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.log[timestamp] = {"action": "computed maximum logpdf"}
            self.save_likelihood_log(overwrite=True, verbose=verbose_sub)
        else:
            print("Maximum likelihood already stored in self.X_logpdf_max and self.Y_logpdf_max",show=verbose)

    def compute_profiled_maxima(self,pars,pars_ranges,spacing="grid",append=False,verbose=None):
        """
        Computes logal maxima of the logpdf for different values of the parameter ``pars``.
        For the list of prameters ``pars``, ranges are passed as ``pars_ranges`` in the form ``(par_min,par_max,n_points)``
        and an array of points is generated according to the argument ``spacing`` (either a grid or a random 
        flat distribution) in the interval. The points in the grid falling outside 
        :attr:`Likelihood.pars_bounds <DNNLikelihood.Likelihood.pars_bounds>` are automatically removed.
        The values of the parameters and of logpdf at the local maxima are stored in the attributes
        :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>` and 
        :attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Likelihood.Y_prof_logpdf_max>`, respectively.
        They could be used both for frequentist profiled likelihood inference or as initial condition for
        Markov Chain Monte Carlo through the :class:`Sampler <DNNLikelihood.Sampler>` object
        (see :ref:`the Sampler object <sampler_object>`). 
        The method uses the function :func:`inference.find_prof_maximum <DNNLikelihood.inference.find_prof_maximum>`
        based on |scipy_optimize_minimize_link| to find the (local) minimum of minus
        :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>`. Since the latter
        method already contains a bounded logpdf, ``pars_bounds`` is set to ``None`` in the 
        :func:`inference.find_prof_maximum <DNNLikelihood.inference.find_prof_maximum>`
        function to optimize speed. See the documentation of the
        :func:`inference.find_prof_maximum <DNNLikelihood.inference.find_prof_maximum>` 
        function for details.

        When using interactive python in Jupyter notebooks if ``verbose=2`` a progress bar is shown through 
        the |ipywidgets_link| package.

        - **Arguments**

            - **pars**
            
                List of positions of the parameters under which logpdf 
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
            
                It can be either ``"grid"`` or ``"random"``. Depending on its value the ``n_points`` for each parameter are taken on an 
                equally spaced grid or are generated randomly in the interval.

                    - **type**: ``str``
                    - **accepted**: ``"grid"`` or ``"random"``
                    - **default**: ``grid``

            - **append**
            
                If ``append=False`` the values of
                :attr:`Likelihood.X_prof_logpdf_max <DNNLikelihood.Likelihood.X_prof_logpdf_max>` and 
                :attr:`Likelihood.Y_prof_logpdf_max <DNNLikelihood.Likelihood.Y_prof_logpdf_max>`
                are replaced, otherwise, newly computed values are appended to the existing ones.
                If the shape ot the newly computed ones is incompatible with the one of the existing ones,
                new values are saved in the temporary attributes 
                :attr:`Likelihood.X_prof_logpdf_max_tmp <DNNLikelihood.Likelihood.X_prof_logpdf_max_tmp>` and 
                :attr:`Likelihood.Y_prof_logpdf_max_tmp <DNNLikelihood.Likelihood.Y_prof_logpdf_max_tmp>` and 
                a warning message is generated. Notice that the latter two attributes, as suggested by their names,
                are temporary, are not saved by the :meth:`Likelihood.save_likelihood <DNNLikelihood.Likelihood.save_likelihood>`
                method, and are always initialized to ``None`` when the :class:`Likelihood <DNNLikelihood.Likelihood>` object
                is created.

                    - **type**: ``str``
                    - **accepted**: ``"grid"`` or ``"random"``
                    - **default**: ``grid``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

.. |ipywidgets_link| raw:: html
    
    <a href="https://ipywidgets.readthedocs.io/en/latest/"  target="_blank"> ipywidgets</a>
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if verbose == 2:
            progressbar = True
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                show_prints.verbose = True
                print("If you want to show a progress bar please install the ipywidgets package.",show=verbose)
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
        print("Total number of points:", len(pars_vals),".",show=verbose)
        pars_vals_bounded = []
        for i in range(len(pars_vals)):
            if (np.all(pars_vals[i] >= self.pars_bounds[pars, 0]) and np.all(pars_vals[i] <= self.pars_bounds[pars, 1])):
                pars_vals_bounded.append(pars_vals[i])
        if len(pars_vals) != len(pars_vals_bounded):
            print("Deleted", str(len(pars_vals)-len(pars_vals_bounded)),"points outside the parameters allowed range.",show=verbose)
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
                    print("New values have been appended to the existing ones.",show=verbose)
                else:
                    self.X_prof_logpdf_max_tmp = X_tmp
                    self.Y_prof_logpdf_max_tmp = Y_tmp
                    print("New values and existing ones have different shape and cannot be concatenated. New values stored in the temporary attributes 'X_prof_logpdf_max_tmp' and 'Y_prof_logpdf_max_tmp'.", show=verbose)
            else:
                self.X_prof_logpdf_max = X_tmp
                self.Y_prof_logpdf_max = Y_tmp
        end = timer()
        print("Log-pdf values lie in the range [",np.min(self.Y_prof_logpdf_max),",",np.max(self.Y_prof_logpdf_max),"]",show=verbose)
        print(len(pars_vals_bounded),"local maxima computed in", end-start, "s.",show=verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "computed profiled maxima", "pars": pars, "pars_ranges": pars_ranges, "number of maxima": len(X_tmp)}
        self.save_likelihood_log(overwrite=True, verbose=verbose_sub)

    def plot_logpdf_par(self, pars=0, npoints=100, pars_init=None, pars_labels="original", overwrite=False, verbose=None):
        """
        Plots the logpdf as a function of of the parameter ``par`` in the range ``(min,max)``
        using a number ``npoints`` of points. Only the parameter ``par`` is veried, while all other parameters are kept
        fixed to their value given in ``pars_init``. The function used for the plot is provided by the 
        :meth:`Likelihood.logpdf_fn <DNNLikelihood.Likelihood.logpdf_fn>`.

        - **Arguments**

            - **pars**
            
                List of lists containing the position of the parametes in the parameters vector, 
                and their minimum and maximum value for the plot.
                For example, to plot parameters ``1`` in the rage ``(1,3)`` and parameter ``5`` in the range
                ``(-3,3)`` one should set ``pars = [[1,1,3],[5,-3,3]]``. 

                    - **type**: ``list``
                    - **shape**: ``[[par,par_max,par_min],...]``

            - **npoints**
            
                Number of points in which the ``(par_min,par_max)`` range
                is divided to compute the logpdf and make the plot.

                    - **type**: ``int``
                    - **default**: ``100``

            - **pars_init**
            
                Central point in the parameter space from which ``par`` is varied and all other parameters are 
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
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``


            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces files**

            - :attr:`Likelihood.figure_files_base_path <DNNLikelihood.Likelihood.figure_files_base_path>` ``+ "_par_" + str(par[0]) + ".pdf"`` for each ``par`` in ``pars``.
            
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        pars_labels = self.__set_pars_labels(pars_labels)
        if pars_init is None:
            pars_init = self.pars_init
        for par in pars:
            par_number = par[0]
            par_min = par[1]
            par_max = par[2]
            vals = np.linspace(par_min, par_max, npoints)
            points = np.array(np.broadcast_to(pars_init,(npoints,len(pars_init))),dtype="float")
            points[:, par_number] = vals
            logpdf_vals = [self.logpdf_fn(point, *self.logpdf_args) for point in points]
            plt.plot(vals, logpdf_vals)
            plt.title(r"%s" % self.name)#.replace("_", "\_"))  # , fontsize=10)
            plt.xlabel(r"%s" % pars_labels[par_number])
            plt.ylabel(r"logpdf")
            plt.tight_layout()
            figure_filename = self.figure_files_base_path + "_par_"+str(par[0])+".pdf"
            utils.append_without_duplicate(self.figures_list, figure_filename)
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            print('Saved figure', figure_filename+'.', show=verbose)
            if verbose:
                plt.show()
            plt.close()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.log[timestamp] = {
                "action": "saved figure", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            self.save_likelihood_log(overwrite=True, verbose=verbose_sub)
