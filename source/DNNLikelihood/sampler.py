__all__ = ["Sampler"]

import builtins
import codecs
import importlib
import json
import sys
import time
from copy import copy
from datetime import datetime
from multiprocessing import Pool
from os import path, listdir
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.optimize import minimize

import emcee

from . import show_prints, utils
from .data import Data
from .show_prints import print

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")


class Sampler(show_prints.Verbosity):
    """
    This class contains the ``Sampler`` object, which allows to perform Markov Chain Monte Carlo
    (MCMC) using the |emcee_link| package (ensemble sampling MCMC). See ref. :cite:`ForemanMackey:2012ig` for
    details about |emcee_link|. On top of performing
    MCMC the ``Sampler`` object contains several methods to check convergence, and export ``Data`` objects
    that can be used to train and test the DNNLikelihood.
    The object can be instantiated both passing a ``Likelihood`` object or a ``likelihood_script_file`` created 
    with the ``Likelihood.save_likelihood_script`` method.

.. |emcee_link| raw:: html
    
    <a href="https://emcee.rhttps://emcee.readthedocs.io/en/stable/"  target="_blank"> emcee</a>
    """
    def __init__(self,
                 new_sampler=None,
                 likelihood_script_file=None,
                 likelihood=None,
                 nsteps=None,
                 moves_str=None,
                 parallel_CPU=None,
                 vectorize=None,
                 sampler_input_file=None,
                 verbose=True
                 ):
        """
        Instantiates the ``Sampler`` object.
        
        The ``Sampler`` object is instantiated differently depending on the value of ``new_sampler``:
        
        1. ``new_sampler=True``: all attributes are set from input arguments. The 
        :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute is set from
        from ``likelihood_script_file`` input if given, otherwise from ``likelihood`` input if given, otherwise from
        ``sampler_input_file``. Afterwards the ``__init_likelihood`` method is called.
        
        2. ``new_sampler=False``: the :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`
        attribute is set from ``sampler_input_file`` if given, otherwise from ``likelihood_script_file`` input if given, 
        otherwise from ``likelihood``. Arguments are set from 
        :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`. Afterwards the ``__init_likelihood`` 
        method is called. If the import fails ``__init__`` proceeds with ``new_sampler=True`` (see 1.).

        Attributes that are always set from input arguments (if they are not ``None``)

            - :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>`
            - :attr:`Sampler.nsteps <DNNLikelihood.Sampler.new_sampler>`
            - :attr:`Sampler.moves_str <DNNLikelihood.Sampler.new_sampler>`
            - :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.new_sampler>`
            - :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>`
            - :attr:`Sampler.verbose <DNNLikelihood.Sampler.verbose>`

        List of arguments that are set from :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`:

            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>`
            - :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>`
            - :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>` (if not given as input)
            - :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` (if not given as input)
            - :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` (if not given as input)
            - :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` (if not given as input)
            - :attr:`Sampler.log <DNNLikelihood.Sampler.log>`
            - :attr:`Sampler.figures_list <DNNLikelihood.Sampler.figures_list>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`
            - :attr:`Sampler.output_files_base_path <DNNLikelihood.Sampler.output_files_base_path>`
            - :attr:`Sampler.sampler_output_json_file <DNNLikelihood.Sampler.sampler_output_json_file>`
            - :attr:`Sampler.sampler_output_log_file <DNNLikelihood.Sampler.sampler_output_log_file>`
            - :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
            - :attr:`Sampler.data_output_file <DNNLikelihood.Sampler.data_output_file>`
            - :attr:`Sampler.figure_files_base_path <DNNLikelihood.Sampler.figure_files_base_path>`

        List of arguments that are set from ``__init_likelihood``:

            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>` (if not imported from :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`)
            - :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>`
            - :attr:`Sampler.pars_pos_poi <DNNLikelihood.Sampler.pars_pos_poi>`
            - :attr:`Sampler.pars_pos_nuis <DNNLikelihood.Sampler.pars_pos_nuis>`
            - :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`
            - :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
            - :attr:`Sampler.generic_pars_labels <DNNLikelihood.Sampler.generic_pars_labels>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` (if not imported from :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`)
            - :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`
            - :attr:`Sampler.ndims <DNNLikelihood.Sampler.ndims>`

        Attributes that are set after ``__init_likelihood``:

            - :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` (set by :meth:`Sampler.__load_backend <DNNLikelihood.Sampler._Sampler__load_backend>`)
            - :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` (set by :meth:`Sampler.__load_backend <DNNLikelihood.Sampler._Sampler__load_backend>`)
            - :attr:`Sampler.moves <DNNLikelihood.Sampler.moves>` (set by evaluating :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>`)

        Checks done by ``__init__``:

            - :meth:`Sampler.__check_vectorize <DNNLikelihood.Sampler._Sampler__check_vectorize>`: checks consistency between :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` and :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :meth:`Sampler.__check_params_backend <DNNLikelihood.Sampler._Sampler__check_params_backend>` check consistency of parameters in the backend

        - **Arguments**

            See Class arguments.
        """
        #show_prints.verbose = verbose
        self.verbose = verbose
        verbose,verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Setting all arguments
        if new_sampler is None:
            self.new_sampler=False
        else:
            self.new_sampler=new_sampler
        self.likelihood_script_file = likelihood_script_file
        self.likelihood = likelihood
        self.nsteps = nsteps
        if moves_str is not None:
            self.moves_str = moves_str
        else:
            print("No moves_str parameter has been specified. moves has been set to the default StretchMove() of emcee")
            self.moves_str = "[(emcee.moves.StretchMove(), 1), (emcee.moves.GaussianMove(0.0005, mode='random', factor=None), 0)]"
        if parallel_CPU is None:
            self.parallel_CPU = True
        else:
            self.parallel_CPU = parallel_CPU
        if vectorize is None:
            self.vectorize = False
        else:
            self.vectorize = vectorize
        self.sampler_input_file = sampler_input_file
        ### Initializing existing sampler
        if not self.new_sampler:
            ### Tries to determine self.sampler_input_file from input arguements
            if self.sampler_input_file is not None:
                self.sampler_input_file = path.abspath(path.splitext(self.sampler_input_file)[0])
            else:
                if self.likelihood_script_file is not None:
                    ### Try to detemine sampler_input_file from likelihood_script_file
                    self.__get_sampler_input_file_from_likelihood_script_file()
                else:
                    if self.likelihood is not None:
                        ### Try to detemine sampler_input_file from likelihood
                        self.__get_sampler_input_file_from_likelihood(verbose=verbose_sub)
                    else:
                        raise Exception("You have to specify at least one argument among 'likelihood', 'likelihood_script_file', and 'sampler_input_file'.")
            self.sampler_input_json_file = path.abspath(self.sampler_input_file+".json")
            self.sampler_input_log_file = path.abspath(self.sampler_input_file+".log")
            try:
                self.__load_sampler(verbose=verbose_sub)
                ##  Defines: name, likelihood_script_file, nsteps, moves_str, parallel_CPU, vectorize, log, figures_list, output_files_base_path,
                #   sampler_output_json_file, sampler_output_log_file, backend_file, data_output_file, figure_files_base_path
                ##  Already defined: verbose
                ##  Updates nsteps, moves_str, parallel_CPU, and vectorize if given.
                self.nsteps = nsteps
                if moves_str is not None:
                    self.moves_str = moves_str
                if parallel_CPU is not None:
                    self.parallel_CPU = parallel_CPU
                if vectorize is not None:
                    self.vectorize = vectorize
                self.__init_likelihood()
            except:
                self.new_sampler = True
                ### Already defined: likelihood_script_file, nsteps, moves_str, parallel_CPU, vectorize, verbose
        ### Initializing new sampler
        if self.new_sampler:
            ### Tries to determine self.likelihood_script_file from input arguements
            if self.likelihood_script_file is not None:
                self.likelihood_script_file = path.splitext(path.abspath(self.likelihood_script_file))[0]+".py"
            else:
                if self.likelihood is not None:
                    ### Try to detemine sampler_input_file from likelihood
                    self.__get_likelihood_script_file_from_likelihood(verbose=verbose_sub)
                else:
                    if self.sampler_input_file is not None:
                        self.__get_likelihood_script_file_from_sampler_input_file()
                    else:
                        raise Exception("You have to specify at least one argument among 'likelihood', 'likelihood_script_file', and 'sampler_input_file'.")
            ## Should define: log, figures_list, output_files_base_path, sampler_output_json_file, sampler_output_log_file,
            #  backend_file, data_output_file, figure_files_base_path
            self.log = {timestamp: {"action": "created"}}
            self.figures_list = []
            self.__init_likelihood()
            self.output_files_base_path = path.join(self.output_folder, self.name)
            self.sampler_output_json_file = self.output_files_base_path+".json"
            self.sampler_output_log_file = self.output_files_base_path+".log"
            self.backend_file = self.output_files_base_path+"_backend.h5"
            self.data_output_file = self.output_files_base_path+"_data.h5"
            self.figure_files_base_path = self.output_files_base_path+"_figure"                       
        #### Checking vectorize, setting moves, backend, sampler, and figures_list, and checking parameters consistency in backend
        self.__check_vectorize(verbose=verbose_sub)
        self.moves = eval(self.moves_str)
        self.__load_backend(verbose=verbose_sub)
        self.__check_params_backend(verbose=verbose_sub)
        try:
            del(self.likelihood)
        except:
            pass
        #### Saving sampler
        if self.new_sampler:
            self.save_sampler_json(overwrite=False, verbose=verbose_sub)
        else:
            self.save_sampler_json(overwrite=True, verbose=verbose_sub)

    def __get_likelihood_script_file_from_sampler_input_file(self):
        self.sampler_input_file = path.abspath(path.splitext(self.sampler_input_file)[0])
        self.likelihood_script_file = self.sampler_input_file.replace("sampler","likelihood_script.py")

    def __get_likelihood_script_file_from_likelihood(self,verbose=None):
        verbose,_=self.set_verbosity(verbose)
        tmp_likelihood = copy(self.likelihood)
        tmp_likelihood.verbose = verbose
        self.likelihood_script_file = tmp_likelihood.likelihood_script_file
        tmp_likelihood.save_likelihood_script(verbose=verbose)

    def __get_sampler_input_file_from_likelihood_script_file(self):
        self.likelihood_script_file = path.splitext(path.abspath(self.likelihood_script_file))[0]+".py"
        folder, file = path.split(self.likelihood_script_file)
        file = path.splitext(file)[0]
        sampler_input_file_name = file.replace("likelihood_script","sampler")
        self.sampler_input_file = path.join(folder, sampler_input_file_name)

    def __get_sampler_input_file_from_likelihood(self, verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        self.__get_likelihood_script_file_from_likelihood(verbose=verbose)
        self.__get_sampler_input_file_from_likelihood_script_file()

    def __init_likelihood(self):
        in_folder, in_file = path.split(self.likelihood_script_file)
        in_file = path.splitext(in_file)[0]
        sys.path.insert(0, in_folder)
        lik = importlib.import_module(in_file)
        #### Setting attributes from Likelihood object
        try:
            self.name
        except:
            self.name = lik.name.replace("likelihood", "sampler")
        self.logpdf = lik.logpdf
        self.logpdf_args = lik.logpdf_args
        self.pars_pos_poi = lik.pars_pos_poi
        self.pars_pos_nuis = lik.pars_pos_nuis
        self.pars_init_vec = lik.pars_init_vec
        self.pars_labels = lik.pars_labels
        self.generic_pars_labels = utils.define_generic_pars_labels(self.pars_pos_poi, self.pars_pos_nuis)
        try:
            self.output_folder
        except:
            self.output_folder = lik.output_folder
        self.nwalkers = len(lik.pars_init_vec)
        self.ndims = len(self.pars_init_vec[0])

    def __load_sampler_log(self,verbose=None):
        """
        Private method used by the ``__init__`` one to load the ``Sampler`` object from the file 
        :attr:`Sampler.sampler_input_json_file <DNNLikelihood.Sampler.sampler_input_json_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        self.set_verbosity(verbose)
        start = timer()
        with open(self.sampler_input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "loaded", "file name": path.split(self.sampler_input_log_file)[-1], "file path": self.sampler_input_log_file}
        print('Loaded sampler log in', str(end-start), '.')
        #self.save_sampler_log(overwrite=True, verbose=False)

    def __load_sampler_json(self, verbose=None):
        """
        Private method used by the ``__init__`` one to load the ``Sampler`` object from the file 
        :attr:`Sampler.sampler_input_json_file <DNNLikelihood.Sampler.sampler_input_json_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        self.set_verbosity(verbose)
        start = timer()
        with open(self.sampler_input_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "loaded", "file name": path.split(self.sampler_input_json_file)[-1], "file path": self.sampler_input_json_file}
        self.save_sampler_log(overwrite=True, verbose=False)
        print('Loaded sampler in', str(end-start), '.')
        

    def __load_sampler(self, verbose=None):
        """
        Private method used by the ``__init__`` one to load the ``Sampler`` object from the file 
        :attr:`Sampler.sampler_input_json_file <DNNLikelihood.Sampler.sampler_input_json_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`verbosity mode <verbosity_mode>` for general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose,_ =self.set_verbosity(verbose)
        self.__load_sampler_log(verbose=verbose)
        self.__load_sampler_json(verbose=verbose)
        
        

    def __check_vectorize(self,verbose=None):
        self.set_verbosity(verbose)
        if self.vectorize:
            try:
                self.logpdf(self.pars_init_vec[0:2],*self.logpdf_args)
            except:
                print("vectorize has been set to True, but logpdf does not seem to be vectorized. Please check your input if you want to use a vectorized logpdf. Continuing with vectorize=False.")
                self.vectorize = False
        if self.vectorize:
            self.parallel_CPU = False
            print("Since vectorize=True the parameter parallel_CPU has been set to False.")

    def __load_backend(self, verbose=None):
        """
        Created a backend :attr:``Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
        when :attr:``Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` is set to ``True`` or
        loads an existig backend :attr:``Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
        when :attr:``Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` is set to ``False``.
        In case :attr:``Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` is set to ``False`` but the backend file
        :attr:``Sampler.backend_file <DNNLikelihood.Sampler.backend_file>` does not exist, ``new_sampler`` is automatically 
        set to ``True`` and a new one is created.
        The method creates or loads the backedn by calling the :meth:``Sampler.sampler <DNNLikelihood.Sampler.run_sampler>`
        method with zero steps. This properly sets both attributes :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>`
        and :attr:``Sampler.sampler <DNNLikelihood.Sampler.sampler>`.

        - **Arguments**

            - **verbose**

                Verbosity mode.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        self.set_verbosity(verbose)
        if not self.new_sampler:
            if not path.exists(self.backend_file):
                print("The new_sampler flag was set to false but the backend file", self.backend_file,
                      "does not exists.\nPlease change filename if you meant to import an existing backend.\nContinuing with new_sampler=True.")
                self.new_sampler = True
        if self.new_sampler:
            #print("Creating sampler in backend file", self.backend_file)
            utils.check_rename_file(self.backend_file)
            nsteps_tmp = self.nsteps
            self.nsteps = 0
            start = timer()
            self.run_sampler(progress=False, verbose=False)
            self.set_verbosity(verbose)
            self.nsteps = nsteps_tmp
            end = timer()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "created backend", "file name": path.split(self.backend_file)[-1], "file path": self.backend_file}
            print("Created backend", self.backend_file, "for chains", self.name, "in", end-start, "s.")
        else:
            #print("Loading existing sampler from backend file", self.backend_file)
            nsteps_tmp = self.nsteps
            self.nsteps = 0
            self.backend = None
            start = timer()
            self.run_sampler(progress=False, verbose=False)
            self.set_verbosity(verbose)
            self.nsteps = nsteps_tmp
            #self.nsteps = self.sampler.iteration
            end = timer()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "loaded backend", "file name": path.split(self.backend_file)[-1], "file path": self.backend_file}
            print("Loaded backend", self.backend_file, "for chains",self.name, "in", end-start, "s.")
            print("Available number of steps: {0}.".format(self.backend.iteration))
        self.save_sampler_log(overwrite=True, verbose=False)

    def __check_params_backend(self,verbose=None):
        """
        Checks consistency between the parameters ``nwalkers``, ``ndims``, and ``nsteps`` assigned in the 
        :meth:``Sampler.__init__ <DNNLikelihood.Sampler.__init__> and the corresponding ones in the existing backend.
        If ``nwalkers`` or ``ndims`` are found to be inconsistent an exception is raise. If ``nsteps`` is found to 
        be inconsistent it is set to the number of available steps in the backend.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        """
        self.set_verbosity(verbose)
        nwalkers_from_backend, ndims_from_backend = self.backend.shape
        nsteps_from_backend = self.backend.iteration
        if nwalkers_from_backend != self.nwalkers:
            raise Exception("Number of walkers (nwalkers) determined from the input likelihood is inconsitent with the loaded backend. Please check inputs.")
        if ndims_from_backend != self.ndims:
            raise Exception("Number of steps (nsteps)  determined from the input likelihood is inconsitent with loaded backend. Please check inputs.")
        if nsteps_from_backend > self.nsteps:
            print("Specified number of steps nsteps is inconsitent with loaded backend. nsteps has been set to",nsteps_from_backend, ".")
            self.nsteps = nsteps_from_backend

    def __set_steps_to_run(self,verbose=None):
        """
        Based on the number of steps already available in the current :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>`,
        it sets the remaining number of steps to run to reach :attr:``Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`. If the
        value of the latter is less or equal to the number of available steps, a warning message asking to increase the value 
        of ``nsteps`` is printed.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        """
        self.set_verbosity(verbose)
        try:
            nsteps_current = self.backend.iteration
        except:
            nsteps_current = 0
        if self.nsteps <= nsteps_current:
            print("Please increase nsteps to run for more steps")
            nsteps_to_run = 0
        else:
            nsteps_to_run = self.nsteps-nsteps_current
        return nsteps_to_run

    def __set_pars_labels(self, pars_labels):
        """
        Returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
                and :attr:`Sampler.generic_pars_labels <DNNLikelihood.Sampler.generic_pars_labels>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                It is used to set the parameters labels in output figures.

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

    def run_sampler(self, progress=True, verbose=None):
        """
        Constructs the attributed :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>` and 
        :attr:``Sampler.sampler <DNNLikelihood.Sampler.sampler>` and calls the ``sampler.run_mcmc`` function to
        run the sampler. Depending on the value of :attr:``Sampler.parallel_cpu <DNNLikelihood.Sampler.parallel_cpu>`
        the sampler is run on a single core (if ``False``) or in parallel using the ``Multiprocessing.Pool`` method
        (if ``True``). When running in parallel, the number of processes is set to the number of available (physical)
        cpu cores using the ``psutil`` package by ``n_processes = psutil.cpu_count(logical=False)``.
        See the documentation of the |emcee_link| and |multiprocessing_link| packages for more details on parallel
        sampling.

        If running a new sampler, the initial value of walkers is set to 
        :attr:``Sampler.pars_init <DNNLikelihood.Sampler.pars_init>`, otherwise it is set to the state of the walkers
        in the last step available in :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>`.

        A progress bar of the sampling is shown by default.

        - **Arguments**

            - **progress**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 

.. |multiprocessing_link| raw:: html
    
    <a href="https://docs.python.org/3/library/multiprocessing.html"  target="_blank"> multiprocessing</a>

.. |psutil_link| raw:: html
    
    <a href="https://pypi.org/project/psutil/"  target="_blank"> psutil</a>

        """
        verbose ,_= self.set_verbosity(verbose)
        if verbose==2:
            progress=True
        # Initializes backend (either connecting to existing one or generating new one)
        # Initilized p0 (chains initial state)
        if self.new_sampler:
            print("Initialize backend in file", self.backend_file)
            self.backend = emcee.backends.HDFBackend(self.backend_file, name=self.name)
            self.backend.reset(self.nwalkers, self.ndims)
            p0 = self.pars_init_vec
        else:
            if self.backend is None:
                try:
                    print("Initialize backend in file", self.backend_file)
                    self.backend = emcee.backends.HDFBackend(self.backend_file, name=self.name)
                    #print(self.backend.iteration)
                    #print(self.nstepss)
                    show_prints.verbose = verbose
                except:
                    raise Exception("Backend file does not exist. Please either change the filename or run with new_sampler=True.")
            try:
                p0 = self.backend.get_last_sample()
            except:
                p0 = self.pars_init_vec
        print("Initial number of steps: {0}".format(self.backend.iteration))

        # Defines sampler and runs the chains
        start = timer()
        nsteps_to_run = self.__set_steps_to_run(verbose=verbose)
        #print(nsteps_to_run)
        if nsteps_to_run == 0:
            progress = False
        if self.parallel_CPU:
            n_processes = psutil.cpu_count(logical=False)
            #if __name__ == "__main__":
            if progress:
                print("Running", n_processes, "parallel processes.")
            with Pool(n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves, pool=pool, backend=self.backend, args=self.logpdf_args)
                self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress, store = True)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves,args=self.logpdf_args, backend=self.backend, vectorize=self.vectorize)
            self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress, store = True)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "run sampler", "nsteps": nsteps_to_run, "available steps": self.backend.iteration}
        self.save_sampler_log(overwrite=True, verbose=verbose)
        print("Done in", end-start, "seconds.")
        print("Final number of steps: {0}.".format(self.backend.iteration))

    def save_sampler_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute in the file
        :attr:`Sampler.sampler_output_json_file <DNNLikelihood.Sampler.sampler_output_json_file>`

        This method is called with ``overwrite=False`` and ``verbose=False`` when the object is created from input arguments
        and with ``overwrite=True`` and ``verbose=False`` each time the 
        :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute is updated.

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

            - :attr:`Sampler.sampler_output_log_file <DNNLikelihood.Sampler.sampler_output_log_file>`
        """
        self.set_verbosity(verbose)
        time.sleep(1)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.sampler_output_log_file)
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #self.log[timestamp] = {"action": "saved", "file name": path.split(self.likelihood_output_log_file)[-1], "file path": self.likelihood_output_log_file}
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.sampler_output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Sampler log file", self.sampler_output_log_file,"saved in", str(end-start), "s.")

    def save_sampler_json(self, overwrite=False, verbose=None):
        """
        ``Sampler`` objects are saved as follows.
        The attributes 
        
            - :attr:`Sampler.sampler_input_file <DNNLikelihood.Sampler.sampler_input_file>`
            - :attr:`Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>`
            - :attr:`Sampler.verbose <DNNLikelihood.Sampler.verbose>`

        are not saved.

        The attribute :attr:`Sampler.log <DNNLikelihood.Sampler.log>` is saved to the file :attr:`Sampler.sampler_output_log_file <DNNLikelihood.Sampler.sampler_output_log_file>`
        
        The attributes which are also attributes of the corresponding :class:`Likelihood <DNNLikelihood.Likelihood>` object,
        or that are determined by them, that is
        
            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>`
            - :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>`
            - :attr:`Sampler.pars_pos_poi <DNNLikelihood.Sampler.pars_pos_poi>`
            - :attr:`Sampler.pars_pos_nuis <DNNLikelihood.Sampler.pars_pos_nuis>`
            - :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`
            - :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
            - :attr:`Sampler.generic_pars_labels <DNNLikelihood.Sampler.generic_pars_labels>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`
            - :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`
            - :attr:`Sampler.ndim <DNNLikelihood.Sampler.ndim>`

        are not saved and always taken from the corresponding likelihood.

        The ``object`` attributes are treated as follows

            - :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` is saved in the backend file :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
            - :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` is reconstructed from backend and the other attributes
            - :attr:`Sampler.moves <DNNLikelihood.Sampler.moves>` is always build from :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>`

        All other attributes, that is

            - :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>
            - :attr:`Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`
            - :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>`
            - :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>`
            - :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>`
            - :attr:`Sampler.output_files_base_path <DNNLikelihood.Sampler.output_files_base_path>`
            - :attr:`Sampler.sampler_output_json_file <DNNLikelihood.Sampler.sampler_output_json_file>`
            - :attr:`Sampler.sampler_output_log_file <DNNLikelihood.Sampler.sampler_output_log_file>`
            - :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
            - :attr:`Sampler.data_output_file <DNNLikelihood.Sampler.data_output_file>`
            - :attr:`Sampler.figure_files_base_path <DNNLikelihood.Sampler.figure_files_base_path>`

        are saved in the :attr:`Sampler.sampler_output_json_file <DNNLikelihood.Sampler.sampler_output_json_file>`.

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

            - :attr:`Likelihood.sampler_output_json_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`

        - **Updates file**

            - :attr:`Likelihood.sampler_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`
        """
        self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.sampler_output_json_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "saved", "file name": path.split(self.sampler_output_json_file)[-1], "file path": self.sampler_output_json_file}
        dictionary = utils.dic_minus_keys(self.__dict__, ["sampler_input_file", "new_sampler","log", 
                                                          "logpdf", "logpdf_args", "pars_pos_poi",
                                                          "pars_pos_nuis", "pars_init_vec", "pars_labels", "generic_pars_labels",
                                                          "nwalkers", "ndim", "backend", "sampler", "moves","verbose"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.sampler_output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Likelihood json file", self.sampler_output_json_file,"saved in", str(end-start), "s.")
        self.save_sampler_log(overwrite=overwrite, verbose=verbose)

    def get_data_object(self, nsamples="all", burnin=0, thin=1, dtype='float64', test_fraction=0, save=True, verbose=None):
        """
        Returns a :class:`DNNLikelihood.Data` object with ``nsamples`` samples by taking chains and logpdf values, discarding ``burnin`` steps,
        thinning by ``thin`` and converting to dtype ``dtype``. When ``nsamples="All"`` all samples available for the 
        given choice of ``burnin`` and ``thin`` are included to the :class:`DNNLikelihood.Data` object, otherwise only the first
        ``nsamples`` are included. If ``nsamples`` is more than the available number all the available samples are included
        and a warning message is printed.
        Before including samples in the :class:`DNNLikelihood.Data` object the method checks if there are duplicate samples
        (which would suggest a larger value of ``thin``) and non finite values of logpdf (e.g. ``np.nan`` or ``np.inf``)
        and print a warning in any of these cases.

        The method also allows one to pass to the :class:`DNNLikelihood.Data` a value for ``test_fraction``, which already
        splits data into ``train`` (sample from which training and valudation data are extracted) and ``test`` (sample only
        used for final test) sets. See :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` for more details.

        Finally, based on the value of ``save``, the generated :class:`DNNLikelihood.Data` object

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``            

            - **verbose**
            
                Verbosity mode. 
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True`` 
        """
        self.set_verbosity(verbose)
        print("Notice: When requiring an unbiased data sample please check that the required burnin is compatible with MCMC convergence.")
        start = timer()
        if nsamples is "all":
            allsamples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
            logpdf_values = self.sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
        else:
            if nsamples > (self.nsteps-burnin)*self.nwalkers/thin:
                print("Less samples than available are requested. Returning all available samples:",
                  str((self.nsteps-burnin)*self.nwalkers/thin),"\nYou may try to reduce burnin and/or thin to get more samples.")
                allsamples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
                logpdf_values = self.sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            else:
                burnin = self.nsteps-nsamples*thin/self.nwalkers
                allsamples=self.sampler.get_chain(discard = burnin, thin = thin, flat = True)
                logpdf_values=self.sampler.get_log_prob(discard = burnin, thin = thin, flat = True)
        if len(np.unique(logpdf_values, axis=0, return_index=False)) < len(logpdf_values):
            print("There are non-unique samples")
        if np.count_nonzero(np.isfinite(logpdf_values)) < len(logpdf_values):
            print("There are non-numeric logpdf values.")
        end = timer()
        print(len(allsamples), "unique samples generated in", end-start, "s.")
        data_sample_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ds = Data(data_X=allsamples,
                  data_Y=logpdf_values,
                  dtype=dtype,
                  pars_pos_poi=self.pars_pos_poi,
                  pars_pos_nuis=self.pars_pos_nuis,
                  pars_labels=self.pars_labels,
                  test_fraction=test_fraction,
                  name=self.name+"_"+data_sample_timestamp,
                  data_sample_input_filename=None,
                  data_sample_output_filename=self.data_output_file,
                  load_on_RAM=False)
        if save:
            ds.save_samples()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "created data object", "file name": path.split(self.data_output_file)[-1], "file path": self.data_output_file}
        self.save_sampler_log(overwrite=True, verbose=False)
        return ds
    
    ##### Functions from the emcee documentation (with some modifications) #####

    def autocorr_func_1d(self,x, norm=True,verbose=None):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        
.. |emcee_tutorial_autocorr_link| raw:: html
    
    <a href="https://emcee.readthedocs.io/en/stable/tutorials/autocorr/"  target="_blank"> emcee autocorrelation tutorial</a>
        """
        self.set_verbosity(verbose)
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError( "invalid dimensions for 1D autocorrelation function")
        if len(np.unique(x)) == 1:
            print("Chain does not change in "+str(len(x))+" steps. Autocorrelation for this chain may return nan.")
        n = utils.next_power_of_two(len(x))
        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2*n)
        acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
        acf /= 4*n
        # Optionally normalize
        if norm:
            acf /= acf[0]
        return acf

    # Automated windowing procedure following Sokal (1989)
    def auto_window(self,taus, c):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010(self, y, c=5.0, verbose=None):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        f = self.autocorr_func_1d(np.mean(y, axis=0), verbose=verbose_sub)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_new(self,y, c=5.0, verbose=None):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        f = np.zeros(y.shape[1])
        counter=0
        for yy in y:
            fp = self.autocorr_func_1d(yy, verbose=verbose_sub)
            if np.isnan(np.sum(fp)):
                print("Chain",counter,"returned nan. Values changed to 0 to proceed.")
                fp = np.full(len(fp),0)
            f += fp
            counter += 1
        f /= len(y)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_ml(self, y, thin=1, c=5.0, bound=5.0,verbose=True):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        from celerite import terms, GP
        # Compute the initial estimate of tau using the standard method
        init = self.autocorr_new(y, c=c, verbose=verbose_sub)
        z = y[:, ::thin]
        N = z.shape[1]

        # Build the GP model
        tau = max(1.0, init / thin)
        kernel = terms.RealTerm(
            np.log(0.9 * np.var(z)), 
            -np.log(tau), 
            bounds=[(-bound, bound), (-np.log(N), 0.0)]
        )
        kernel += terms.RealTerm(
            np.log(0.1 * np.var(z)),
            -np.log(0.5 * tau),
            bounds=[(-bound, bound), (-np.log(N), 0.0)],
        )
        gp = GP(kernel, mean=np.mean(z))
        gp.compute(np.arange(z.shape[1]))

        # Define the objective
        def nll(p):
            # Update the GP model
            gp.set_parameter_vector(p)

            # Loop over the chains and compute likelihoods
            v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

            # Combine the datasets
            return -np.sum(v), -np.sum(g, axis=0)

        # Optimize the model
        p0 = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(nll, p0, jac=True, bounds=bounds)
        gp.set_parameter_vector(soln.x)

        # Compute the maximum likelihood tau
        a, c = kernel.coefficients[:2]
        tau = thin * 2 * np.sum(a / c) / np.sum(a)
        return tau

    def gelman_rubin(self, pars=0, nsteps="all"):
        """
        Given a parameter (or list of parameters) ``pars`` and a number of ``nsteps``, the method computes 
        the Gelman-Rubin :cite:`Gelman:1992zz` ratio and related quantities for monitoring convergence.
        The formula for :math:`R_{c}` implements the correction due to :cite:`Brooks_1998` and is implemented here as
        
        .. math::
            R_{c} = \\sqrt{\\frac{\\hat{d}+3}{\\hat{d}+1}\\frac{\\hat{V}}{W}}.

        See the original papers for the notation.
        
        In order to be able to monitor not only the :math:`R_{c}` ratio, but also the values of :math:`\\hat{V}` 
        and :math:`W` independently, the method also computes these quantities. Usually a reasonable convergence 
        condition is :math:`R_{c}<1.1`, together with stability of both :math:`\hat{V}` and :math:`W` :cite:`Brooks_1998`.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: 0

            - **nsteps**

                If ``"all"`` then all nsteps available in current ``backend`` are included. Otherwise an integer
                number of steps or a list of integers to monitor for different steps numbers can be input.

                    - **type**: ``str`` or ``int`` or ``list``
                    - **allowed str**: ``all``
                    - **shape of list**: ``[ ]``
                    - **default**: ``all``
        
        - **Returns**

            An array constructed concatenating lists of the type ``[par, nsteps, Rc, Vhat, W]`` for each parameter
            and each choice of nsteps.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(len(pars)*len(nsteps),5)``
        """
        res = []
        pars = np.array([pars]).flatten()
        for par in pars:
            if nsteps is "all":
                chain = self.sampler.get_chain()[:, :, par]
                si2 = np.var(chain, axis=0, ddof=1)
                W = np.mean(si2, axis=0)
                ximean = np.mean(chain, axis=0)
                xmean = np.mean(ximean, axis=0)
                n = chain.shape[0]
                m = chain.shape[1]
                B = n / (m - 1) * np.sum((ximean - xmean)**2, axis=0)
                sigmahat2 = (n - 1) / n * W + 1 / n * B
                # Exact
                Vhat = sigmahat2+B/m/n
                varVhat = ((n-1)/n)**2 * 1/m * np.var(si2, axis=0)+((m+1)/(m*n))**2 * 2/(m-1) * B**2 + 2*(
                    (m+1)*(n-1)/(m*(n**2)))*n/m * (np.cov(si2, ximean**2)[0, 1]-2*xmean*np.cov(si2, ximean)[0, 1])
                df = (2*Vhat**2) / varVhat
                Rc = np.sqrt((Vhat / W)*(df+3)/(df+1))  # correct Brooks-Gelman df
                res.append([par, n, Rc, Vhat, W])
            else:
                nsteps = np.array([nsteps]).flatten()
                for step in nsteps:
                    chain = self.sampler.get_chain()[:step, :, par]
                    si2 = np.var(chain, axis=0, ddof=1)
                    W = np.mean(si2, axis=0)
                    ximean = np.mean(chain, axis=0)
                    xmean = np.mean(ximean, axis=0)
                    n = chain.shape[0]
                    m = chain.shape[1]
                    B = n / (m - 1) * np.sum((ximean - xmean)**2, axis=0)
                    sigmahat2 = (n - 1) / n * W + 1 / n * B
                    # Exact
                    Vhat = sigmahat2+B/m/n
                    varVhat = ((n-1)/n)**2 * 1/m * np.var(si2, axis=0)+((m+1)/(m*n))**2 * 2/(m-1) * B**2 + 2*((m+1)*(n-1)/(m*(n**2)))*n/m *(np.cov(si2,ximean**2)[0,1]-2*xmean*np.cov(si2,ximean)[0,1])
                    df = (2*Vhat**2) / varVhat
                    Rc = np.sqrt((Vhat / W)*(df+3)/(df+1)) #correct Brooks-Gelman df
                    res.append([par, n, Rc, Vhat, W])
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "compited Gelman-Rubin", "pars": pars, "nsteps": nsteps}
        self.save_sampler_log(overwrite=True, verbose=False)
        return np.array(res)

    def plot_gelman_rubin(self, pars=0, npoints=5, pars_labels="original", overwrite=False, verbose=None):
        """
        Produces plots of the evolution with the number of steps of the convergence metrics :math:`R_{c}`, 
        :math:`\\sqrt{\\hat{V}}`, and :math:`\\sqrt{W}` computed by the method 
        :meth:`Sampler.gelman_rubin <DNNLikelihood.Sampler.gelman_rubin>` for parameter (or list of parameters) ``pars``. 
        The plots are produced by computing the quantities in ``npoints`` equally spaced (in base-10 log scale) points  
        between one and the total number of available steps. 

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **npoints**

                Number of points in which the convergence metrics are computed to produce the plot.
                The points are taken equally spaced in base-10 log scale.

                    - **type**: ``int``
                    - **default**: ``5``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
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

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        verbose, _ = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_files_base_path
        for par in pars:
            idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps))))])
            idx = np.unique(idx[idx <= self.nsteps])
            idx = utils.get_spaced_elements(idx, numElems=npoints+1)
            idx = idx[1:]
            gr = self.gelman_rubin(par, nsteps=idx)
            plt.plot(gr[:,1], gr[:,2], '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\hat{R}_{c}(%s)$" % (pars_labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            figure_filename = filename+"_GR_Rc_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            plt.plot(gr[:, 1], np.sqrt(gr[:, 3]), '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{\hat{V}}(%s)$"% (pars_labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            figure_filename = filename+"_GR_sqrtVhat_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            plt.plot(gr[:, 1], np.sqrt(gr[:, 4]), '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{W}(%s)$"% (pars_labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            figure_filename = filename+"_GR_sqrtW_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
        self.save_sampler_log(overwrite=True, verbose=False)
            
    def plot_dist(self, pars=0, pars_labels="original", overwrite=False, verbose=None):
        """
        Plots the 1D distribution of parameter (or list of parameters) ``pars``.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
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

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        verbose ,_= self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_files_base_path
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            counts, bins = np.histogram(chain.flatten(), 100)
            integral = counts.sum()
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.step(bins[:-1], counts/integral, where='post')
            plt.xlabel(r"$%s$" % (pars_labels[par].replace('$', '')))
            plt.ylabel(r"$p(%s)$" % (pars_labels[par].replace('$', '')))
            plt.tight_layout()
            figure_filename = filename+"_distr_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
        self.save_sampler_log(overwrite=True, verbose=False)

    def plot_autocorr(self, pars=0, pars_labels="original", methods=["G&W 2010", "DFM 2017", "DFM 2017: ML"], overwrite=False, verbose=None):
        """
        Plots the autocorrelation time estimate evolution with the number of steps for parameter (or list of parameters) ``pars``.
        Three different methods are used to estimate the autocorrelation time: "G&W 2010", "DFM 2017", and "DFM 2017: ML", described in details
        in the |emcee_tutorial_autocorr_link|. The function accepts a list of methods and by default it makes the plot including all available
        methods. Notice that to use the method "DFM 2017: ML" based on fitting an autoregressive model, the |celerite_link| package needs to be
        installed.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**

                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **methods**

                List of methods to estimate the autocorrelation time. The three availanle methods are "G&W 2010", "DFM 2017", and "DFM 2017: ML". 
                One curve for each method will be produced.
                    - **type**: ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``["G&W 2010", "DFM 2017", "DFM 2017: ML"]``

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``

.. |celerite_link| raw:: html
    
    <a href="https://celerite.readthedocs.io/en/stable/"  target="_blank"> celerite</a>
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_files_base_path
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            # Compute the largest number of duplicated at the beginning of chains
            n_dupl = []
            for c in chain:
                n_dupl.append(utils.check_repeated_elements_at_start(c))
            n_start = max(n_dupl)+10
            if n_start > 100:
                print("There is at least one chain starting with", str(
                        n_start-10), "duplicate steps. Autocorrelation will be computer starting at", str(n_start), "steps.")
            else:
                n_start = 100
            N = np.exp(np.linspace(np.log(n_start), np.log(chain.shape[1]), 10)).astype(int)
            # GW10 method
            if "G&W 2010" in methods:
                gw2010 = np.empty(len(N))
            # New method
            if "DFM 2017" in methods:
                new = np.empty(len(N))
            # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                new = np.empty(len(N))
                ml = np.empty(len(N))
                ml[:] = np.nan

            for i, n in enumerate(N):
                # GW10 method
                if "G&W 2010" in methods:
                    gw2010[i] = self.autocorr_gw2010(chain[:, :n], verbose=verbose_sub)
                # New method
                if "DFM 2017" in methods or "DFM 2017: ML" in methods:
                    new[i] = self.autocorr_new(chain[:, :n],verbose=verbose_sub)
                # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                succeed = None
                bound = 5.0
                while succeed is None:
                    try:
                        for i, n in enumerate(N[1:-1]):
                            k = i + 1
                            thin = max(1, int(0.05 * new[k]))
                            ml[k] = self.autocorr_ml(chain[:, :n], thin=thin, bound=bound,verbose=verbose_sub)
                        succeed = True
                        if bound > 5.0:
                            print("Succeeded with bounds (",str(-(bound)), ",", str(bound), ").")
                    except:
                        print("Bounds (", str(-(bound)), ",", str(bound), ") delivered non-finite log-prior. Increasing bound to (",
                              str(-(bound+5)), ",", str(bound+5), ") and retrying.")
                        bound = bound+5
            # Plot the comparisons
            plt.plot(N, N / 50.0, "--k", label=r"$\tau = S/50$")
            #plt.plot(N, N / 100.0, "--k", label=r"$\tau = S/100$")
            # GW10 method
            if "G&W 2010" in methods:
                plt.loglog(N, gw2010, "o-", label=r"G\&W 2010")
            # New method
            if "DFM 2017" in methods:
                plt.loglog(N, new, "o-", label="DFM 2017")
            # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                plt.loglog(N, ml, "o-", label="DFM 2017: ML")
            ylim = plt.gca().get_ylim()
            plt.ylim(ylim)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\tau_{%s}$ estimates" % (pars_labels[par].replace('$', '')))
            plt.legend()
            plt.tight_layout()
            figure_filename = filename+"_autocorr_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
        self.save_sampler_log(overwrite=True, verbose=False)

    def plot_chains(self, pars=0, n_chains=100, pars_labels="original", overwrite=False, verbose=None):
        """
        Plots the evolution of chains (walkers) with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:``Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **n_chains**
            
                The number of chains to 
                add to the plot.
                    - **type**: ``int``
                    - **default**: ``100``

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        verbose, _ =self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_files_base_path
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("n_chains larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.")
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par]
            idx = np.sort([(i)*(10**j) for i in range(1, 11)
                           for j in range(int(np.ceil(np.log10(self.nsteps))))])
            idx = np.unique(idx[idx < len(chain)])
            plt.plot(idx,chain[idx][:,rnd_chains], '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$%s$" %(pars_labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            figure_filename = filename+"_chains_"+str(par)+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
            print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
        self.save_sampler_log(overwrite=True, verbose=False)

    def plot_chains_logprob(self, n_chains=100, overwrite=False, verbose=None):
        """
        Plots the evolution of minus the logpdf values with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:``Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **n_chains**
            
                The number of chains to 
                add to the plot.
                    - **type**: ``int``
                    - **default**: ``100``

            - **overwrite**
            
                It determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` is used  
                to append a time-stamp to the file name.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`Verbosity mode <verbosity_mode>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        self.set_verbosity(verbose)
        filename = self.figure_files_base_path
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("n_chains larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.")
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        chain_lp = self.sampler.get_log_prob()
        idx = np.sort([(i)*(10**j) for i in range(1, 11)
                       for j in range(int(np.ceil(np.log10(self.nsteps))))])
        idx = np.unique(idx[idx < len(chain_lp)])
        plt.plot(idx, -chain_lp[:, rnd_chains][idx], '-', alpha=0.8)
        plt.xlabel("number of steps, $S$")
        plt.ylabel(r"-logpdf")
        plt.xscale('log')
        plt.tight_layout()
        figure_filename = filename+"_chains_logpdf.pdf"
        if not overwrite:
            utils.check_rename_file(figure_filename)
        plt.savefig(figure_filename)
        utils.append_without_duplicate(self.figures_list, figure_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log[timestamp] = {"action": "saved", "file name": path.split(figure_filename)[-1], "file path": figure_filename}
        print('Saved figure', figure_filename+'.')
        if verbose:
            plt.show()
        plt.close()
        self.save_sampler_log(overwrite=True, verbose=False)
