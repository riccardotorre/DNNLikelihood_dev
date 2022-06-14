__all__ = ["Sampler"]

import builtins
import codecs
import importlib
import json
import sys
import time
from copy import copy
from datetime import datetime
from multiprocessing import Pool, freeze_support
from os import listdir, path
from shutil import copyfile
from timeit import default_timer as timer

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.optimize import minimize

import emcee

from . import utils
from .data import Data
from .show_prints import Verbosity, print

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

header_string = "=============================="
footer_string = "------------------------------"

class Sampler(Verbosity):
    """
    This class contains :ref:`the Sampler object <sampler_object>`, which allows to perform Markov Chain Monte Carlo
    (MCMC) using the |emcee_link| package (ensemble sampling MCMC). See ref. :cite:`ForemanMackey:2012ig` for
    details about |emcee_link|. On top of performing
    MCMC :class:`Sampler <DNNLikelihood.Sampler>` class several methods to check convergence, and export 
    the :mod:`Data <data>` object used to train and test the DNNLikelihood.
    The object can be instantiated both passing a ``Lik`` object or a ``likelihood_script_file`` created 
    with the ``Lik.save_script`` method.
    """
    def __init__(self,
                 likelihood_script_file=None,
                 likelihood=None,
                 nwalkers=None,
                 nsteps_required=None,
                 moves_str=None,
                 parallel_CPU=None,
                 vectorize=None,
                 output_folder=None,
                 input_file=None,
                 verbose=True
                 ):
        """
        The :class:`Sampler <DNNLikelihood.Sampler>` object can be initialized in two different ways, depending on the value of 
        the local variable ``new_sampler``, which is set to ``True`` if ``input_file`` is ``None`` and to ``False`` if ``input_file``
        is not ``None``. Depending on its value the object is initialized as follows:

        1. ``new_sampler`` is ``False``
        
            The :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`
            attribute is set from :argument:`input_file`. Attributes are set from 
            :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>` through the 
            :meth:`Sampler.__load__ <DNNLikelihood.Sampler.__init__>` method. Afterwards the 
            :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` 
            method is called. If the import fails :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` 
            proceeds with ``new_sampler=True`` (see below).
            Depending on the value of the input argument :argument:`output_folder` the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` method behaves as follows:

                - If :argument:`output_folder` is ``None`` (default)
                    
                    The attribute :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`
                    is set from the :attr:`Sampler.input_folder <DNNLikelihood.Sampler.input_folder>` one.
                - If :argument:`output_folder` corresponds to a path different from that stored in the loaded :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` attribute
                    
                    - if path stored in the loaded :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` attribute exists, then its content is copied to the new ``output_folder`` (if the new ``output_foler`` already exists it is renamed by adding a timestamp);
                    - if path stored in the loaded :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` does not exists, then the content of the path :attr:`Sampler.input_folder <DNNLikelihood.Sampler.input_folder>` is copied to the new ``output_folder``.
                - If :argument:`output_folder` corresponds to the same path stored in the loaded :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` attribute
                    
                    Output folder, files, and path attributes are not updated and everything is read from the loaded object.
        
        2. ``new_sampler`` is ``True`` (default)
            
            All attributes are set from input arguments. The 
            :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute is set from
            from ``likelihood_script_file`` input if given, otherwise from ``likelihood`` input. 
            Afterwards the :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` 
            method is called.

        Attributes that are always set from input arguments (if they are not ``None``)

            - :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>`
            - :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`
            - :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>`
            - :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>`
            - :attr:`Sampler.verbose <DNNLikelihood.Sampler.verbose>`

        Attributes that are set from :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`
        if ``new_sampler=False``:

            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>`
            - :attr:`Sampler.input_folder <DNNLikelihood.Sampler.input_folder>`
            - :attr:`Sampler.input_h5_file <DNNLikelihood.Sampler.input_h5_file>`
            - :attr:`Sampler.input_log_file <DNNLikelihood.Sampler.input_log_file>`
            - :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>`
            - :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` (if not given as input)
            - :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>` (if not given as input)
            - :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` (if not given as input)
            - :attr:`Sampler.predictions <DNNLikelihood.Sampler.predictions>`
            - :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` (if not given as input)
            - :attr:`Sampler.log <DNNLikelihood.Sampler.log>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` (if not given as input)
            - :attr:`Sampler.output_h5_file <DNNLikelihood.Sampler.output_h5_file>` (same as :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`)
            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`(same as :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`)
            - :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>` (same as :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`)
            - :attr:`Sampler.output_figures_base_file <DNNLikelihood.Sampler.output_figures_base_file>`(same as :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`)

        Attributes that are set by :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>`:

            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>` (if not imported from :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`)
            - :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>`
            - :attr:`Sampler.pars_pos_poi <DNNLikelihood.Sampler.pars_pos_poi>`
            - :attr:`Sampler.pars_pos_nuis <DNNLikelihood.Sampler.pars_pos_nuis>`
            - :attr:`Sampler.pars_central <DNNLikelihood.Sampler.pars_central>`
            - :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`
            - :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
            - :attr:`Sampler.pars_labels_auto <DNNLikelihood.Sampler.pars_labels_auto>`
            - :attr:`Sampler.pars_bounds <DNNLikelihood.Sampler.pars_bounds>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` (if not imported from :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`)
            - :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`
            - :attr:`Sampler.ndims <DNNLikelihood.Sampler.ndims>`

        Attributes that are set after :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood>` :

            - :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` (set by :meth:`Sampler.__init_backend <DNNLikelihood.Sampler._Sampler__init_backend>`)
            - :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` (set by :meth:`Sampler.__init_backend <DNNLikelihood.Sampler._Sampler__init_backend>`)
            - :attr:`Sampler.moves <DNNLikelihood.Sampler.moves>` (set by evaluating :attr:`Sampler.moves_str <DNNLikelihood.Sampler.moves_str>`)

        Checks done by :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` :

            - :meth:`Sampler.__check_vectorize <DNNLikelihood.Sampler._Sampler__check_vectorize>`: checks consistency between :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` and :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :meth:`Sampler.__check_params_backend <DNNLikelihood.Sampler._Sampler__check_params_backend>` check consistency of parameters in the backend

        - **Arguments**

            See Class arguments.

        - **Creates/updates file**

            - :attr:`Sampler.output_h5_file <DNNLikelihood.Sampler.output_h5_file>`
            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
            - :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
            - :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` (if not found in the :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` folder)
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        print(header_string,"\nInitialize Sampler object.\n",show=verbose)
        # Setting all arguments
        self.likelihood_script_file = likelihood_script_file
        self.likelihood = likelihood
        self.output_folder = output_folder
        self.input_file = input_file
        if self.input_file is None:
            new_sampler=True
        else:
            new_sampler=False
        self.nwalkers = nwalkers
        self.nsteps_required = nsteps_required
        if moves_str != None:
            self.moves_str = moves_str
        if parallel_CPU == None:
            self.parallel_CPU = True
        else:
            self.parallel_CPU = parallel_CPU
        if vectorize == None:
            self.vectorize = False
        else:
            self.vectorize = vectorize
        
        self.output_folder = output_folder
        self.__check_define_input_files(verbose=verbose_sub)
        if not new_sampler:
            #self.__load(verbose=verbose_sub)
            try:
                self.__load(verbose=verbose_sub)
                self.nsteps_required = nsteps_required
                if moves_str != None:
                    self.moves_str = moves_str
                if parallel_CPU != None:
                    self.parallel_CPU = parallel_CPU
                if vectorize != None:
                    self.vectorize = vectorize
                self.__init_likelihood(verbose=verbose_sub) # Also sets self.name
                self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
                try:
                    self.predictions["gelman_rubin"]
                    self.predictions["Figures"]
                except:
                    self.reset_predictions(delete_figures=True, verbose=verbose_sub)
            except:
                print(header_string,"\nNo sampler files have been found. Initializing a new Sampler object.\n",show=verbose)
                new_sampler = True
        if new_sampler:
            if moves_str == None:
                print(header_string,"\nNo 'moves_str' argument specified: the 'moves_str' attribute has been set to the emcee default 'StretchMove()'.\n", show=verbose)
                self.moves_str = "[(emcee.moves.StretchMove(), 1), (emcee.moves.GaussianMove(0.0005, mode='random', factor=None), 0)]"
            self.log = {timestamp: {"action": "created"}}
            self.predictions = {"gelman_rubin": {}, 
                                "Figures": {}}
            self.__init_likelihood(verbose=verbose_sub) # Also sets self.name
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
        self.moves = eval(self.moves_str)
        self.__check_vectorize(verbose=verbose_sub)
        self.__init_backend(verbose=verbose_sub)
        self.__check_params_backend(verbose=verbose_sub)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
        del(self.likelihood)
        self.save(overwrite=True, verbose=verbose_sub)
        if path.split(self.likelihood_script_file)[0] != self.output_folder:
            copyfile(self.likelihood_script_file, path.join(self.output_folder,path.split(self.likelihood_script_file)[1]))
            self.likelihood_script_file = path.join(self.output_folder,path.split(self.likelihood_script_file)[1])
        

    def __check_define_input_files(self,verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one
        to set the input files attributes. The local variable ``new_sampler`` is set to ``True`` if ``input_file`` 
        is ``None`` and to ``False`` if ``input_file`` is not ``None``. Depending on its value input files are set
        as follows:

        1. ``new_sampler=True``: the 
            :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute is set from
            from ``likelihood_script_file`` input if given, otherwise from ``likelihood`` input.

        2. ``new_sampler=False``:  both the 
            :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>` and
            :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>`
            attributes are set from ``input_file``.

        The method also sets the attributes corresponding to input files
        
            - :attr:`Sampler.input_h5_file <DNNLikelihood.Sampler.input_h5_file>`,
            - :attr:`Sampler.input_log_file <DNNLikelihood.Sampler.input_log_file>`

        and input folder 
        
            - :attr:`Sampler.input_folder <DNNLikelihood.Sampler.input_folder>`

        depending on the value of the 
        :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        ### Sets output folder if needed to find existing files
        if self.input_file is None:
            new_sampler = True
        else:
            new_sampler = False
        if new_sampler:
            if self.likelihood_script_file is not None:
                self.likelihood_script_file = path.splitext(path.abspath(self.likelihood_script_file))[0]+".py"
                #print("Option 1: 'likelihood_script_file' is given as input")
            else:
                if self.likelihood is not None:
                    ### Try to detemine likelihood_script_file from likelihood
                    #print("Option 2: 'likelihood' is given as input")
                    self.__get_likelihood_script_file_from_likelihood(verbose=verbose_sub)
                else:
                    raise Exception("You have to specify at least one argument among 'likelihood', 'likelihood_script_file', and 'input_file'.")
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.__get_likelihood_script_file_from_input_file()
        try:
            self.input_h5_file = path.abspath(self.input_file+".h5")
            self.input_log_file = path.abspath(self.input_file+".log")
            self.input_folder = path.split(self.input_file)[0]
            print(header_string,"\nSampler input folder set to\n\t", self.input_folder,".\n",show=verbose)
        except:
            self.input_h5_file = None
            self.input_log_file = None
            self.input_folder = None
            print(header_string,"\nNo Sampler input files and folders specified.\n", show=verbose)
                
    def __check_define_output_files(self,timestamp=None,verbose=False):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one
        to set the attributes corresponding to the output folder

            - :attr:`Sampler.output_figures_folder <DNNLikelihood.Sampler.output_figures_folder>`

        and output files

            - :attr:`Sampler.output_h5_file <DNNLikelihood.Sampler.output_h5_file>`
            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
            - :attr:`Sampler.output_predictions_json_file <DNNLikelihood.Sampler.output_predictions_json_file>`
            - :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
            - :attr:`Sampler.output_figures_base_file <DNNLikelihood.Sampler.output_figures_base_file>`

        depending on the value of the 
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` attribute.
        The latter is set from the input argument :argument:`output_folder` if given, otherwise it is
        taken from the same attribute of the :class:`Lik <DNNLikelihood.Lik>` object when creating a new
        :class:`Sampler <DNNLikelihood.Sampler>` object or imported from files if loading an existing
        :class:`Sampler <DNNLikelihood.Sampler>` object.
        It also creates the folders
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` and
        :attr:`Sampler.output_figures_folder <DNNLikelihood.Sampler.output_figures_folder>` if 
        they do not exist.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates folders if they do not exist**

            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`
            - :attr:`Sampler.output_figures_folder <DNNLikelihood.Sampler.output_figures_folder>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.output_folder is not None:
            self.output_folder = path.abspath(self.output_folder)
            if self.input_folder is not None and self.output_folder != self.input_folder:
                utils.copy_and_save_folder(self.input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if self.input_folder is not None:
                self.output_folder = self.input_folder
            else:
                self.output_folder = path.abspath("")
        self.output_folder = utils.check_create_folder(self.output_folder)
        self.output_figures_folder =  utils.check_create_folder(path.join(self.output_folder, "figures"))
        self.output_h5_file = path.join(self.output_folder, self.name+".h5")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        self.output_predictions_json_file = path.join(self.output_folder, self.name+"_predictions.json")
        self.backend_file = path.join(self.output_folder, self.name+"_backend.h5")
        self.output_figures_base_file_name = self.name+"_figure"
        self.output_figures_base_file_path = path.join(self.output_figures_folder, self.output_figures_base_file_name)
        print(header_string,"\nSampler output folder set to\n\t", self.output_folder, ".\n",show=verbose)

    def __get_likelihood_script_file_from_input_file(self):
        """
        Private method that attempts to determine the 
        :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute
        from the :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`
        attribute.
        """
        self.likelihood_script_file = path.join(path.split(self.input_file)[0],path.split(self.input_file)[1].replace("sampler","likelihood_script.py"))

    def __get_likelihood_script_file_from_likelihood(self,verbose=None):
        """
        Private method that generates the
        :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` attribute
        and corresponding file from the 
        :attr:`Sampler.likelihood <DNNLikelihood.Sampler.likelihood>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        tmp_likelihood = copy(self.likelihood)
        tmp_likelihood.verbose = verbose_sub
        tmp_likelihood_script_file = tmp_likelihood.script_file
        likelihood_script_file_name = path.split(tmp_likelihood_script_file)[-1]
        tmp_likelihood_script_file_new = path.join(self.output_folder,likelihood_script_file_name)
        if not path.exists(tmp_likelihood_script_file_new):
            if not path.exists(tmp_likelihood_script_file):
                try:
                    timestamp=list(tmp_likelihood.predictions["logpdf_profiled_max"].keys())[-1]
                    tmp_likelihood.save_script(timestamp=timestamp,verbose=verbose_sub)
                except:
                    tmp_likelihood.save_script(verbose=verbose_sub)
            self.likelihood_script_file = tmp_likelihood_script_file
        else:
            self.likelihood_script_file = tmp_likelihood_script_file_new
            #copyfile(tmp_likelihood_script_file,tmp_likelihood_script_file_new)

    def __init_likelihood(self,verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one
        to set the likelihood related attributes from the 
        :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>` attribute.
        It imports the latter as a module, which instantiates a :class:`Lik <DNNLikelihood.Lik>`
        object and defines parameters. It is used to set the attributes:

            - :attr:`Sampler.name <DNNLikelihood.Sampler.name>` (if not imported from :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`)
            - :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>`
            - :attr:`Sampler.pars_central <DNNLikelihood.Sampler.pars_central>`
            - :attr:`Sampler.pars_pos_poi <DNNLikelihood.Sampler.pars_pos_poi>`
            - :attr:`Sampler.pars_pos_nuis <DNNLikelihood.Sampler.pars_pos_nuis>`
            - :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`
            - :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
            - :attr:`Sampler.pars_bounds <DNNLikelihood.Sampler.pars_bounds>`
            - :attr:`Sampler.pars_labels_auto <DNNLikelihood.Sampler.pars_labels_auto>`
            - :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` (if not imported from :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`)
            - :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`
            - :attr:`Sampler.ndims <DNNLikelihood.Sampler.ndims>`

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        #Set verbosity in likelihood_script_file
        utils.replace_strings_in_file(self.likelihood_script_file,["verbose = True","verbose = False","verbose = None"],"verbose = "+str(verbose_sub))
        in_folder, in_file = path.split(self.likelihood_script_file)
        in_file = path.splitext(in_file)[0]
        sys.path.insert(0, in_folder)
        lik = importlib.import_module(in_file)
        try:
            self.name
        except:
            self.name = lik.name.replace("likelihood", "sampler")
        self.logpdf = lik.logpdf
        self.logpdf_args = lik.logpdf_args
        self.logpdf_kwargs = lik.logpdf_kwargs
        self.ndims = lik.ndims
        if self.nwalkers is None:
            self.nwalkers = 2*self.ndims
            print(header_string,"\nThe 'nwalkers' input argument was set to None. 'nwalkers' attribute has been set to twice the number of dimensions (",self.nwalkers,").\n", show=verbose)
        self.pars_central = lik.pars_central
        self.pars_pos_poi = lik.pars_pos_poi
        self.pars_pos_nuis = lik.pars_pos_nuis
        if lik.pars_init_vec is None:
            self.pars_init_vec = np.array([np.random.normal(lik.pars_central) for i in range(self.nwalkers)])
            print(header_string,"\nNo profiled maxima information were find in the Likelihood object. \
                   The walkers initialization has been generated with gaussian smooth of the central parameters values.\n",show=verbose)
        else:
            if len(lik.pars_init_vec) >= self.nwalkers:
                print(header_string,"\nThe walkers initialization has been generated from profiled maxima information available in the Likelihood object.\n",show=verbose)
                self.pars_init_vec = lik.pars_init_vec[:self.nwalkers]
            else:
                print(header_string,"\nProfiled maxima information found in the Likelihood object. However, the number of maxima is less than the number of walkers.\
                       The walkers initialization has been generated from profiled maxima information available in the Likelihood object \
                       and, for the missing walkers, with gaussian smooth of the central parameters values.\n", show=verbose)
                self.pars_init_vec = np.concatenate([lik.pars_init_vec, np.array([np.random.normal(lik.pars_central) for i in range(self.nwalkers-len(lik.pars_init_vec))])])
        self.pars_labels = lik.pars_labels
        self.pars_bounds = lik.pars_bounds
        self.pars_labels_auto = utils.define_pars_labels_auto(self.pars_pos_poi, self.pars_pos_nuis)
        if self.output_folder == None:
            self.output_folder = lik.output_folder
        self.nwalkers = len(self.pars_init_vec)
        #lik.output_folder = path.abspath(self.output_folder)
        #lik.save_script(verbose=verbose_sub)
        
    def __load(self,verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one 
        to load a previously saved
        :class:`Sampler <DNNLikelihood.Sampler>` object from the files 
        
            - :attr:`Sampler.input_h5_file <DNNLikelihood.Sampler.input_h5_file>`
            - :attr:`Sampler.input_log_file <DNNLikelihood.Sampler.input_log_file>`

        The method loads, with the |deepdish_link| package, the content od the 
        :attr:`Sampler.input_h5_file <DNNLikelihood.Sampler.input_h5_file>` file into a temporary dictionary, subsequently used to update the 
        :attr:`Sampler.__dict__ <DNNLikelihood.Sampler.__dict__>` attribute.
        The method also loads the content of the :attr:`Sampler.input_log_file <DNNLikelihood.Sampler.input_log_file>`
        file, assigning it to the :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        start = timer()
        dictionary = dd.io.load(self.input_h5_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_h5_file)[-1],
                                                path.split(self.input_log_file)[-1]]}
        print(header_string,"\nSampler object loaded in", str(end-start), ".\n",show=verbose)

    def __init_backend(self, verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one 
        to initialize the backend. The local variable ``new_sampler`` is set to ``True`` if ``input_file`` 
        is ``None`` and to ``False`` if ``input_file`` is not ``None``.
        Depending on its value the backend is initialized as follows:

        1. ``new_sampler=True``: the backend is initialized in the file
        :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`
        
        2. ``new_sampler`` is ``False``: the backend is loaded from the existing backend file 
        :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`.

        In case ``new_sampler`` is ``False`` but the backend file
        :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>` does not exist, then
        ``new_sampler`` is automatically set to ``True`` and a new backend is created.
        The method creates or loads the backend by calling the :meth:`Sampler.__init_sampler <DNNLikelihood.Sampler._Sampler__init_sampler>`
        private method, which creates and runs the sampler for zero steps. This is done to properly sets both attributes 
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` and :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>`.
        Notice that when the object is loaded from input files, 
        but a new :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` is specified,
        then the function attemts to copy an existing backend file from
        :attr:`Sampler.input_folder <DNNLikelihood.Sampler.input_folder>` to
        :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if self.input_file is None:
            new_sampler = True
        else:
            new_sampler = False
        if not new_sampler:
            # Try to copy backend file in input directory to new backend file self.backend_file
            if self.input_folder is not None:
                old_backend_file = path.join(self.input_folder, self.name+"_backend.h5")
                if self.input_folder != self.output_folder:
                    if path.exists(old_backend_file):
                        copyfile(old_backend_file, self.backend_file)
                        print(header_string,"\nA backend file has been found in the input folder and has been copied to the new output folder.\n")
            if not path.exists(self.backend_file):
                print(header_string,"\nAn input_file has been specified but the corresponding backend file\n\t", self.backend_file,
                      "\ndoes not exists.\nPlease change filename if you meant to import an existing backend.\nContinuing with a new sampler.\n",show=verbose)
                new_sampler = True
        if new_sampler:
            utils.check_rename_file(self.backend_file,verbose=verbose_sub)
            nsteps_tmp = self.nsteps_required
            self.nsteps_required = 0
            start = timer()
            self.__init_sampler(verbose=verbose_sub)
            self.nsteps_required = nsteps_tmp
            end = timer()
            self.log[timestamp] = {"action": "created backend", 
                                   "file name": path.split(self.backend_file)[-1]}
            print(header_string,"\nCreated backend\n\t", self.backend_file, "\nfor chains", self.name, "in", end-start, "s.\n",show=verbose)
        else:
            nsteps_tmp = self.nsteps_required
            self.nsteps_required = 0
            self.backend = None
            start = timer()
            self.__init_sampler(verbose=verbose_sub)
            self.nsteps_required = nsteps_tmp
            #self.nsteps_required = self.sampler.iteration
            end = timer()
            self.log[timestamp] = {"action": "loaded backend", 
                                   "file name": path.split(self.backend_file)[-1]}
            print(header_string,"\nLoaded backend\n\t", self.backend_file, "\nfor chains",self.name, "in", end-start, "s.\n",show=verbose)

    def __init_sampler(self, verbose=None):
        """
        Private method used by the :meth:`Sampler.__init_backend <DNNLikelihood.Sampler._Sampler__init_backend>` one 
        to initialize :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` and 
        :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` attributes. 
        The local variable ``new_sampler`` is set to ``True`` if ``input_file`` 
        is ``None`` and to ``False`` if ``input_file`` is not ``None``. Depending on its value the sampler is initialized
        as follows:

        1. ``new_sampler=True``: a new backend is created,
        walkers are initialized to :attr:`Sampler.pars_init <DNNLikelihood.Sampler.pars_init>`, and MCMC is run for zero
        steps to properly create the backend file and initialize the sampler.
        
        2. ``new_sampler=False``: the existing backend is set, walkers are initialized to 
        to the last sample available in :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>`, and MCMC is run for zero
        steps to properly initialize the :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>` attribute.

        The method runs the mcmc by calling the |emcee_run_mcmc_link| method.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_file is None:
            new_sampler = True
        else:
            new_sampler = False
        # Initializes backend (either connecting to existing one or generating new one)
        # Initilized p0 (chains initial state)
        if new_sampler:
            print(header_string,"\nInitialize backend in file\n\t", self.backend_file,"\n",show=verbose)
            self.backend = emcee.backends.HDFBackend(self.backend_file, name=self.name)
            self.backend.reset(self.nwalkers, self.ndims)
            p0 = self.pars_init_vec
        else:
            if self.backend == None:
                try:
                    print(header_string,"\nInitialize backend from file\n\t", self.backend_file,"\n",show=verbose)
                    self.backend = emcee.backends.HDFBackend(self.backend_file, name=self.name)
                except:
                    raise Exception("Backend file does not exist. Please either change the filename or generate a new sampler.")
            try:
                p0 = self.backend.get_last_sample()
            except:
                p0 = self.pars_init_vec
        try:
            self.nsteps_available = self.backend.iteration
        except:
            self.nsteps_available = 0
        # Defines sampler and runs the chains
        start = timer()
        nsteps_to_run = self.__set_steps_to_run()
        #if self.parallel_CPU:
        #    n_processes = psutil.cpu_count(logical=False)
        #    with Pool(n_processes) as pool:
        #        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves, pool=pool, backend=self.backend, args=self.logpdf_args)
        #        self.sampler.run_mcmc(p0, nsteps_to_run, progress=False, store=True)
        #else:
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, args=self.logpdf_args, kwargs=self.logpdf_kwargs, moves=self.moves, backend=self.backend, vectorize=self.vectorize)
        self.sampler.run_mcmc(p0, nsteps_to_run, progress=False, store=True, skip_initial_state_check=True)
        if self.sampler._previous_state == None:
            try:
                self.sampler._previous_state = self.backend.get_last_sample()
            except:
                pass 
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "init sampler", 
                               "available steps": self.backend.iteration}
        print(header_string,"\nNumber of available steps: {0}.".format(self.backend.iteration),".\n",show=verbose)

    def __check_vectorize(self, verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one
        to check consistency between the :attr:`Sampler.vectorize <DNNLikelihood.Sampler.vectorize>` and
        :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>` attributes. In particular, when 
        :attr:`Sampler.vectorize=True <DNNLikelihood.Sampler.vectorize>`
        it tries to compute logpdf on a vector of parameters. If it fails, then it sets 
        :attr:`Sampler.vectorize=False <DNNLikelihood.Sampler.vectorize>`.
        Finally, whenever :attr:`Sampler.vectorize=False <DNNLikelihood.Sampler.vectorize>`, the attribute 
        :attr:`Sampler.parallel_CPU <DNNLikelihood.Sampler.parallel_CPU>` is set to ``False``.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.vectorize:
            try:
                self.logpdf(self.pars_init_vec[0:2], *self.logpdf_args)
            except:
                print(header_string,"\n'vectorize' has been set to True, but logpdf does not seem to be vectorized. Please check your input if you want to use a vectorized logpdf. Continuing with vectorize=False.\n", show=verbose)
                self.vectorize = False
        if self.vectorize:
            self.parallel_CPU = False
            print(header_string,"\nSince 'vectorize' has been set to True, the parameter 'parallel_CPU' has been automatically set to False.\n", show=verbose)

    def __check_params_backend(self,verbose=None):
        """
        Private method used by the :meth:`Sampler.__init__ <DNNLikelihood.Sampler.__init__>` one
        to check consistency between the attributes
        :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>` and :attr:`Sampler.ndims <DNNLikelihood.Sampler.ndims>`
        assigned by the :meth:`Sampler.__init_likelihood <DNNLikelihood.Sampler._Sampler__init_likelihood> method 
        with those determined from the existing backedn :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>`.
        It also checks if the value of :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` is smaller than the number
        of available steps in the backend and, in such case, it sets :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>`
        equal to the number of available steps.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        nwalkers_from_backend, ndims_from_backend = self.backend.shape
        if nwalkers_from_backend != self.nwalkers:
            raise Exception("Number of walkers (nwalkers) determined from the input likelihood is inconsitent with the loaded backend. Please check inputs.",show=verbose)
        if ndims_from_backend != self.ndims:
            raise Exception("Number of steps (nsteps)  determined from the input likelihood is inconsitent with loaded backend. Please check inputs.",show=verbose)
        if self.nsteps_required == None:
            self.nsteps_required = self.nsteps_available
        elif self.nsteps_available > self.nsteps_required:
            print(header_string,"\nSpecified number of steps nsteps is inconsitent with loaded backend. nsteps has been set to",self.nsteps_available, ".\n",show=verbose)
            self.nsteps_required = self.nsteps_available

    def __set_steps_to_run(self):
        """
        Private method that returns the number of steps to run computed as the difference between the value of
        :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` and the value of 
        :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`, i.e. the
        number of steps available in 
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>`.
        """
        if self.nsteps_required <= self.nsteps_available and self.nsteps_available > 0:
            nsteps_to_run = 0
        else:
            nsteps_to_run = self.nsteps_required-self.nsteps_available
        return nsteps_to_run

    def __set_pars_labels(self, pars_labels):
        """
        Private method that returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns the value of :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
                or :attr:`Sampler.pars_labels_auto <DNNLikelihood.Sampler.pars_labels_auto>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``
        """
        if pars_labels == "original":
            return self.pars_labels
        elif pars_labels == "generic":
            return self.pars_labels_auto
        else:
            return pars_labels

    def run_sampler(self, progress=True, verbose=None):
        """
        Runs MCMC sampling. Parameters are initialized to the last last sample available in 
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` or to 
        :attr:`Sampler.pars_init <DNNLikelihood.Sampler.pars_init>` if no sample is available.

        Depending on the value of :attr:`Sampler.parallel_cpu <DNNLikelihood.Sampler.parallel_cpu>`
        the sampler is run on a single core (if ``False``) or in parallel using the |multiprocessing_pool_link| method
        (if ``True``). When running in parallel, the number of processes is set to the number of available (physical)
        cpu cores using the |psutil_link| package by ``n_processes = psutil.cpu_count(logical=False)``.
        See the documentation of the |emcee_ensemble_sampler_link| class and of the |multiprocessing_link| package 
        for more details on parallel sampling.

        - **Arguments**

            - **progress**
            
                If ``True`` 
                then  a progress monitor is shown.
                    
                    - **type**: ``bool``
                    - **default**: ``True`` 

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nRunning sampler", show=verbose)
        # Initializes backend (either connecting to existing one or generating new one)
        # Initilized p0 (chains initial state)
        try:
            p0 = self.backend.get_last_sample()
        except:
            p0 = self.pars_init_vec
        print("Initial number of steps: {0}".format(self.backend.iteration), ".", show=verbose)
        # Defines sampler and runs the chains
        start = timer()
        nsteps_to_run = self.__set_steps_to_run()
        if nsteps_to_run == 0:
            progress = False
            print("Please increase nsteps to run for more steps.", show=verbose)
        if self.parallel_CPU:
            n_processes = psutil.cpu_count(logical=False)
            #if __name__ == "__main__":
            freeze_support()
            if progress:
                print("Running", n_processes,"parallel processes.", show=verbose)
            with Pool(n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(
                    self.nwalkers, self.ndims, self.logpdf, moves=self.moves, pool=pool, backend=self.backend, args=self.logpdf_args)
                self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress, store=True, skip_initial_state_check=True)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves, args=self.logpdf_args, backend=self.backend, vectorize=self.vectorize)
            self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress, store=True, skip_initial_state_check=True)
        self.nsteps_available = self.backend.iteration
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "run sampler", 
                               "nsteps": nsteps_to_run, 
                               "available steps": self.nsteps_available,
                               "file name": path.split(self.backend_file)[-1]}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nDone in", end-start, "seconds.", show=verbose)
        print("Final number of steps: {0}.".format(self.backend.iteration), ".", show=verbose)

    def save_log(self, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Sampler.log <DNNLikelihood.Sampler.log>` attribute in the file
        :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`

        This method is called by the methods
        
        - :meth:`Sampler.run_sampler <DNNLikelihood.Sampler.run_sampler>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.save <DNNLikelihood.Sampler.save>` with ``overwrite=overwrite`` and ``verbose=verbose``
        - :meth:`Sampler.get_data_object <DNNLikelihood.Sampler.get_data_object>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.compute_gelman_rubin <DNNLikelihood.Sampler.compute_gelman_rubin>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.plot_gelman_rubin <DNNLikelihood.Sampler.plot_gelman_rubin>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.plot_dist <DNNLikelihood.Sampler.plot_dist>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.plot_autocorr <DNNLikelihood.Sampler.plot_autocorr>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.plot_chains <DNNLikelihood.Sampler.plot_chains>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Sampler.plot_chains_logpdf <DNNLikelihood.Sampler.plot_chains>` with ``overwrite=True`` and ``verbose=verbose_sub``

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.
                    
            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_log_file = utils.generate_dump_file_name(self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(self.output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nLikelihood log file\n\t", output_log_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nLikelihood log file\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nLikelihood log file dump\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions_json(self, timestamp=None,overwrite=False, verbose=None):
        """ Save predictions json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_json_file = self.output_predictions_json_file
            if not overwrite:
                utils.check_rename_file(output_predictions_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_json_file = utils.generate_dump_file_name(self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1]}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nPredictions json file dump\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save(self, timestamp=None, overwrite=False, verbose=None):
        """
        The :class:`Sampler <DNNLikelihood.Sampler>` object is saved to the HDF5 file
        :attr:`Sampler.output_h5_file <DNNLikelihood.Sampler.output_h5_file>`, the sampler attribute
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` is saved to the HDF5 file
        :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`, and the object log is saved
        to the json file :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`.
        The object is saved by storing the content of the :attr:``Sampler.__dict__ <DNNLikelihood.Sampler.__dict__>`` 
        attribute in an h5 file using the |deepdish_link| package. The following attributes are excluded from the saved
        dictionary:

            - :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>` (saved to the file :attr:`Sampler.backend_file <DNNLikelihood.Sampler.backend_file>`)
            - :attr:`Sampler.input_file <DNNLikelihood.Sampler.input_file>`
            - :attr:`Sampler.input_h5_file <DNNLikelihood.Sampler.input_h5_file>`
            - :attr:`Sampler.input_log_file <DNNLikelihood.Sampler.input_log_file>`
            - :attr:`Sampler.log <DNNLikelihood.Sampler.log>` (saved to the file :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`)
            - :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`
            - :attr:`Sampler.moves <DNNLikelihood.Sampler.moves>`
            - :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>`
            - :attr:`Sampler.verbose <DNNLikelihood.Sampler.verbose>`

        and the attributes that are associated with the corresponding :class:`Lik <DNNLikelihood.Lik>` object:

            - :attr:`Sampler.logpdf <DNNLikelihood.Sampler.logpdf>`
            - :attr:`Sampler.logpdf_args <DNNLikelihood.Sampler.logpdf_args>`
            - :attr:`Sampler.logpdf_kwargs <DNNLikelihood.Sampler.logpdf_kwargs>`
            - :attr:`Sampler.ndims <DNNLikelihood.Sampler.ndims>`
            - :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`
            - :attr:`Sampler.pars_init_vec <DNNLikelihood.Sampler.pars_init_vec>`
            - :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
            - :attr:`Sampler.pars_labels_auto <DNNLikelihood.Sampler.pars_labels_auto>`
            - :attr:`Sampler.pars_pos_nuis <DNNLikelihood.Sampler.pars_pos_nuis>`
            - :attr:`Sampler.pars_pos_poi <DNNLikelihood.Sampler.pars_pos_poi>`

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Sampler.output_h5_file <DNNLikelihood.Sampler.output_h5_file>`
            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if type(overwrite) == bool:
            output_h5_file = self.output_h5_file
            if not overwrite:
                utils.check_rename_file(output_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_h5_file = utils.generate_dump_file_name(self.output_h5_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__, ["backend", 
                                                          "backend_file",
                                                          "input_file", 
                                                          "input_folder", 
                                                          "input_h5_file", 
                                                          "input_log_file", 
                                                          "log", 
                                                          "logpdf",
                                                          "logpdf_args", 
                                                          "moves", 
                                                          "ndims",
                                                          "nsteps_available",
                                                          "output_figures_base_file_name",
                                                          "output_figures_base_file_path",
                                                          "output_figures_folder",
                                                          "output_folder",
                                                          "output_h5_file",
                                                          "output_log_file",
                                                          "output_predictions_json_file",
                                                          "pars_init_vec", 
                                                          "pars_labels", 
                                                          "pars_labels_auto",
                                                          "pars_pos_nuis", 
                                                          "pars_pos_poi", 
                                                          "sampler", 
                                                          "verbose"])
        dd.io.save(output_h5_file, dictionary)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(output_h5_file)[-1]}

        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nSampler h5 file\n\t", output_h5_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nSampler h5 file\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nSampler h5 file dump\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        self.save_predictions_json(timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub)
        self.save_log(timestamp=timestamp, overwrite=overwrite, verbose=verbose)
    
    def reset_predictions(self, 
                          delete_figures=False, 
                          verbose=None):
        """
        Re-initializes the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary to

         .. code-block:: python

            predictions = {"gelman_rubin": {},
                           "Figures": figs}

        Where ``figs`` may be either an empty dictionary or the present value of the corresponding one,
        depending on the value of the ``delete_figures`` argument.

        - **Arguments**

            - **delete_figures**
            
                If ``True`` all files in the :attr:`Lik.output_figures_folder <DNNLikelihood.Lik.output_figures_folder>` 
                folder are deleted and the ``"Figures"`` item is reset to an empty dictionary.
                    
                    - **type**: ``bool``
                    - **default**: ``True`` 
            
            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if delete_figures:
            utils.check_delete_all_files_in_path(self.output_figures_folder)
            figs = {}
            print(header_string,"\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n")
        else:
            figs = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            print(header_string,"\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n")
        self.predictions = {"gelman_rubin": {},
                            "Figures": figs}

    ##### Functions from the emcee documentation (with some modifications) #####

    def autocorr_func_1d(self, x, norm=True, verbose=None):
        """
        Function adapted from the |emcee_tutorial_autocorr_link| (see the link for documentation)
        and used to compute and plot autocorrelation time estimates.
        """
        verbose, _ = self.set_verbosity(verbose)
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError( "invalid dimensions for 1D autocorrelation function")
        if len(np.unique(x)) == 1:
            print(header_string,"\nChain does not change in "+str(len(x))+" steps. Autocorrelation for this chain may return nan.\n",show=verbose)
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
    def auto_window(self, taus, c):
        """
        Function adapted from the |emcee_tutorial_autocorr_link| (see the link for documentation)
        and used to compute and plot autocorrelation time estimates.
        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010(self, y, c=5.0, verbose=None):
        """
        Function adapted from the |emcee_tutorial_autocorr_link| (see the link for documentation).
        Estimate of the integrated autocorrelation time based on ref. :cite:`GoodmanWeare:2010qj`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        f = self.autocorr_func_1d(np.mean(y, axis=0), verbose=verbose_sub)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_new(self,y, c=5.0, verbose=None):
        """
        Function adapted from the |emcee_tutorial_autocorr_link| (see the link for documentation).
        Estimate of the integrated autocorrelation time based on ref. :cite:`fardal:2017`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        f = np.zeros(y.shape[1])
        counter=0
        for yy in y:
            fp = self.autocorr_func_1d(yy, verbose=verbose_sub)
            if np.isnan(np.sum(fp)):
                print(header_string,"\nChain",counter,"returned nan. Values changed to 0 to proceed.\n",show=verbose)
                fp = np.full(len(fp),0)
            f += fp
            counter += 1
        f /= len(y)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_ml(self, y, thin=1, c=5.0, bound=5.0,verbose=True):
        """
        Function adapted from the |emcee_tutorial_autocorr_link| (see the link for documentation).
        Estimate of the integrated autocorrelation time obtained by fitting an autoregressive model
        (2nd order ARMA model) :cite:`DanForeman-Mackey:2017` using the |celerite_link| package.
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

    def compute_gelman_rubin(self, 
                     pars=0, 
                     nsteps="all", 
                     timestamp=None, 
                     verbose=None):
        """
        Given a parameter (or list of parameters) ``pars`` and a number of ``nsteps``, the method computes 
        the Gelman and Rubin ratio :cite:`Gelman:1992zz` and related quantities for monitoring convergence and saves them 
        in the :attr:`Sampler.predictions["gelman_rubin"] <DNNLikelihood.Sampler.predictions>` dictionary under the item
        ``<timestamp>``
        The formula for :math:`R_{c}` implements the correction due to Brooks and Gelman :cite:`Brooks_1998` 
        and is implemented here as
        
        .. math::
            R_{c} = \\sqrt{\\frac{\\hat{d}+3}{\\hat{d}+1}\\frac{\\hat{V}}{W}}.

        See the original papers for the notation.
        
        In order to be able to monitor not only the :math:`R_{c}` ratio, but also the values of :math:`\\hat{V}` 
        and :math:`W` independently, the method also computes these quantities. Usually a reasonable convergence 
        condition is :math:`R_{c}<1.1`, together with stability of both :math:`\\hat{V}` and :math:`W` :cite:`Brooks_1998`.

        - **Arguments**

            - **pars**

                Parameter or list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: 0

            - **nsteps**

                Array or list of number of steps at which the metrics are computed. If it is an integer it is automatically converted
                in a |Numpy_link| array with a single entry. It it is ``"all"`` it is set to a list with the only entry
                :attr:`Sampler.nsteps_available <DNNLikelihood.Sampler.nsteps_available>`. 

                    - **type**: ``str`` or ``int`` or ``list`` or ``numpy.ndarray``
                    - **allowed str**: ``all``
                    - **shape of list**: ``[ ]``
                    - **shape of array**: ``(len(nsteps),)``
                    - **default**: ``"all"``

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        
        - **Returns**

            An array constructed concatenating lists of the type ``[par, nsteps, Rc, Vhat, W]`` for each parameter
            and each choice of nsteps.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(len(pars)*len(nsteps),5)``

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nComputing Gelman-Rubin convergence metric", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        utils.check_set_dict_keys(self.predictions["gelman_rubin"],
                                  [timestamp],
                                  [{}], verbose=False)
        utils.check_set_dict_keys(self.predictions["gelman_rubin"][timestamp],
                                  ["evaluation_times"],
                                  [[]], verbose=False)
        res = []
        pars = np.array([pars]).flatten()
        for par in pars:
            start_sub = timer()
            if nsteps == "all":
                nsteps = np.array([self.nsteps_available])
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
                end_sub = timer()
                self.predictions["gelman_rubin"][timestamp]["evaluation_times"].append(end_sub-start_sub)
        res = np.array(res)
        self.predictions["gelman_rubin"][timestamp]["pars"] = pars
        self.predictions["gelman_rubin"][timestamp]["pars_vals"] = res[:,0]
        self.predictions["gelman_rubin"][timestamp]["nsteps"] = res[:,1]
        self.predictions["gelman_rubin"][timestamp]["Rc"] = res[:,2]
        self.predictions["gelman_rubin"][timestamp]["Vhat"] = res[:,3]
        self.predictions["gelman_rubin"][timestamp]["W"] = res[:,4]
        self.predictions["gelman_rubin"][timestamp]["evaluation_times"] = np.array(self.predictions["gelman_rubin"][timestamp]["evaluation_times"])
        self.predictions["gelman_rubin"][timestamp]["global_evaluation_time"] = np.sum(self.predictions["gelman_rubin"][timestamp]["evaluation_times"])
        end = timer()
        self.log[timestamp] = {"action": "computed Gelman-Rubin", 
                               "pars": pars, 
                               "nsteps": nsteps}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nGelman-Rubin statistics for parameters", pars,"computed in",str(end-start),"s.\n",show=verbose)
        #return np.array(res)

    def update_figures(self,figure_file=None,timestamp=None,overwrite=False,verbose=None):
        """
        Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
        When ``overwrite=False`` it calls the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function and, if 
        ``figure_file`` already existed in the :attr:`Sampler.predictions <DNNLikelihood.Sampler.predictions>` dictionary, then it
        updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
        (that is the key of the :attr:`Sampler.predictions["Figures"] <DNNLikelihood.Sampler.predictions>` dictionary).
        When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_name <DNNLikelihood.utils.generate_dump_file_name>` function
        to generate the dump file name.
        It returns the new figure_file.

        - **Arguments**

            - **figure_file**

                Figure file path. If the figure already exists in the 
                :meth:`Sampler.predictions <DNNLikelihood.Sampler.predictions>` dictionary, then its name is updated with the corresponding timestamp.

            - **overwrite**

                The method updates file names and :attr:`Sampler.predictions <DNNLikelihood.Sampler.predictions>` dictionary only if
                ``overwrite=False``. If ``overwrite="dump"`` the method generates and returns the dump file path. 
                If ``overwrite=True`` the method just returns ``figure_file``.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        
        - **Returns**

            - **new_figure_file**
                
                String identical to the input string ``figure_file`` unless ``verbose="dump"``.

        - **Creates/updates files**

            - Updates ``figure_file`` file name.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print("Checking and updating figures dictionary",show=verbose)
        if figure_file is None:
            raise Exception("figure_file input argument of update_figures method needs to be specified while it is None.")
        else:
            new_figure_file = figure_file
            if type(overwrite) == bool:
                if not overwrite:
                    # search figure
                    timestamp=None
                    for k, v in self.predictions["Figures"].items():
                        if figure_file in v:
                            timestamp = k
                    old_figure_file = utils.check_rename_file(path.join(self.output_figures_folder,figure_file),timestamp=timestamp,return_value="file_name",verbose=verbose_sub)
                    if timestamp is not None:
                        self.predictions["Figures"][timestamp] = [f.replace(figure_file,old_figure_file) for f in v] 
            elif overwrite == "dump":
                new_figure_file = utils.generate_dump_file_name(figure_file, timestamp=timestamp)
        return new_figure_file

    def plot_gelman_rubin(self, 
                          pars=0, 
                          npoints=5, 
                          pars_labels="original", 
                          show_plot=False, 
                          timestamp=None,
                          overwrite=True, 
                          verbose=None):
        """
        Plots the evolution with the number of steps of the convergence metrics :math:`R_{c}`, 
        :math:`\\sqrt{\\hat{V}}`, and :math:`\\sqrt{W}` computed by the method 
        :meth:`Sampler.compute_gelman_rubin <DNNLikelihood.Sampler.compute_gelman_rubin>` for parameter (or list of parameters) ``pars``. 
        The plots are produced by computing the quantities in ``npoints`` equally spaced (in base-10 log scale) points  
        between one and the total number of available steps. 

        - **Arguments**

            - **pars**

                Parameter or list of parameters 
                for which the plots are produced.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **npoints**

                Number of points in which the convergence metrics are computed to produce the plot.
                The points are taken equally spaced in base-10 log scale.

                    - **type**: ``int``
                    - **default**: ``5``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting Gelman-Rubin convergence metric", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        for par in pars:
            start = timer()
            idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps_available))))])
            idx = np.unique(idx[idx <= self.nsteps_available])
            idx = utils.get_spaced_elements(idx, numElems=npoints+1)
            idx = idx[1:]
            try:
                self.predictions["gelman_rubin"][timestamp]
            except:
                self.compute_gelman_rubin(par, nsteps=idx, timestamp=timestamp)
            gr_pars = self.predictions["gelman_rubin"][timestamp]["pars_vals"]
            gr_nsteps = self.predictions["gelman_rubin"][timestamp]["nsteps"]
            gr_Rc = self.predictions["gelman_rubin"][timestamp]["Rc"]
            gr_What = self.predictions["gelman_rubin"][timestamp]["Vhat"]
            gr_V = self.predictions["gelman_rubin"][timestamp]["W"]
            plt.plot(gr_nsteps, gr_Rc, "-", alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\hat{R}_{c}(%s)$" % (pars_labels[par].replace("$", "")))
            plt.xscale("log")
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_GR_Rc_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print("\n"+header_string+"\nFigure file\n\t",r"%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
            start = timer()
            plt.plot(gr_nsteps, np.sqrt(gr_What), "-", alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{\hat{V}}(%s)$"% (pars_labels[par].replace("$", "")))
            plt.xscale("log")
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_GR_sqrtVhat_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print(header_string+"\nFigure file\n\t",r"%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
            start = timer()
            plt.plot(gr_nsteps, np.sqrt(gr_V), "-", alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{W}(%s)$"% (pars_labels[par].replace("$", "")))
            plt.xscale("log")
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_GR_sqrtW_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print(header_string+"\nFigure file\n\t",r"%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
        self.save_log(overwrite=True, verbose=verbose_sub)
            
    def plot_dist(self, 
                  pars=0, 
                  pars_labels="original", 
                  show_plot=False, 
                  timestamp=None,
                  overwrite=True, 
                  verbose=None):
        """
        Plots the 1D distribution of parameter (or list of parameters) ``pars``.

        - **Arguments**

            - **pars**

                Parameter or list of parameters 
                for which the plots are produced.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting distribution of parameters", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        for par in pars:
            start = timer()
            chain = self.sampler.get_chain()[:, :, par].T
            counts, bins = np.histogram(chain.flatten(), 100)
            integral = counts.sum()
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.step(bins[:-1], counts/integral, where="post")
            plt.xlabel(r"$%s$" % (pars_labels[par].replace("$", "")))
            plt.ylabel(r"$p(%s)$" % (pars_labels[par].replace("$", "")))
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_distr_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print("\n"+header_string+"\nFigure file\n\t",r"%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
        self.save_log(overwrite=True, verbose=verbose_sub)

    def plot_autocorr(self, 
                      pars=0, 
                      pars_labels="original", 
                      methods=["G&W 2010", "Fardal 2017", "DFM 2017: ML"], 
                      show_plot=False,
                      timestamp=None,
                      overwrite=True, 
                      verbose=None):
        """
        Plots the integrated autocorrelation time estimate evolution with the number of steps for parameter (or list of parameters) ``pars``.
        Three different methods are used to estimate the autocorrelation time: "G&W 2010" :cite:`GoodmanWeare:2010qj`, 
        "Fardal 2017" :cite:`fardal:2017`, and "DFM 2017: ML" :cite:`DanForeman-Mackey:2017`, described in details
        in the |emcee_tutorial_autocorr_link| and corresponding to the three methods
        :meth:`Sampler.autocorr_gw2010 <DNNLikelihood.Sampler.autocorr_gw2010>`, :meth:`Sampler.autocorr_new <DNNLikelihood.Sampler.autocorr_new>`,
        and :meth:`Sampler.autocorr_ml <DNNLikelihood.Sampler.autocorr_ml>`, respectively.
        The function accepts a list of methods and by default it makes the plot including all available
        methods. Notice that to use the method "DFM 2017: ML" based on fitting an autoregressive model, the |celerite_link| package needs to be
        installed.

        - **Arguments**

            - **pars**

                Parameter or list of parameters 
                for which the plots are produced.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **methods**

                List of methods to estimate the autocorrelation time. The three availanle methods are "G&W 2010", "Fardal 2017", and "DFM 2017: ML". 
                One curve for each method will be produced.
                    
                    - **type**: ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``["G&W 2010", "Fardal 2017", "DFM 2017: ML"]``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting autocorrelation time", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        for par in pars:
            start = timer()
            chain = self.sampler.get_chain()[:, :, par].T
            # Compute the largest number of duplicated at the beginning of chains
            n_dupl = []
            for c in chain:
                n_dupl.append(utils.check_repeated_elements_at_start(c))
            n_start = max(n_dupl)+10
            if n_start > 100:
                print("There is at least one chain starting with", str(
                    n_start-10), "duplicate steps. Autocorrelation will be computer starting at", str(n_start), "steps.", show=verbose)
            else:
                n_start = 100
            N = np.exp(np.linspace(np.log(n_start), np.log(chain.shape[1]), 10)).astype(int)
            # GW10 method
            if "G&W 2010" in methods:
                gw2010 = np.empty(len(N))
            # New method
            if "Fardal 2017" in methods:
                new = np.empty(len(N))
            # Approx method (Maximum Lik)
            if "DFM 2017: ML" in methods:
                new = np.empty(len(N))
                ml = np.empty(len(N))
                ml[:] = np.nan
            for i, n in enumerate(N):
                # GW10 method
                if "G&W 2010" in methods:
                    gw2010[i] = self.autocorr_gw2010(chain[:, :n], verbose=verbose_sub)
                # New method
                if "Fardal 2017" in methods or "DFM 2017: ML" in methods:
                    new[i] = self.autocorr_new(chain[:, :n],verbose=verbose_sub)
                # Approx method (Maximum Lik)
            if "DFM 2017: ML" in methods:
                succeed = None
                bound = 5.0
                while succeed == None:
                    try:
                        for i, n in enumerate(N[1:-1]):
                            k = i + 1
                            thin = max(1, int(0.05 * new[k]))
                            ml[k] = self.autocorr_ml(chain[:, :n], thin=thin, bound=bound,verbose=verbose_sub)
                        succeed = True
                        if bound > 5.0:
                            print("Succeeded with bounds (", str(-(bound)),
                                  ",", str(bound), ").", show=verbose)
                    except:
                        print("Bounds (", str(-(bound)), ",", str(bound), ") delivered non-finite log-prior. Increasing bound to (",
                              str(-(bound+5)), ",", str(bound+5), ") and retrying.", show=verbose)
                        bound = bound+5
            # Plot the comparisons
            plt.plot(N, N / 50.0, "--k", label=r"$\tau = S/50$")
            #plt.plot(N, N / 100.0, "--k", label=r"$\tau = S/100$")
            # GW10 method
            if "G&W 2010" in methods:
                plt.loglog(N, gw2010, "o-", label=r"G\&W 2010")
            # New method
            if "Fardal 2017" in methods:
                plt.loglog(N, new, "o-", label="Fardal 2017")
            # Approx method (Maximum Lik)
            if "DFM 2017: ML" in methods:
                plt.loglog(N, ml, "o-", label="DFM 2017: ML")
            ylim = plt.gca().get_ylim()
            plt.ylim(ylim)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\tau_{%s}$ estimates" % (pars_labels[par].replace("$", "")))
            plt.legend()
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_autocorr_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print("\n"+header_string,"\n\t%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
        self.save_log(overwrite=True, verbose=verbose_sub)

    def plot_chains(self, 
                    pars=0, 
                    n_chains=100, 
                    pars_labels="original", 
                    show_plot=False,
                    timestamp=None,
                    overwrite=True, 
                    verbose=None):
        """
        Plots the evolution of chains (walkers) with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **pars**

                Parameter or list of parameters 
                for which the plots are produced.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **n_chains**
            
                The number of chains to 
                add to the plot.

                    - **type**: ``int``
                    - **default**: ``100``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting chains evolution with number of steps", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        pars = np.array([pars]).flatten()
        pars_labels = self.__set_pars_labels(pars_labels)
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("'n_chains' larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.",show=verbose)
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        for par in pars:
            start = timer()
            chain = self.sampler.get_chain()[:, :, par]
            idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps_available))))])
            idx = np.unique(idx[idx < len(chain)])
            plt.plot(idx,chain[idx][:,rnd_chains], "-", alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$%s$" %(pars_labels[par].replace("$", "")))
            plt.xscale("log")
            plt.tight_layout()
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_chains_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": figure_file_name}
            end = timer()
            print("\n"+header_string+"\nFigure file\n\t",r"%s" % (figure_file_name), "\ncreated and saved in",str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
        self.save_log(overwrite=True, verbose=verbose_sub)

    def plot_chains_logpdf(self, 
                           n_chains=100, 
                           show_plot=False,
                           timestamp=None,
                           overwrite=True, 
                           verbose=None):
        """
        Plots the evolution of minus the logpdf values with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:`Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **n_chains**
            
                The number of chains to 
                add to the plot.

                    - **type**: ``int``
                    - **default**: ``100``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

                    - **default**: ``True``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting chains logpdf evolution with number of steps", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("'n_chains' larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.",show=verbose)
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        chain_lp = self.sampler.get_log_prob()
        idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps_available))))])
        idx = np.unique(idx[idx < len(chain_lp)])
        plt.plot(idx, -chain_lp[:, rnd_chains][idx], "-", alpha=0.8)
        plt.xlabel("number of steps, $S$")
        plt.ylabel(r"-logpdf")
        plt.xscale("log")
        plt.tight_layout()
        figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_chains_logpdf.pdf",timestamp=timestamp,overwrite=overwrite) 
        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
        self.log[timestamp] = {"action": "saved figure", 
                               "file name": figure_file_name}
        end = timer()
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show=verbose)
        if show_plot:
            plt.show()
        plt.close()
        self.save_log(overwrite=True, verbose=verbose_sub)

    def get_data_object(self, nsamples="all", burnin="auto", thin="auto", dtype="float64", test_fraction=0, output_folder=None, verbose=None):
        """
        Returns a :mod:`Data <data>` object with ``nsamples`` samples by taking chains and logpdf values, discarding 
        ``burnin`` steps, thinning every ``thin``, deleting duplicates, and converting to dtype ``dtype``. When ``nsamples="all"`` (default) all samples 
        available for the given choice of ``burnin`` and ``thin`` are included to the :class:`DNNLikelihood.Data` object, 
        otherwise only the last ``nsamples`` are included. If ``nsamples`` is more than the available number of samples, then all available 
        samples are included and a warning message is printed. 
        The ``burnin`` and ``thin`` options can be set to ``"auto"`` to optimize them and/or get an idea of the optimal values.
        For details on the ``"auto"`` option see the arguments documentation below. When chains are very large the ``"auto"`` mode
        can take a long time. 
        It prints a warning is there are non finite values of logpdf (e.g. ``np.nan`` or ``np.inf``).

        The method also allows one to pass to the :class:`DNNLikelihood.Data` a value for ``test_fraction``, which already
        splits data into ``train`` (sample from which training and valudation data are extracted) and ``test`` (sample only
        used for final test) sets. See :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` for more details.

        Finally, based on the value of ``save``, the generated :class:`DNNLikelihood.Data` object

        - **Arguments**

            - **nsamples**
            
                Number of samples to include in the 
                :mod:`Data <data>` object.
                    
                    - **type**: ``int`` or ``str``
                    - **allowed string**: ``"all"``
                    - **default**: ``"all"``

            - **burnin**
            
                Number of samples to skip as burnin. If set to "auto", the autocorrelation time of parameters is computed
                using the |emcee_autocorr_time_link| method of the sampler :attr:`Sampler.sampler <DNNLikelihood.Sampler.sampler>`
                and the burnin is set to the minimum between 5 times the autocorrelation time and half the total number of available steps.
                If "auto" option fails it raises an error asking the user to manually specify ``burnin``.
                Notice that for very large chains the calculation of the autocorrelation time can take a long time.
                    
                    - **type**: ``int`` or ``str``
                    - **default**: ``0``
                    - **allowed string**: ``"auto"``

            - **thin**
            
                If larger than ``1`` then one sampler every thin is taken. If set to ``auto`` thin is optimized by taking the largest possible
                thin compatible with ``burnin`` and ``nsamples``.
                If "auto" option fails it raises an error asking the user to manually specify ``thin``.
                Notice that for very large chains the optimization of ``thin`` can take a long time.
                    
                    - **type**: ``int`` or ``str``
                    - **default**: ``1``
                    - **allowed string**: ``"auto"``

            - **dtype**
            
                dtype of the data included in the 
                :mod:`Data <data>` object.
                    
                    - **type**: ``str``
                    - **default**: ``"float64"``

            - **test_fraction**
            
                If specified, in the :mod:`Data <data>` object
                data are already split into train (and valudation) and test sets.
                See the :mod:`Data <data>` object documentation.
                    
                    - **type**: ``float`` in the range ``(0,1)``
                    - **default**: ``0``

            - **output_folder**
            
                If specified is passed as input to the :class:`Data <DNNLikelihood.Data>`, otherwise
                the :attr:`Sampler.output_folder <DNNLikelihood.Sampler.output_folder>` is passed
                and the :mod:`Data <data>` object is saved in the same folder as the 
                :class:`Sampler <DNNLikelihood.Sampler>` object.
                    
                    - **type**: ``str``
                    - **default**: ``None``
            
            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            :mod:`Data <data>` object.

        - **Creates files**

            - :attr:`Data.output_h5_file <DNNLikelihood.Data.output_h5_file>`
            - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`

        - **Updates file**

            - :attr:`Sampler.output_log_file <DNNLikelihood.Sampler.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nCreating 'Data' object\n", show=verbose)
        if output_folder == None:
            output_folder = self.output_folder
        start = timer()
        ### Compute available samples
        if burnin == "auto":
            print(header_string)
            try:
                print("Estimating autocorrelation time to optimize burnin. For very large chains this could take a while.", show=verbose)
                autocorr_max = int(np.max(self.sampler.get_autocorr_time()))
            except:
                raise Exception("Could not automatically determine optimal 'burnin'. You must manually specify the 'burnin' input.")
            burnin=int(np.min([5*autocorr_max,self.nsteps_available/2]))
            print("Maximum estimated autocorrelation time of all parameters is:",autocorr_max,".",show=verbose)
            print("Burning automatically set to:", burnin, ".\n", show=verbose)
        else:
            print(header_string,"\nWarning: When requiring an unbiased data sample please check that the required burnin is compatible with MCMC convergence.\n", show=verbose)
        if thin == "auto":
            print(header_string)
            try:
                autocorr_max
            except:
                autocorr_max = None
            try:
                print("Estimating optimal 'thin'. For very large chains this could take a while.", show=verbose)
                thin=int((self.nsteps_available-burnin)*self.nwalkers/nsamples)
                while len(np.unique(self.sampler.get_log_prob(discard=burnin,thin=thin, flat=True),return_index=False)) < nsamples and thin>1:
                    thin=thin-1
                if autocorr_max != None:
                    if thin < autocorr_max:
                        print("The required number of samples does not allow a thin value larger than the estimated autocorrelation time.\nThin hase been set to the maximum possible value compatible with 'burnin':",thin,".\n",show=verbose)
                    else:
                        print("Thin automatically set to:",thin,".\n",show=verbose)
                else:
                    print("Thin automatically set to:",thin,".\n",show=verbose)
            except:
                raise Exception("Could not automatically determine optimal 'thin'. You must manually specify the 'thin' input.")
        logpdf_values=self.sampler.get_log_prob(discard = burnin, thin = thin, flat = True)
        allsamples=self.sampler.get_chain(discard = burnin, thin = thin, flat = True)
        unique_indices = np.sort(np.unique(logpdf_values,return_index=True)[1])
        available_samples = len(unique_indices)
        if nsamples == "all":
            pass
        elif nsamples <= available_samples:
            unique_indices = unique_indices[-nsamples:]
        else:
            print("There are less unique samples (", available_samples, ") than requested samples (", nsamples, "). Returning all available samples.\nYou may try to reduce burnin and/or thin to get more samples.\n", show=verbose)
        logpdf_values = logpdf_values[unique_indices]
        allsamples = allsamples[unique_indices]
        if np.count_nonzero(np.isfinite(logpdf_values)) < len(logpdf_values):
            print("There are non-numeric logpdf values.\n",show=verbose)
        end = timer()
        print(str(len(allsamples)), "unique samples generated in", end-start, "s.",show=verbose)
        data_sample_timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        ds = Data(name=self.name.replace("_sampler", "_data"),
                  data_X=allsamples,
                  data_Y=logpdf_values,
                  dtype=dtype,
                  pars_central=self.pars_central,
                  pars_pos_poi=self.pars_pos_poi,
                  pars_pos_nuis=self.pars_pos_nuis,
                  pars_labels=self.pars_labels,
                  pars_bounds = self.pars_bounds,
                  test_fraction=test_fraction,
                  load_on_RAM=False,
                  output_folder=output_folder,
                  input_file=None,
                  verbose=self.verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "created data object", 
                               "files names": [path.split(ds.output_log_file)[-1],
                                               path.split(ds.output_predictions_json_file)[-1],
                                               path.split(ds.output_h5_file)[-1],
                                               path.split(ds.output_samples_h5_file)[-1],
                                               path.split(self.output_log_file)[-1]]}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string, "\nData object created and saved in", str(end-start), "s.\n", show=verbose)
        return ds
