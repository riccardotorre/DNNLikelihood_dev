__all__ = ["DnnLik"]

import builtins
import codecs
import json
import multiprocessing
import pickle
import re
import time
from datetime import datetime
from decimal import Decimal
from os import path, remove, sep, stat
from timeit import default_timer as timer

import deepdish as dd
import h5py
import keras2onnx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnx
import scipy
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, losses, metrics, optimizers
from tensorflow.keras.layers import (AlphaDropout, BatchNormalization, Dense,
                                     Dropout, InputLayer)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

from . import inference, utils
from .corner import corner, extend_corner_range, get_1d_hist
from .data import Data
from .likelihood import Lik
from .resources import Resources
from .show_prints import print

try:
    from livelossplot import PlotLossesKerasTF as PlotLossesKeras
except:
    print("No module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'liveflossplot'.")

sns.set()
kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

class DnnLik(Resources): #show_prints.Verbosity inherited from resources.Resources
    """
    This class contains the ``DnnLik`` object, that is the core object of the DNNLikelihood package.
    It represents the DNNLikelihood through a |tf_keras_model_link| object. The class allows one to train and evaluate the performance 
    of the DNNLikelihood, to produce several plots, make inference with the DNNLikelihood from both a Bayesian and a frequentist
    perspective, and to export the trained model in the standard |onnx_link| format.
    """
    def __init__(self,
                 name=None,
                 data=None,
                 input_data_file=None,
                 input_likelihood_file=None,
                 load_on_RAM=False,
                 seed=None,
                 dtype=None,
                 same_data=True,
                 model_data_inputs=None,
                 model_define_inputs=None,
                 model_optimizer_inputs=None,
                 model_compile_inputs=None,
                 model_callbacks_inputs=None,
                 model_train_inputs=None,
                 resources_inputs=None,
                 output_folder=None,
                 ensemble_name=None,
                 input_summary_json_file=None,
                 verbose=True
                 ):
        """
        The :class:`DnnLik <DNNLikelihood.DnnLik>` object can be initialized in two different ways, 
        depending on the value of the :argument:`input_summary_json_file` argument.

        - :argument:`input_summary_json_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no :argument:`name` is given, 
            then one is created. If the input argument :argument:`ensemble_name` is not ``None`` the object is assumed to be
            part of a :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object and the corresponding 
            attribute :attr:`DnnLik.ensemble_folder <DNNLikelihood.DnnLik.ensemble_folder>` is set to the
            partent directory of :argument:`output_folder`.
            The full object is saved through the 
            :meth:`DnnLik.save <DNNLikelihood.DnnLik.save>` method. 
        
        - :argument:`input_summary_json_file` is not ``None``

            The object is reconstructed from the input files. First the log and summary_json files are loaded through 
            the private method
            :meth:`DnnLik.__load_summary_json_and_log <DNNLikelihood.DnnLik._Lik__load_summary_json_and_log>`,
            then, if the input arguments :argument:`dtype`, :argument:`seed`, and :argument:`output_folder` are provided,
            the corresponding attributes are set to their values (overwriting the values imported from files), and finally 
            all other attributes are loaded through the private methods:

                - :meth:`DnnLik.__load_data_indices <DNNLikelihood.DnnLik._Lik__load_data_indices>`
                - :meth:`DnnLik.__load_history_and_log <DNNLikelihood.DnnLik._Lik__load_history>`
                - :meth:`DnnLik.__load_model <DNNLikelihood.DnnLik._Lik__load_model>`
                - :meth:`DnnLik.__load_predictions <DNNLikelihood.DnnLik._Lik__load_predictions>`
                - :meth:`DnnLik.__load_scalers_and_log <DNNLikelihood.DnnLik._Lik__load_scalers>`

            Only the json summary and log files are saved (overwriting the old ones) through the methods:
                
                - :meth:`DnnLik.save_summary_json <DNNLikelihood.DnnLik.save_summary_json>` 
                - :meth:`DnnLik.save_log <DNNLikelihood.DnnLik.save_log>` 
    
        Resources, data, seed, dtype and |tf_link| objects are set in the same way, independently of the value of
        :argument:`input_summary_json_file`, through the private methods:
            
            - :meth:`DnnLik.__set_data <DNNLikelihood.DnnLik._Lik__set_data>`
            - :meth:`DnnLik.__set_dtype <DNNLikelihood.DnnLik._Lik__set_dtype>`
            - :meth:`DnnLik.__set_resources <DNNLikelihood.DnnLik._Lik__set_resources>`
            - :meth:`DnnLik.__set_seed <DNNLikelihood.DnnLik._Lik__set_seed>`            
            - :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._Lik__set_tf_objects>`

        - **Arguments**

            See class :ref:`Arguments documentation <DnnLik_arguments>`.

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
            - :attr:`DnnLik.output_summary_json_file <DNNLikelihood.DnnLik.output_summary_json_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        #### Set input files
        self.input_summary_json_file = input_summary_json_file
        self.input_data_file = input_data_file
        self.input_likelihood_file = input_likelihood_file
        self.__check_define_input_files()  
        ############ Check wheather to create a new DNNLik object from inputs or from files
        if self.input_files_base_name == None:
            ############ Initialize input parameters from arguments
            #### Set main inputs
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.data = data
            self.load_on_RAM = load_on_RAM
            self.seed = seed
            self.dtype = dtype
            self.same_data = same_data
            self.__set_likelihood(verbose=verbose_sub)
            self.__model_data_inputs = model_data_inputs
            self.__check_define_model_data_inputs()
            self.__model_define_inputs = model_define_inputs
            self.__check_define_model_define_inputs()
            self.__model_optimizer_inputs = model_optimizer_inputs
            self.__model_compile_inputs = model_compile_inputs
            self.__check_define_model_compile_inputs(verbose=verbose_sub)
            self.__model_callbacks_inputs = model_callbacks_inputs
            self.__model_train_inputs = model_train_inputs
            self.__check_define_model_train_inputs()
            self.npoints_train, self.npoints_val, self.npoints_test = self.__model_data_inputs["npoints"]
            #### Set output folder and files
            self.output_folder = output_folder
            self.__check_define_output_files()
            #### Set ensemble attributes if the DNNLikelihood is part of an ensemble
            self.ensemble_name = ensemble_name
            self.__check_define_ensemble_folder(verbose=verbose_sub)
            ### Set model hyperparameters parameters
            self.__set_model_hyperparameters()
        else:
            ############ Initialize input parameters from file
            #### Load summary_log dictionary
            print("When providing DNNLik input folder all arguments but data, input_likelihood_file, load_on_RAM and dtype are ignored and the object is constructed from saved data.",show=verbose)
            self.__load_summary_json_and_log(verbose=verbose_sub)
            if input_likelihood_file != None:
                self.input_likelihood_file = path.abspath(input_likelihood_file)
            self.__set_likelihood(verbose=verbose_sub)
            self.data = None
            #### Set main inputs and DataSample
            self.load_on_RAM = load_on_RAM
            if dtype != None:
                self.dtype = dtype
            if seed != None:
                self.seed = seed
            ### Set name, folders and files names
            if output_folder != None:
                self.output_folder = path.abspath(output_folder)
                self.__check_define_output_files()
        #### Set resources (__resources_inputs is None for a standalone DNNLikelihood and is passed only if the DNNLikelihood
        #### is part of an ensemble)
        self.__resources_inputs = resources_inputs
        self.__set_resources(verbose=verbose_sub)
        #### Set additional inputs
        self.__set_seed()
        self.__set_dtype()
        self.__set_data(verbose=verbose_sub) # also sets self.ndims
        self.__set_tf_objects(verbose=verbose_sub) # optimizer, loss, metrics, callbacks
        ### Initialize model, history,scalers, data indices, and predictions
        if self.input_files_base_name != None:
            self.__load_data_indices(verbose=verbose_sub)
            self.__load_history(verbose=verbose_sub)
            self.__load_model(verbose=verbose_sub)
            self.__load_predictions(verbose=verbose_sub)
            self.__load_scalers(verbose=verbose_sub)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
        else:
            self.epochs_available = 0
            self.idx_train, self.idx_val, self.idx_test = [np.array([], dtype="int"),np.array([], dtype="int"),np.array([], dtype="int")]
            self.scalerX, self.scalerY = [None,None]
            self.model = None
            self.history = {}
            self.predictions = {"Model_evaluation": {}, 
                                "Bayesian_inference": {}, 
                                "Frequentist_inference": {},
                                "Figures": {}}
        self.X_train, self.Y_train, self.W_train = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype),np.array([], dtype=self.dtype)]
        self.X_val, self.Y_val = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype)]
        self.X_test, self.Y_test = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype)]
        ### Save object
        if self.input_files_base_name == None:
            self.save_summary_json(overwrite=False, verbose=verbose_sub)
            self.save_log(overwrite=False, verbose=verbose_sub)
            #self.save(overwrite=False, verbose=verbose_sub)
            #self.save_log(overwrite=False, verbose=verbose_sub)
        else:
            self.save_summary_json(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    def __set_resources(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one to set resources.
        If :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik.__resources_inputs` is ``None``, it 
        calls the methods 
        :meth:`DnnLik.get_available_cpu <DNNLikelihood.DnnLik.get_available_cpu` and
        :meth:`DnnLik.set_gpus <DNNLikelihood.DnnLik.set_gpus` inherited from the
        :class:`Verbosity <DNNLikelihood.Verbosity>` class, otherwise it sets resources from input arguments.
        The latter method is needed, when the object is a member of an esemble, to pass available resources from the parent
        :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object.
        
        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.__resources_inputs is None:
            #self.get_available_gpus(verbose=False)
            self.get_available_cpu(verbose=verbose_sub)
            self.set_gpus(gpus_list="all", verbose=verbose_sub)
        else:
            self.available_gpus = self.__resources_inputs["available_gpus"]
            self.available_cpu = self.__resources_inputs["available_cpu"]
            self.active_gpus = self.__resources_inputs["active_gpus"]
            self.gpu_mode = self.__resources_inputs["gpu_mode"]

    def __set_likelihood(self,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_likelihood_file is not None:
            try:
                self.likelihood = Lik(input_file=self.input_likelihood_file,verbose=verbose_sub)
                timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
                self.log[timestamp] = {"action": "loaded likelihood object",
                                       "file name": path.split(self.input_likelihood_file)[-1],
                                       "file path": self.input_likelihood_file}
                print("The Likelihood object stored in",self.input_likelihood_file,"has been imported.",show=verbose)
            except:
                self.likelihood = None
                print("The Likelihood object stored in",self.input_likelihood_file,"could not be imported.",show=verbose)
        else:
            print("No Likelihood object has been specified.",show=verbose)
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings


    def __check_define_input_files(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the attributes corresponding to input files:

            - :attr:`DnnLik.input_files_base_name <DNNLikelihood.DnnLik.input_files_base_name>`
            - :attr:`DnnLik.input_history_json_file <DNNLikelihood.DnnLik.input_history_json_file>`
            - :attr:`DnnLik.input_idx_h5_file <DNNLikelihood.DnnLik.input_idx_h5_file>`
            - :attr:`DnnLik.input_log_file <DNNLikelihood.DnnLik.input_log_file>`
            - :attr:`DnnLik.input_predictions_h5_file <DNNLikelihood.DnnLik.input_predictions_h5_file>`
            - :attr:`DnnLik.input_scalers_pickle_file <DNNLikelihood.DnnLik.input_scalers_pickle_file>`
            - :attr:`DnnLik.input_tf_model_h5_file <DNNLikelihood.DnnLik.input_tf_model_h5_file>`

        depending on the value of the 
        :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` attribute.
        It also sets the attribute
        :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>` if the object has
        been initialized directly from a :mod:`Data <data>` object.
        """
        if self.input_likelihood_file is not None:
            self.input_likelihood_file = path.abspath(self.input_likelihood_file)
        if self.input_summary_json_file == None:
            self.input_files_base_name = self.input_summary_json_file
            self.input_history_json_file = self.input_files_base_name
            self.input_idx_h5_file = self.input_files_base_name
            self.input_log_file = self.input_summary_json_file
            self.input_predictions_h5_file = self.input_files_base_name
            self.input_scalers_pickle_file = self.input_files_base_name
            self.input_tf_model_h5_file = self.input_files_base_name
        else:
            self.input_files_base_name = path.abspath(path.splitext(self.input_summary_json_file)[0].replace("_summary",""))
            self.input_history_json_file = self.input_files_base_name+"_history.json"
            self.input_idx_h5_file = self.input_files_base_name+"_idx.h5"
            self.input_log_file = self.input_files_base_name+".log"
            self.input_predictions_h5_file = self.input_files_base_name+"_predictions.h5"
            self.input_scalers_pickle_file = self.input_files_base_name+"_scalers.pickle"
            self.input_tf_model_h5_file = self.input_files_base_name+"_model.h5"
        if self.input_data_file != None:
            self.input_data_file = path.abspath(path.splitext(self.input_data_file)[0])

    def __check_define_output_files(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the attributes corresponding to output folders

            - :attr:`DnnLik.output_figures_folder <DNNLikelihood.DnnLik.output_figures_folder>`
            - :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>`

        and output files

            - :attr:`DnnLik.output_figures_base_file <DNNLikelihood.DnnLik.output_figures_base_file>`
            - :attr:`DnnLik.output_files_base_name <DNNLikelihood.DnnLik.output_files_base_name>`
            - :attr:`DnnLik.output_history_json_file <DNNLikelihood.DnnLik.output_history_json_file>`
            - :attr:`DnnLik.output_idx_h5_file <DNNLikelihood.DnnLik.output_idx_h5_file>`
            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
            - :attr:`DnnLik.output_predictions_h5_file <DNNLikelihood.DnnLik.output_predictions_h5_file>`
            - :attr:`DnnLik.output_scalers_pickle_file <DNNLikelihood.DnnLik.output_scalers_pickle_file>`
            - :attr:`DnnLik.output_summary_json_file <DNNLikelihood.DnnLik.output_summary_json_file>`
            - :attr:`DnnLik.output_tf_model_graph_pdf_file <DNNLikelihood.DnnLik.output_tf_model_graph_pdf_file>`
            - :attr:`DnnLik.output_tf_model_h5_file <DNNLikelihood.DnnLik.output_tf_model_h5_file>`
            - :attr:`DnnLik.output_tf_model_json_file <DNNLikelihood.DnnLik.output_tf_model_json_file>`
            - :attr:`DnnLik.output_tf_model_onnx_file <DNNLikelihood.DnnLik.output_tf_model_onnx_file>`
        
        depending on the value of the 
        :attr:`DnnLik.input_file <DNNLikelihood.DnnLik.input_files_base_name>` and
        :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` attributes.
        It also initializes (to ``None``) the attributes:

            - :attr:`DnnLik.output_checkpoints_files <DNNLikelihood.DnnLik.output_checkpoints_files>`
            - :attr:`DnnLik.output_checkpoints_folder <DNNLikelihood.DnnLik.output_checkpoints_folder>`
            - :attr:`DnnLik.output_figure_plot_losses_keras_file <DNNLikelihood.DnnLik.output_figure_plot_losses_keras_file>`
            - :attr:`DnnLik.output_tensorboard_log_dir <DNNLikelihood.DnnLik.output_tensorboard_log_dir>`
        
        and creates the folders
        :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>`
        and 
        :attr:`DnnLik.output_figures_folder <DNNLikelihood.DnnLik.output_figures_folder>`
        if they do not exist.
        """
        if self.output_folder == None:
            self.output_folder = ""
        self.output_folder = path.abspath(self.output_folder)
        self.output_figures_folder = path.join(self.output_folder, "figures")
        self.output_figures_base_file = path.join(self.output_figures_folder, self.name+"_figure")
        self.output_files_base_name = path.join(self.output_folder, self.name)
        self.output_history_json_file = self.output_files_base_name+"_history.json"
        self.output_idx_h5_file = self.output_files_base_name+"_idx.h5"
        self.output_log_file = self.output_files_base_name+".log"
        self.output_predictions_h5_file = self.output_files_base_name+"_predictions.h5"
        self.output_predictions_json_file = self.output_files_base_name+"_predictions.json"
        self.output_scalers_pickle_file = self.output_files_base_name+"_scalers.pickle"
        self.output_summary_json_file = self.output_files_base_name+"_summary.json"
        self.output_tf_model_graph_pdf_file = self.output_files_base_name+"_model_graph.pdf"
        self.output_tf_model_h5_file = self.output_files_base_name+"_model.h5"
        self.output_tf_model_json_file = self.output_files_base_name+"_model.json"
        self.output_tf_model_onnx_file = self.output_files_base_name+"_model.onnx"
        self.script_file = self.output_files_base_name+"_script.py"
        self.output_checkpoints_files = None
        self.output_checkpoints_folder = None
        self.output_figure_plot_losses_keras_file = None
        self.output_tensorboard_log_dir = None
        utils.check_create_folder(self.output_folder)
        utils.check_create_folder(self.output_figures_folder)
        
    def __check_define_name(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to define the :attr:`DnnLik.name <DNNLikelihood.DnnLik.name>` attribute.
        If it is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_DNNLikelihood"``.
        """
        if self.name == None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_DNNLikelihood"

    def __check_npoints(self):
        """
        Private method used by the :meth:`DnnLik.__set_data <DNNLikelihood.DnnLik._DnnLik__set_data>` one
        to check that the required number of points for train/val/test is less than the total number
        of available points in the :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>` object.
        """
        self.npoints_available = self.data.npoints
        self.npoints_train_val_available = int(
            (1-self.data.test_fraction)*self.npoints_available)
        self.npoints_test_available = int(
            self.data.test_fraction*self.npoints_available)
        required_points_train_val = self.npoints_train+self.npoints_val
        required_points_test = self.npoints_test
        if required_points_train_val > self.npoints_train_val_available:
            self.data.opened_dataset.close()
            raise Exception("npoints_train+npoints_val larger than the available number of points in data.\
                Please reduce npoints_train+npoints_val or change test_fraction in the :mod:`Data <data>` object.")
        if required_points_test > self.npoints_test_available:
            self.data.opened_dataset.close()
            raise Exception("npoints_test larger than the available number of points in data.\
                Please reduce npoints_test or change test_fraction in the :mod:`Data <data>` object.")

    def __check_define_model_data_inputs(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to check the private dictionary 
        :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`.
        It checks if the item ``"npoints"`` is correctly specified and if it is not it raises an exception. If valitadion 
        and test number of points are input
        as fractions of the training one, then it converts them to absolute number of points.
        It checks if the items ``"scalerX"``, ``"scalerY"``, and ``"weighted"`` are defined and, if they are not, it sets
        them to their default value ``False``.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self.__model_data_inputs["npoints"]
        except:
            raise Exception("model_data_inputs dictionary should contain at least a key 'npoints'.")
        if self.__model_data_inputs["npoints"][1] <= 1:
            self.__model_data_inputs["npoints"][1] = round(self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][1])
        if self.__model_data_inputs["npoints"][2] <= 1:
            self.__model_data_inputs["npoints"][2] = round(self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][2])
        utils.check_set_dict_keys(self.__model_data_inputs, ["scaleX",
                                                             "scaleY",
                                                             "weighted"],
                                                            [False,False,False],verbose=verbose_sub)

    def __check_define_model_define_inputs(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to check the private dictionary 
        :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`.
        It checks if the items ``"act_func_out_layer"``, ``"dropout_rate"``, and ``"batch_norm"`` are defined and, if they are not, 
        it sets them to their default values ``"linear"``, ``0``, and ``False``, respectively.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self.__model_define_inputs["hidden_layers"]
        except:
            raise Exception("model_define_inputs dictionary should contain at least a key 'hidden_layers'.")
        utils.check_set_dict_keys(self.__model_define_inputs, ["act_func_out_layer",
                                                               "dropout_rate",
                                                               "batch_norm"],
                                                              ["linear", 0, False], verbose=verbose_sub)

    def __check_define_model_compile_inputs(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to check the private dictionary 
        :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>`.
        It checks if the attribure exists and, if it does not, it defines it as an empty dictionary.
        It checks if the items ``"loss"`` and ``"metrics"`` are defined and, if they are not, 
        it sets them to their default values ``"mse"`` and ``["mse","mae","mape","msle"]``, respectively.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.__model_compile_inputs == None:
            self.__model_compile_inputs = {}
        utils.check_set_dict_keys(self.__model_compile_inputs, ["loss",
                                                               "metrics"],
                                                               ["mse",["mse","mae","mape","msle"]], verbose=verbose_sub)

    def __check_define_model_train_inputs(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to check the private dictionary 
        :attr:`DnnLik.__model_train_inputs <DNNLikelihood.DnnLik._DnnLik__model_train_inputs>`.
        It checks if the items ``"epochs"`` and ``"batch_size"`` are defined and, if they are not, 
        it raises an exception.
        """
        try:
            self.__model_train_inputs["epochs"]
        except:
            raise Exception("model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")
        try:
            self.__model_train_inputs["batch_size"]
        except:
            raise Exception("model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")

    def __check_define_ensemble_folder(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the
        :attr:`DnnLik.ensemble_folder <DNNLikelihood.DnnLik.ensemble_folder>` and 
        :attr:`DnnLik.standalone <DNNLikelihood.DnnLik.standalone>` attributes. If the object is a member
        of a :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object, i.e. of the
        :attr:`DnnLik.ensemble_name <DNNLikelihood.DnnLik.ensemble_name>` attribute is not ``None``,
        then the two attributes are set to the parent directory of
        :attr:`DnnLik.output_folder <DNNLikelihood.DnnLik.output_folder>` and to ``False``, respectively, otherwise
        they are set to ``None`` and ``False``, respectively.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.ensemble_name == None:
            self.ensemble_folder = None
            self.standalone = True
            print("This is a 'standalone' DNNLikelihood and does not belong to a DNNLikelihood_ensemble. The attributes 'ensemble_name' and 'ensemble_folder' are therefore been set to None.",show=verbose)
        else:
            self.enseble_folder = path.abspath(path.join(self.output_folder,".."))
            self.standalone = False

    def __set_seed(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to initialize the random state of |numpy_link| and |tf_link| to the value of 
        :attr:`DnnLik.seed <DNNLikelihood.DnnLik.seed>`.
        """
        if self.seed == None:
            self.seed = 1
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __set_dtype(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the dtype of the train/val/test data and of the internal |tf_keras_link| calculations.
        If the :attr:`DnnLik.dtype <DNNLikelihood.DnnLik.dtype>` attribute is ``None``, then it is
        set to the default value ``"float64"``.
        """
        if self.dtype == None:
            self.dtype = "float64"
        K.set_floatx(self.dtype)

    def __set_data(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to initialize the :mod:`Data <data>` object saved in the
        :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>` attribute and used to provide data to the 
        :class:`DnnLik <DNNLikelihood.DnnLik>` object.
        Data are set differently depending on the value of the attributes
        :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>` and 
        :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>`, corresponding to the two
        input class arguments: :argument:`data` and :argument:`input_data_file`, respectively. If both
        are not ``None``, then the former is ignored. If only :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`
        is not ``None``, then :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>`
        is set to the :attr:`Data.input_file <DNNLikelihood.Data.input_file>` attribute of the :mod:`Data <data>` object.
        If :attr:`DnnLik.input_data_file <DNNLikelihood.DnnLik.input_data_file>` is not ``None`` the 
        :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>` attribute is set by importing the :class:`Data <DNNLikelihood.Data>` 
        object from file.
        Once the :mod:`Data <data>` object has been set, the 
        :attr:`DnnLik.ndims <DNNLikelihood.DnnLik.ndims>` attribute == set from the same attribute of the 
        :mod:`Data <data>` object, and the two private methods
        :meth:`DnnLik.__check_npoints <DNNLikelihood.DnnLik._DnnLik__check_npoints>` and
        :meth:`DnnLik.__set_pars_info <DNNLikelihood.DnnLik._DnnLik__set_pars_info>`
        are called.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.data == None and self.input_data_file == None:
            raise Exception(
                "At least one of the arguments 'data' and 'input_data_file' should be specified.\nPlease specify one and retry.")
        elif self.data != None and self.input_data_file == None:
            self.input_data_file = self.data.input_file
            self.input_data_file = path.abspath(path.splitext(self.input_data_file)[0])
        else:
            if self.data != None:
                print("Both the arguments 'data' and 'input_data_file' have been specified. 'data' will be ignored and the :mod:`Data <data>` object will be set from 'input_data_file'.", show=verbose)
            self.data = Data(name=None,
                             data_X=None,
                             data_Y=None,
                             dtype=self.dtype,
                             pars_central=None,
                             pars_pos_poi=None,
                             pars_pos_nuis=None,
                             pars_labels=None,
                             pars_bounds=None,
                             test_fraction=None,
                             load_on_RAM=self.load_on_RAM,
                             output_folder=None,
                             input_file=self.input_data_file,
                             verbose=verbose_sub
                             )
        self.ndims = self.data.ndims
        self.__check_npoints()
        self.__set_pars_info()

    def __set_pars_info(self):
        """
        Private method used by the :meth:`DnnLik.__set_data <DNNLikelihood.DnnLik._DnnLik__set_data>` one
        to set parameters info. It sets the attributes:

            - :attr:`DnnLik.pars_central <DNNLikelihood.DnnLik.pars_central>`
            - :attr:`DnnLik.pars_pos_poi <DNNLikelihood.DnnLik.pars_pos_poi>`
            - :attr:`DnnLik.pars_pos_nuis <DNNLikelihood.DnnLik.pars_pos_nuis>`
            - :attr:`DnnLik.pars_labels <DNNLikelihood.DnnLik.pars_labels>`
            - :attr:`DnnLik.pars_labels_auto <DNNLikelihood.DnnLik.pars_labels_auto>`
            - :attr:`DnnLik.pars_bounds <DNNLikelihood.DnnLik.pars_bounds>`
        
        by copying the corresponding attributes of the :mod:`Data <data>` object 
        :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.
        """
        self.pars_central = self.data.pars_central
        self.pars_pos_poi = self.data.pars_pos_poi
        self.pars_pos_nuis = self.data.pars_pos_nuis
        self.pars_labels = self.data.pars_labels
        self.pars_labels_auto = self.data.pars_labels_auto
        self.pars_bounds = self.data.pars_bounds

    def __set_model_hyperparameters(self):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set attributes corresponding to model hyperparameters from the private dictionaries

            - :attr:`DnnLik.__model_data_inputs <DNNLikelihood.DnnLik._DnnLik__model_data_inputs>`
            - :attr:`DnnLik.__model_define_inputs <DNNLikelihood.DnnLik._DnnLik__model_define_inputs>`
            - :attr:`DnnLik.__model_train_inputs <DNNLikelihood.DnnLik._DnnLik__model_train_inputs>`
        
        The following attributes are set:
            
            - :attr:`DnnLik.scalerX_bool <DNNLikelihood.DnnLik.scalerX_bool>`
            - :attr:`DnnLik.scalerY_bool <DNNLikelihood.DnnLik.scalerY_bool>`
            - :attr:`DnnLik.weighted <DNNLikelihood.DnnLik.weighted>`
            - :attr:`DnnLik.hidden_layers <DNNLikelihood.DnnLik.hidden_layers>`
            - :attr:`DnnLik.act_func_out_layer <DNNLikelihood.DnnLik.act_func_out_layer>`
            - :attr:`DnnLik.dropout_rate <DNNLikelihood.DnnLik.dropout_rate>`
            - :attr:`DnnLik.batch_norm <DNNLikelihood.DnnLik.batch_norm>`
            - :attr:`DnnLik.epochs_required <DNNLikelihood.DnnLik.epochs_required>`
            - :attr:`DnnLik.batch_size <DNNLikelihood.DnnLik.batch_size>`
        """
        self.scalerX_bool = self.__model_data_inputs["scalerX"]
        self.scalerY_bool = self.__model_data_inputs["scalerY"]
        self.weighted = self.__model_data_inputs["weighted"]
        self.hidden_layers = self.__model_define_inputs["hidden_layers"]
        self.act_func_out_layer = self.__model_define_inputs["act_func_out_layer"]
        self.dropout_rate = self.__model_define_inputs["dropout_rate"]
        self.batch_norm = self.__model_define_inputs["batch_norm"]
        #self.kernel_initializer = self.__model_define_inputs["kernel_initializer"]
        self.epochs_required = self.__model_train_inputs["epochs"]
        self.batch_size = self.__model_train_inputs["batch_size"]

    def __set_tf_objects(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set attributes corresponding to |tf_keras_link| objects by calling the private methods:

            - :meth:`DnnLik.__set_optimizer <DNNLikelihood.DnnLik._DnnLik__set_optimizer>`
            - :meth:`DnnLik.__set_loss <DNNLikelihood.DnnLik._DnnLik__set_loss>`
            - :meth:`DnnLik.__set_metrics <DNNLikelihood.DnnLik._DnnLik__set_metrics>`
            - :meth:`DnnLik.__set_callbacks <DNNLikelihood.DnnLik._DnnLik__set_callbacks>`
        
        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        _, verbose_sub = self.set_verbosity(verbose)
        self.__set_optimizer(verbose=verbose_sub)  # this defines the string optimizer_string and object optimizer
        self.__set_loss(verbose=verbose_sub)  # this defines the string loss_string and the object loss
        self.__set_metrics(verbose=verbose_sub)  # this defines the lists metrics_string and metrics
        self.__set_callbacks(verbose=verbose_sub)  # this defines the lists callbacks_strings and callbacks

    def __load_summary_json_and_log(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one 
        to import part of a previously saved
        :class:`DnnLik <DNNLikelihood.DnnLik>` object from the files 
        :attr:`DnnLik.input_summary_json_file <DNNLikelihood.DnnLik.input_summary_json_file>` and
        :attr:`DnnLik.input_log_file <DNNLikelihood.DnnLik.input_log_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        with open(self.input_summary_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        #if self.model_max != {}:
        #    self.model_max["x"] = np.array(self.model_max["x"])
        self.log = dictionary
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded summary and log json",
                               "files names": [path.split(self.input_summary_json_file)[-1],
                                               path.split(self.input_log_file)[-1]],
                               "files paths": [self.input_summary_json_file,
                                               self.input_log_file]}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik summary json and log files loaded in", str(end-start), ".", show=verbose)

    def __load_history(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one 
        to set the :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>` attribute
        from the file
        :attr:`DnnLik.input_history_json_file <DNNLikelihood.DnnLik.input_history_json_file>`.
        Once the attribute is set, it is used to set the 
        :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>` one, determined from the 
        length of the ``"loss"`` item of the :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>` dictionary.
        If the file is not found the :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>` and
        :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>` attributes are set to
        an empty dictionary ``{}`` and ``0``, respectively. 

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            with open(self.input_history_json_file) as json_file:
                self.history = json.load(json_file)
            self.epochs_available = len(self.history['loss'])
        except:
            print("No history file available. The history attribute will be initialized to {}.")
            self.history = {}
            self.epochs_available = 0
            return
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded history json",
                               "file name": path.split(self.input_history_json_file)[-1],
                               "file path": self.input_history_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik history json file loaded in", str(end-start), ".", show=verbose)

    def __load_model(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one 
        to set the :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` attribute, 
        corresponding to the |tf_keras_model_link|, from the file
        :attr:`DnnLik.input_tf_model_h5_file <DNNLikelihood.DnnLik.input_tf_model_h5_file>`.
        If the file is not found the attribute is set to ``None``.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            self.model = load_model(self.input_tf_model_h5_file, custom_objects={"mean_error": self.mean_error,
                                                                                 "mean_percentage_error": self.mean_percentage_error,
                                                                                 "R2_metric": self.R2_metric})
        except:
            print("No model file available. The model and epochs_available attributes will be initialized to None and 0, respectively.")
            self.model = None
            return
        if self.model is not None:
            try:
                self.model.history = callbacks.History()
                self.model.history.model = self.model
                self.model.history.history = self.history
                self.model.history.params = {"verbose": 1, "epochs": self.epochs_available}
                self.model.history.epoch = np.arange(self.epochs_available).tolist()
            except:
                print("No training history file available.")
                return
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded tf model h5 and tf model history pickle",
                               "file name": path.split(self.input_tf_model_h5_file)[-1],
                               "file path": self.input_tf_model_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik tf model h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_scalers(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the :attr:`DnnLik.scalerX <DNNLikelihood.DnnLik.scalerX>` and 
        :attr:`DnnLik.scalerY <DNNLikelihood.DnnLik.scalerY>` attributes from the file
        :attr:`DnnLik.input_scalers_pickle_file <DNNLikelihood.DnnLik.input_scalers_pickle_file>`.
        If the file is not found the attributes are set to ``None``.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            pickle_in = open(self.input_scalers_pickle_file, "rb")
            self.scalerX = pickle.load(pickle_in)
            self.scalerY = pickle.load(pickle_in)
            pickle_in.close()
        except:
            print("No scalers file available. The scalerX and scalerY attributes will be initialized to None.")
            self.scalerX = None
            self.scalerY = None
            return
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded scalers h5",
                               "file name": path.split(self.input_scalers_pickle_file)[-1],
                               "file path": self.input_scalers_pickle_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik scalers h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_data_indices(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one 
        to set the attributes:
        
            - :attr:`DnnLik.idx_train <DNNLikelihood.DnnLik.idx_train>`
            - :attr:`DnnLik.idx_val <DNNLikelihood.DnnLik.idx_val>`
            - :attr:`DnnLik.idx_test <DNNLikelihood.DnnLik.idx_test>`

        from the file :attr:`DnnLik.input_idx_h5_file <DNNLikelihood.DnnLik.input_idx_h5_file>`.
        Once the attributes are set, the items ``"idx_train"``, ``"idx_val"``, and ``"idx_test"``
        of the :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` dictionary attribute
        of the :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`
        is updated to match the three index attributes
        If the file is not found the attributes are set to ``None`` and the :mod:`Data <data>` object is
        not touched.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            h5_in = h5py.File(self.input_idx_h5_file, "r")
        except:
            print("No data indices file available. The idx_train, idx_val, and idx_test attributes will be initialized to empty arrays.")
            self.idx_train, self.idx_val, self.idx_test = [np.array([], dtype="int"),np.array([], dtype="int"),np.array([], dtype="int")]
            return
        data = h5_in.require_group("idx")
        self.idx_train = data["idx_train"][:]
        self.idx_val = data["idx_val"][:]
        self.idx_test = data["idx_test"][:]
        self.data.data_dictionary["idx_train"] = self.idx_train
        self.data.data_dictionary["idx_val"] = self.idx_val
        self.data.data_dictionary["idx_test"] = self.idx_test
        h5_in.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded data indices h5",
                               "file name": path.split(self.input_idx_h5_file)[-1],
                               "file path": self.input_idx_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik data indices h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_predictions(self,verbose=None):
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one
        to set the :attr:`DnnLik.predictions <DNNLikelihood.DnnLik.predictions>` attribute 
        from the file
        :attr:`DnnLik.input_predictions_h5_file <DNNLikelihood.DnnLik.input_predictions_h5_file>`.
        If the file is not found the attributes is set to an empty dictionary ``{}``.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            dictionary = dd.io.load(self.input_predictions_h5_file)
            self.predictions = dictionary
        except:
            print("No predictions file available. The predictions attribute will be initialized to {}.")
            self.reset_predictions(verbose=verbose_sub)
            return
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded predictions h5",
                               "file name": path.split(self.input_predictions_h5_file)[-1],
                               "file path": self.input_predictions_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DnnLik predictions h5 file loaded in",str(end-start), ".", show=verbose)

    #def __load(self,verbose=None):
    #    """
    #    Bla bla
    #    """
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    self.__load_summary_json_and_log(verbose=verbose)
    #    self.__load_model(verbose=verbose)
    #    self.__load_history(verbose=verbose)
    #    self.__load_scalers(verbose=verbose)
    #    self.__load_data_indices(verbose=verbose)
    #    self.__load_predictions(verbose=verbose)

    def __set_optimizer(self,verbose=None):
        """
        Private method used by the 
        :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._DnnLik__set_tf_objects>` one
        to set the |tf_keras_optimizers_link| object. It sets the
        :attr:`DnnLik.optimizer_string <DNNLikelihood.DnnLik.optimizer_string>`
        and :attr:`DnnLik.optimizer <DNNLikelihood.DnnLik.optimizer>` attributes. The former is set from the
        :attr:`DnnLik.__model_optimizer_inputs <DNNLikelihood.DnnLik._DnnLik__model_optimizer_inputs>` 
        dictionary, while the latter is set by evaluating the former.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if type(self.__model_optimizer_inputs) == str:
            self.optimizer_string = self.__model_optimizer_inputs
        elif type(self.__model_optimizer_inputs) == dict:
            name = self.__model_optimizer_inputs["name"]
            string = name+"("
            for key, value in utils.dic_minus_keys(self.__model_optimizer_inputs,["name"]).items():
                if type(value) == str:
                    value = "'"+value+"'"
                string = string+str(key)+"="+str(value)+", "
            optimizer_string = str("optimizers."+string+")").replace(", )", ")")
            self.optimizer_string = optimizer_string
        else:
            raise Exception("Could not set optimizer. The model_optimizer_inputs argument does not have a valid format (str or dict).")
        self.optimizer = eval(optimizer_string)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "optimizer set",
                               "optimizer": self.optimizer_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__
        print("Optimizer set to:", self.optimizer_string, show=verbose)

    def __set_loss(self, verbose=None):
        """
        Private method used by the 
        :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._DnnLik__set_tf_objects>` one
        to set the |tf_keras_losses_link| object. It sets the
        :attr:`DnnLik.loss_string <DNNLikelihood.DnnLik.loss_string>`
        and :attr:`DnnLik.loss <DNNLikelihood.DnnLik.loss>` attributes. The former is set from the
        :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>` 
        dictionary, while the latter is set by evaluating the former.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        loss_string = self.__model_compile_inputs["loss"]
        try:
            loss_obj = losses.deserialize(loss_string)
            print("Loss set to:",loss_string,show=verbose)
        except:
            try:
                loss_obj = eval("self."+loss_string)
                print("Loss set to:", loss_string, show=verbose)
            except:
                print("Could not set loss", loss_string, ".", show=verbose)
        self.loss_string = loss_string
        self.loss = loss_obj
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loss set",
                               "loss": self.loss_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_metrics(self, verbose=None):
        """
        Private method used by the 
        :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._DnnLik__set_tf_objects>` one
        to set the |tf_keras_metrics_link| objects. It sets the
        :attr:`DnnLik.metrics_string <DNNLikelihood.DnnLik.metrics_string>`
        and :attr:`DnnLik.metrics <DNNLikelihood.DnnLik.metrics>` attributes. The former is set from the
        :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>` 
        dictionary, while the latter is set by evaluating each item in the the former.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        metrics_string = self.__model_compile_inputs["metrics"]
        metrics_obj = list(range(len(metrics_string)))
        print("Setting metrics")
        for i in range(len(metrics_string)):
            try:
                metrics_obj[i] = metrics.deserialize(metrics_string[i])
                print("\tAdded metric:",metrics_string[i],show=verbose)
            except:
                try:
                    metrics_obj[i] = eval("self."+metrics_string[i])
                    print("\tAdded metric:",metrics_string[i],show=verbose)
                except:
                    try:
                        metrics_obj[i] = eval("self."+utils.metric_name_unabbreviate(metrics_string[i]))
                        print("\tAdded metric:",metrics_string[i],show=verbose)
                    except:
                        print("\tCould not set metric", metrics_string[i], ".",show=verbose)
        self.metrics_string = metrics_string
        self.metrics = metrics_obj
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "metrics set",
                               "metrics": self.metrics_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_callbacks(self, verbose=None):
        """
        Private method used by the 
        :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._DnnLik__set_tf_objects>` one
        to set the |tf_keras_callbacks_link| objects. It sets the
        :attr:`DnnLik.callbacks_strings <DNNLikelihood.DnnLik.callbacks_strings>`
        and :attr:`DnnLik.callbacks <DNNLikelihood.DnnLik.callbacks>` attributes. The former is set from the
        :attr:`DnnLik.__model_callbacks_inputs <DNNLikelihood.DnnLik._DnnLik__model_callbacks_inputs>` 
        dictionary, while the latter is set by evaluating each item in the the former.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        callbacks_strings = []
        callbacks_string = [cb for cb in self.__model_callbacks_inputs if type(cb) == str]
        callbacks_dict = [cb for cb in self.__model_callbacks_inputs if type(cb) == dict]
        print("Setting callbacks")
        for cb in callbacks_string:
            if cb == "PlotLossesKeras":
                #self.output_figure_plot_losses_keras_file = self.output_figures_base_file+"_plot_losses_keras.pdf"
                #utils.check_rename_file(self.output_figure_plot_losses_keras_file)
                #string = "PlotLossesKeras(fig_path='" + self.output_figure_plot_losses_keras_file+"')"
                string = "PlotLossesKeras()"
            elif cb == "ModelCheckpoint":
                self.output_checkpoints_folder = path.join(self.output_folder, "checkpoints")
                self.output_checkpoints_files = path.join(self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                utils.check_create_folder(self.output_checkpoints_folder)
                string = "callbacks.ModelCheckpoint(filepath='" + self.output_checkpoints_files+"')"
            elif cb == "TensorBoard":
                self.output_tensorboard_log_dir = path.join(self.output_folder, "tensorboard_logs")
                utils.check_create_folder(self.output_tensorboard_log_dir)
                #utils.check_create_folder(path.join(self.output_folder, "tensorboard_logs/fit"))
                string = "callbacks.TensorBoard(log_dir='" + self.output_tensorboard_log_dir+"')"
            else:
                string = "callbacks."+cb+"()"
            callbacks_strings.append(string)
            print("\tAdded callback:", string, show=verbose)
        for cb in callbacks_dict:
            name = cb["name"]
            #if name == "PlotLossesKeras":
            #    self.output_figure_plot_losses_keras_file = self.output_figures_base_file+"_plot_losses_keras.pdf"
            #    utils.check_rename_file(self.output_figure_plot_losses_keras_file)
            #    string = "fig_path = '"+self.output_figure_plot_losses_keras_file + "', "
            ##    name = "callbacks."+name
            if name == "ModelCheckpoint":
                self.output_checkpoints_folder = path.join(self.output_folder, "checkpoints")
                self.output_checkpoints_files = path.join(self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                utils.check_create_folder(self.output_checkpoints_folder)
                string = "filepath = '"+self.output_checkpoints_files+"', "
                name = "callbacks."+name
                for key, value in utils.dic_minus_keys(cb, ["name"]).items():
                    if key == "monitor" and type(value) == str:
                        if "val_" in value:
                            value = value.split("val_")[1]
                        if value == "loss":
                            value = "val_loss"
                        else:
                            value = "val_" + utils.metric_name_unabbreviate(value)
                    if type(value) == str:
                        value = "'"+value+"'"
                    if not "filepath" in key:
                        string = string+str(key)+"="+str(value)+", "
            elif name == "TensorBoard":
                self.output_tensorboard_log_dir = path.join(self.output_folder, "tensorboard_logs")
                utils.check_create_folder(self.output_tensorboard_log_dir)
                #utils.check_create_folder(path.join(self.output_folder, "tensorboard_logs/fit"))
                string = "log_dir = '"+self.output_tensorboard_log_dir+"', "
                name = "callbacks."+name
                for key, value in utils.dic_minus_keys(cb, ["name"]).items():
                    if key == "monitor" and type(value) == str:
                        if "val_" in value:
                            value = value.split("val_")[1]
                        if value == "loss":
                            value = "val_loss"
                        else:
                            value = "val_" + utils.metric_name_unabbreviate(value)
                    if type(value) == str:
                        value = "'"+value+"'"
                    if not "log_dir" in key:
                        string = string+str(key)+"="+str(value)+", "
            else:
                string = ""
                name = "callbacks."+name
                for key, value in utils.dic_minus_keys(cb,["name"]).items():
                    if key == "monitor" and type(value) == str:
                        if "val_" in value:
                            value = value.split("val_")[1]
                        if value == "loss":
                            value = "val_loss"
                        else:
                            value = "val_" + utils.metric_name_unabbreviate(value)
                    if type(value) == str:
                        value = "'"+value+"'"
                    string = string+str(key)+"="+str(value)+", "
            string = str(name+"("+string+")").replace(", )", ")")
            callbacks_strings.append(string)
            #callbacks.append(eval(string))
            print("\tAdded callback:", string, show=verbose)
        if not "TerminateOnNaN" in str(callbacks_strings):
            callbacks_strings.append("callbacks.TerminateOnNaN()")
        callbacks_strings = [s.replace("'C:\\","r'C:\\") for s in callbacks_strings]
        callbacks_obj = list(range(len(callbacks_strings)))
        for i in range(len(callbacks_strings)):
            try:
                callbacks_obj[i] = eval(callbacks_strings[i])
            except Exception as e:
                print("Could not set callbacks", callbacks_strings[i], ".",show=verbose)
                print(e)
        self.callbacks_strings = callbacks_strings
        self.callbacks = callbacks_obj
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "callbacks set",
                               "callbacks": self.callbacks_strings}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_epochs_to_run(self):
        """
        Private method that returns the number of epochs to run computed as the difference between the value of
        :attr:`DnnLik.epochs_required <DNNLikelihood.DnnLik.epochs_required>` and the value of
        :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>`, i.e. the 
        number of epochs available in 
        :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>`.
        """
        if self.epochs_required <= self.epochs_available and self.epochs_available > 0:
            epochs_to_run = 0
        else:
            epochs_to_run = self.epochs_required-self.epochs_available
        return epochs_to_run

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

    def compute_sample_weights(self, nbins=100, power=1, verbose=None):
        """
        Method that computes weights of :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.Y_train>`
        points given their distribution by calling the
        :meth:`Data.compute_sample_weights <DNNLikelihood.Data.compute_sample_weights>` method of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.
        When the :attr:`DnnLik.weighted <DNNLikelihood.DnnLik.weighted>` is ``True``, the method is called
        with default arguments when data are generated. Manually call the method after data generation to compute weights with
        custom arguments.
        
        - **Arguments**

            - **nbins**

                Number of bins to histogram the 
                sample data

                    - **type**: ``int``
                    - **default**: ``100``

            - **power**

                Exponent of the inverse of the bin count used
                to assign weights.

                    - **type**: ``float``
                    - **default**: ``1``

            - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Returns**

            |Numpy_link| array of the same length of ``sample`` containing
            the required weights.
            
                - **type**: ``numpy.ndarray``
                - **shape**: ``(len(sample),)``
        """
        _, verbose_sub = self.set_verbosity(verbose)
        self.W_train = self.data.compute_sample_weights(
            self.Y_train, nbins=nbins, power=power, verbose=verbose_sub).astype(self.dtype)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "computed sample weights"}
        #self.save_log(overwrite=True, verbose=verbose_sub)

    def define_scalers(self, verbose=None):
        """
        Method that defines |standard_scalers_link| based on the values of the
        :attr:`DnnLik.scalerX_bool <DNNLikelihood.DnnLik.scalerX_bool>` and 
        :attr:`DnnLik.scalerY_bool <DNNLikelihood.DnnLik.scalerY_bool>` attributes.
        When the boolean attribute is ``True`` the scaler is fit to the corresponding training data, otherwise it is set
        equal to the identity.
        The method computes the scalers by calling the corresponding method 
        :meth:`Data.define_scalers <DNNLikelihood.Data.define_scalers>` method of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.

        - **Arguments**

           - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``
        """
        _, verbose_sub = self.set_verbosity(verbose)
        self.scalerX, self.scalerY = self.data.define_scalers(self.X_train, self.Y_train, self.scalerX_bool, self.scalerY_bool, verbose=verbose_sub)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined scalers",
                               "scaler X": self.scalerX_bool,
                               "scaler Y": self.scalerY_bool}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved by generate_train_data

    def generate_train_data(self, verbose=None):
        """
        Method that generates training and validation data corresponding to the attributes

            - :attr:`DnnLik.idx_train <DNNLikelihood.DnnLik.idx_train>`
            - :attr:`DnnLik.X_train <DNNLikelihood.DnnLik.X_train>`
            - :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.Y_train>`
            - :attr:`DnnLik.idx_val <DNNLikelihood.DnnLik.idx_val>`
            - :attr:`DnnLik.X_val <DNNLikelihood.DnnLik.X_val>`
            - :attr:`DnnLik.Y_val <DNNLikelihood.DnnLik.Y_val>`

        Data are generated by calling the methods
        :meth:`Data.update_train_data <DNNLikelihood.Data.update_train_data>` or
        :meth:`Data.generate_train_data <DNNLikelihood.Data.generate_train_data>` of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`
        depending on the value of :attr:`DnnLik.same_data <DNNLikelihood.DnnLik.same_data>`.

        When the :class:`DnnLik <DNNLikelihood.DnnLik>` object is not part of a
        :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object, that is when the
        :attr:`DnnLik.standalone <DNNLikelihood.DnnLik.standalone>` attribute is ``True``, 
        or when the :attr:`DnnLik.same_data <DNNLikelihood.DnnLik.same_data>`
        attribute is ``True``, that means that all members of the ensemble will share the same data (or a
        subset of the same data if they have different number of points), then data are kept up-to-date in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` attribute of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.
        This means that data are not generated again if they are already available from another member and that
        if the number of points is increased, data are added to the existing ones and are not re-generated from scratch.

        When instead the :class:`DnnLik <DNNLikelihood.DnnLik>` object is part of a
        :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object and the
        :attr:`DnnLik.same_data <DNNLikelihood.DnnLik.same_data>` attribute is ``False``,
        data are re-generated from scratch at each call of the method.

        If the :attr:`DnnLik.weighted <DNNLikelihood.DnnLik.weighted>` attribute is ``True`` the
        :meth:`DnnLik.compute_sample_weights <DNNLikelihood.DnnLik.compute_sample_weights>` is called
        with default arguments. In order to compute sample weights with different arguments, the user should call again the
        :meth:`DnnLik.compute_sample_weights <DNNLikelihood.DnnLik.compute_sample_weights>` method after 
        data generation.

        - **Arguments**

           - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Generate data
        if self.same_data:
            self.data.update_train_data(self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        else:
            self.data.generate_train_data(self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        self.idx_train = self.data.data_dictionary["idx_train"][:self.npoints_train]
        self.X_train = self.data.data_dictionary["X_train"][:self.npoints_train].astype(self.dtype)
        self.Y_train = self.data.data_dictionary["Y_train"][:self.npoints_train].astype(self.dtype)
        self.idx_val = self.data.data_dictionary["idx_val"][:self.npoints_train]
        self.X_val = self.data.data_dictionary["X_val"][:self.npoints_val].astype(self.dtype)
        self.Y_val = self.data.data_dictionary["Y_val"][:self.npoints_val].astype(self.dtype)
        self.pars_bounds_train = np.vstack([np.min(self.X_train,axis=0),np.max(self.X_train,axis=0)]).T
        self.pred_bounds_train = np.array([np.min(self.Y_train), np.max(self.Y_train)])
        # Define scalers
        self.define_scalers(verbose=verbose_sub)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "generated train data",
                               "data": ["idx_train", "X_train", "Y_train", "idx_val", "X_val", "Y_val"],
                               "npoints train": self.npoints_train,
                               "npoints val": self.npoints_val}
        if self.weighted:
            print("In order to compute sample weights with the desired parameters please run the function\
                   self.compute_sample_weights(bins=100, power=1) before training.\n Proceding with sample weights\
                   computed with default parameters (bins=100 and power=1).", show=verbose)
            self.compute_sample_weights()
        self.save_log(overwrite=True, verbose=verbose_sub)

    def generate_test_data(self, verbose=None):
        """
        Method that generates test data corresponding to the attributes

            - :attr:`DnnLik.idx_train <DNNLikelihood.DnnLik.idx_test>`
            - :attr:`DnnLik.X_train <DNNLikelihood.DnnLik.X_test>`
            - :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.Y_test>`

        Differently from the training and validation data, the test data are always shared by different 
        :class:`DnnLik <DNNLikelihood.DnnLik>` objects within an
        :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object.
        Therefore test data are always kept up-to-date in the 
        :attr:`Data.data_dictionary <DNNLikelihood.Data.data_dictionary>` attribute of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`
        and are always generated by calling the :meth:`Data.generate_test_data <DNNLikelihood.Data.generate_test_data>` of the 
        :mod:`Data <data>` object :attr:`DnnLik.data <DNNLikelihood.DnnLik.data>`.
        In this way test data are never re-generated if they are (even partially) available.

        - **Arguments**

           - **verbose**

                Verbosity mode.
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.

                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Generate data
        self.data.generate_test_data(self.npoints_test, verbose=verbose)
        self.idx_test = self.data.data_dictionary["idx_test"][:self.npoints_train]
        self.X_test = self.data.data_dictionary["X_test"][:self.npoints_test].astype(self.dtype)
        self.Y_test = self.data.data_dictionary["Y_test"][:self.npoints_test].astype(self.dtype)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "generated test data",
                               "data": ["idx_test", "X_test", "Y_test"],
                               "npoints test": self.npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_define(self,verbose=None):
        """
        Method that defines the |tf_keras_model_link| stored in the 
        :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` attribute.
        The model is defined from the attributes

            - :attr:`DnnLik.hidden_layers <DNNLikelihood.DnnLik.hidden_layers>`
            - :attr:`DnnLik.batch_norm <DNNLikelihood.DnnLik.batch_norm>`
            - :attr:`DnnLik.dropout_rate <DNNLikelihood.DnnLik.dropout_rate>`

        All hidden layers in the module are |tf_keras_layers_dense_link| layers.
        If :attr:`DnnLik.batch_norm <DNNLikelihood.DnnLik.batch_norm>` is ``True``, then a 
        |tf_keras_batch_normalization_link| layer is added after the input layer and after each hidden layer.
        If :attr:`DnnLik.dropout_rate <DNNLikelihood.DnnLik.dropout_rate>` is larger than ``0``, then
        a |tf_keras_dropout_link| layer is added after each hidden layer with the given dropout rate.
        
        The method also sets the three attributes:

            - :attr:`DnnLik.model_params <DNNLikelihood.DnnLik.model_params>`
            - :attr:`DnnLik.model_trainable_params <DNNLikelihood.DnnLik.model_trainable_params>`
            - :attr:`DnnLik.model_non_trainable_params <DNNLikelihood.DnnLik.model_non_trainable_params>`

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``
        
        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Define model
        start = timer()
        inputLayer = Input(shape=(self.ndims,))
        if self.batch_norm:
            x = BatchNormalization()(inputLayer)
        if self.hidden_layers[0][1] == "selu":
            x = Dense(self.hidden_layers[0][0], activation=self.hidden_layers[0][1], kernel_initializer="lecun_normal")(inputLayer)
        elif self.hidden_layers[0][1] != "selu" and len(self.hidden_layers[0]) < 3:
            x = Dense(self.hidden_layers[0][0], activation=self.hidden_layers[0][1])(inputLayer)
        else:
            x = Dense(self.hidden_layers[0][0], activation=self.hidden_layers[0][1], kernel_initializer=self.hidden_layers[0][2])(inputLayer)
        if self.batch_norm:
            x = BatchNormalization()(x)
        if self.dropout_rate != 0:
            if self.hidden_layers[0][1] == "selu":
                x = AlphaDropout(self.dropout_rate)(x)
            else:
                x = Dropout(self.dropout_rate)(x)
        if len(self.hidden_layers) > 1:
            for i in self.hidden_layers[1:]:
                if i[1] == "selu":
                    x = Dense(i[0], activation=i[1], kernel_initializer="lecun_normal")(x)
                if i[1] != "selu" and len(i) <3:
                    x = Dense(i[0], activation=i[1])(x)
                else:
                    x = Dense(i[0], activation=i[1], kernel_initializer=i[2])(x)
                if self.batch_norm:
                    x = BatchNormalization()(x)
                if self.dropout_rate != 0:
                    if i[1] == "selu":
                        x = AlphaDropout(self.dropout_rate)(x)
                    else:
                        x = Dropout(self.dropout_rate)(x)
        outputLayer = Dense(1, activation=self.act_func_out_layer)(x)
        self.model = Model(inputs=inputLayer, outputs=outputLayer)
        self.model_params = int(self.model.count_params())
        self.model_trainable_params = int(np.sum([K.count_params(p) for p in self.model.trainable_weights]))
        self.model_non_trainable_params = int(np.sum([K.count_params(p) for p in self.model.non_trainable_weights]))
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x.replace("\"","'")))
        self.log[timestamp] = {"action": "defined tf model",
                               "model summary": summary_list}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for DNNLikelihood",self.name,"defined in", str(end-start), "s.",show=verbose)
        print(self.model.summary(), show=verbose)

    def model_compile(self,verbose=None):
        """
        Method that compiles the |tf_keras_model_link| stored in the 
        :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` attribute.
        The model is compiled by calling the |tf_keras_model_compile_link| method and passing it the attributes

            - :attr:`DnnLik.loss <DNNLikelihood.DnnLik.loss>`
            - :attr:`DnnLik.optimizer <DNNLikelihood.DnnLik.optimizer>`
            - :attr:`DnnLik.metrics <DNNLikelihood.DnnLik.metrics>`

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Compile model
        start = timer()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "compiled tf model"}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for DNNLikelihood",self.name,"compiled in",str(end-start),"s.",show=verbose)

    def model_build(self, gpu="auto", force=False, verbose=None):
        """
        Method that calls the methods

            - :meth:`DnnLik.model_define <DNNLikelihood.DnnLik.model_define>`
            - :meth:`DnnLik.model_compile <DNNLikelihood.DnnLik.model_compile>`

        on a specific GPU by using the |tf_distribute_onedevicestrategy_link| class.
        Using this method different :class:`DNNLik <DNNLikelihood.DNNLik>` members of a
        :class:`DNNLikEnsemble <DNNLikelihood.DNNLikEnsemble>` object can be compiled and run in parallel 
        on different GPUs (when available).
        Notice that, in case the model has already been created and compiled, the method does not re-builds the
        model on a different GPU unless the ``force`` flag is set to ``True`` (default is ``False``).

        - **Arguments**

            - **gpu**
            
                GPU number (e.g. 0,1,etc..) of the GPU where the model should be built.
                The available GPUs are listed in the 
                :attr:`DnnLik.active_gpus <DNNLikelihood.DnnLik.active_gpus>`.
                If ``gpu="auto"`` the first GPU, corresponding to number ``0`` is automatically set.
                    
                    - **type**: ``int`` or ``str``
                    - **default**: ``auto`` (``0``)
            
            - **force**
            
                If set to ``True`` the model is re-built even if it was already 
                available.
                    
                    - **type**: ``bool``
                    - **default**: ``False``
            
            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self.model
            create = False
            if self.model._is_compiled:
                compile = False
            else:
                compile = True
        except:
            create = True
            compile = True
        if force:
            create = True
            compile = True
        if not create and not compile:
            print("Model already built.", show=verbose)
            return
        if self.gpu_mode:
            if gpu == "auto":
                gpu = 0
            elif gpu > len(self.available_gpus):
                print("gpu", gpu,
                      "does not exist. Continuing on first gpu.", show=verbose)
                gpu = 0
            self.training_device = self.available_gpus[gpu]
            device_id = self.training_device[0]
        else:
            if gpu != "auto":
                print("GPU mode selected without any active GPU. Proceeding with CPU support.",show=verbose)
            self.training_device = self.available_cpu
            device_id = self.training_device[0]
        strategy = tf.distribute.OneDeviceStrategy(device=device_id)
        print("Building tf model for DNNLikelihood", self.name,"on device", self.training_device, show=verbose)
        with strategy.scope():
            if create:
                self.model_define(verbose=verbose_sub)
            if compile:
                self.model_compile(verbose=verbose_sub)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "built tf model",
                               "gpu mode": self.gpu_mode,
                               "device id": device_id}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_train(self,verbose=None):
        """
        Method that trains the |tf_keras_model_link| stored in the 
        :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>` attribute.
        The model is trained by calling the |tf_keras_model_fit_link| method and passing it the inputs 

            - X_train (attribute :attr:`DnnLik.X_train <DNNLikelihood.DnnLik.X_train>` scaled with :attr:`DnnLik.scalerX <DNNLikelihood.DnnLik.scalerX>`)
            - X_val (attribute :attr:`DnnLik.X_val <DNNLikelihood.DnnLik.X_val>` scaled with :attr:`DnnLik.scalerX <DNNLikelihood.DnnLik.scalerX>`)
            - Y_train (attribute :attr:`DnnLik.Y_train <DNNLikelihood.DnnLik.Y_train>` scaled with :attr:`DnnLik.scalerY <DNNLikelihood.DnnLik.scalerY>`)
            - Y_val (attribute :attr:`DnnLik.Y_val <DNNLikelihood.DnnLik.Y_val>` scaled with :attr:`DnnLik.Y_val <DNNLikelihood.DnnLik.Y_val>`)
            - :attr:`DnnLik.W_train <DNNLikelihood.DnnLik.W_train>` (if :attr:`DnnLik.weighted <DNNLikelihood.DnnLik.weighted>` is ``True``)
            - epochs_to_run (difference between :attr:`DnnLik.epochs_required <DNNLikelihood.DnnLik.epochs_required>` and :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>`)
            - :attr:`DnnLik.batch_size <DNNLikelihood.DnnLik.batch_size>`
            - :attr:`DnnLik.callbacks <DNNLikelihood.DnnLik.callbacks>`

        After training the method updates the attributes

            - :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>`
            - :attr:`DnnLik.history <DNNLikelihood.DnnLik.history>`
            - :attr:`DnnLik.epochs_available <DNNLikelihood.DnnLik.epochs_available>`

        - **Arguments**

            - **verbose**
            
                Verbosity mode. The second value returned by the
                :meth:`DNNLik.set_verbosity <DNNLikelihood.DNNLik.set_verbosity>` method is the verbosity mode passed to the
                |tf_keras_model_fit_link|. See the documentation of |tf_keras_model_fit_link| for the available verbosity modes.
                Also see the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Scale data
        start = timer()
        epochs_to_run = self.__set_epochs_to_run()
        print("Required a total of",self.epochs_required,"epochs.",self.epochs_available,"epochs already available. Training for a maximum of",epochs_to_run,"epochs.", show=verbose)
        if epochs_to_run == 0:
            print("Please increase epochs_required to train for more epochs.", show=verbose)
        else:
            if len(self.X_train) <= 1:
                print("Generating train/val data",show=verbose)
                self.generate_train_data(verbose=verbose_sub)
            print("Scaling training/val data.", show=verbose)
            X_train = self.scalerX.transform(self.X_train)
            X_val = self.scalerX.transform(self.X_val)
            Y_train = self.scalerY.transform(self.Y_train.reshape(-1, 1)).reshape(len(self.Y_train))
            Y_val = self.scalerY.transform(self.Y_val.reshape(-1, 1)).reshape(len(self.Y_val))
            #print([type(X_train),type(X_val),type(Y_train),type(Y_train)],show=verbose)
            # If PlotLossesKeras == in callbacks set plot style
            if "PlotLossesKeras" in str(self.callbacks_strings):
                plt.style.use(mplstyle_path)
            # Train model
            print("Start training of model for DNNLikelihood",self.name, ".",show=verbose)
            if self.weighted:
                # Train
                history = self.model.fit(X_train, Y_train, sample_weight=self.W_train, initial_epoch=self.epochs_available, epochs=self.epochs_required, batch_size=self.batch_size, verbose=verbose_sub,
                        validation_data=(X_val, Y_val), callbacks=self.callbacks)
            else:
                history = self.model.fit(X_train, Y_train, initial_epoch=self.epochs_available, epochs=self.epochs_required, batch_size=self.batch_size, verbose=verbose_sub,
                        validation_data=(X_val, Y_val), callbacks=self.callbacks)
            end = timer()
            self.training_time = (end - start)/epochs_to_run
            history = history.history
            for k, v in history.items():
                history[k] = list(np.array(v, dtype=self.dtype))
            if self.history == {}:
                print("no existing history",show=verbose)
                self.history = history
            else:
                print("existing history", show=verbose)
                for k, v in self.history.items():
                    self.history[k] = v + history[k]
            self.epochs_available = len(self.history["loss"])
            print("Updating model.history and model.epoch attribute.",show=verbose)
            self.model.history.history = self.history
            self.model.history.params["epochs"] = self.epochs_available
            self.model.history.epoch = np.arange(self.epochs_available).tolist()
            if "PlotLossesKeras" in str(self.callbacks_strings):
                plt.close()
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.log[timestamp] = {"action": "trained tf model",
                                   "epochs run": epochs_to_run,
                                   "epochs total": self.epochs_available,
                                   "batch size": self.batch_size,
                                   "training time": self.training_time}
            self.save_log(overwrite=True, verbose=verbose_sub)
            print("Model for DNNLikelihood", self.name, "successfully trained for",
                  epochs_to_run, "epochs in", self.training_time, "s.", show=verbose)

    def check_x_bounds(self,pars_val,pars_bounds):
        res = []
        for i in range(len(pars_val)):
            tmp = []
            if pars_bounds[i][0] ==-np.inf:
                tmp.append(True)
            else:
                if pars_val[i] >= pars_bounds[i][0]:
                    tmp.append(True)
                else:
                    tmp.append(False)
            if pars_bounds[i][1] ==np.inf:
                tmp.append(True)
            else:
                if pars_val[i] <= pars_bounds[i][1]:
                    tmp.append(True)
                else:
                    tmp.append(False)
            res.append(tmp)
        return np.all(res)

    def check_y_bounds(self,pred_val, pred_bounds):
        return pred_val >= pred_bounds[0] and pred_val <= pred_bounds[1]

    def model_predict(self, X, batch_size=None, steps=None, x_boundaries=False, y_boundaries=False, save_log=True, verbose=None):
        """
        Method that predicts in batches by calling the |tf_keras_model_predict_link| method of
        :attr:`DnnLik.model <DNNLikelihood.DnnLik.model>`.
        To predict, the method first scales the vector ``X`` with the scaler 
        :attr:`DnnLik.scalerX <DNNLikelihood.DnnLik.scalerX>`, then computed the prediction, and finally returns
        the predicted vercor ``Y`` inversely scaled with :attr:`DnnLik.scalerY <DNNLikelihood.DnnLik.scalerY>` and the
        prediction time normalized to a signle point (i.e. divided by ``batch_size``).
        The method takes into account original boundaries for the likelihood function by setting the value of the prediction
        for all points outside boundaries to ``-np.inf``.

        - **Arguments**

            - **X**
            
                Input vector ``X`` for which the predictions ``Y``
                are computed
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndim)``

            - **batch_size**
            
                Batch size to be used for predictions.
                If ``None`` (default), then the value stored in the :attr:`DnnLik.batch_size <DNNLikelihood.DnnLik.batch_size>`
                attribute is used.
                    
                    - **type**: ``int``
                    - **default**: ``None``

            - **steps**
            
                Total number of batches on which the prediction is made. If ``None`` (default), predictions are computed
                for all ``X``. See the documentation of |tf_keras_model_predict_link| for more details.
                    
                    - **type**: ``int``
                    - **default**: ``None``

            - **x_boundaries**
            
                Implements boundaries on the input vector. If an ``x`` point has a parameter that falls outside
                ``pars_bounds``, the corresponding prediction is set to ``-np.inf``. It could have the following values:

                    - ``False``: no bounds are imposed and all predictions are accepted.
                    - ``True``: the parameters bounds of the original likelihood function stored in the 
                        :attr:`DnnLik.pars_bounds <DNNLikelihood.DnnLik.pars_bounds>` are used.
                    - ``"original"``: same as ``True``
                    - ``"train"``: the parameters bounds computed from the maximum and minimum values of the training points
                        stored in the :attr:`DnnLik.pars_bounds_train <DNNLikelihood.DnnLik.pars_bounds_train>` are used.
                    
                    - **type**: ``bool`` or ``str``
                    - **default**: ``False``

            - **y_boundaries**

                Implements boundaries on the output vector. If ``True``, then if a ``y`` point is bigger(smaller) than the
                largest(smallest) ``y`` training point stored in the 
                :attr:`DNNLik.pred_bounds_train <DNNLikelihood.DNNLik.pred_bounds_train>` attribute, 
                the corresponding prediction is set to ``-np.inf``. If ``False`` then 
                no constraint is imposed.

                    - **type**: ``bool``
                    - **default**: ``"False"``

            - **save_log**
            
                If ``True`` the :attr:`DnnLik.log <DNNLikelihood.DnnLik.log>` attribute is updated and the corresponding file is
                saved. This option is used to switch off log updated when the method is called by the method
                :meth:`DNNLik.model_predict_scalar <DNNLikelihood.DNNLik.model_predict_scalar>`, which predicts one point at a time.

                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>` (only if ``save_log=True``)
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if x_boundaries == "original": 
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == True:
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == "train":
            pars_bounds = np.array(self.pars_bounds_train)
        elif x_boundaries == False:
            pass
            #pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        else:
            print("Invalid input for 'x_boundaries'. Assuming False.")
            x_boundaries = False
        # Scale data
        if batch_size == None:
            batch_size = self.batch_size
        print("Scaling data.", show=verbose)
        X = self.scalerX.transform(X)
        pred = self.scalerY.inverse_transform(self.model.predict(X, batch_size=batch_size, steps=steps, verbose=verbose_sub)).reshape(len(X))
        if x_boundaries:
            for i in range(len(X)):
                x = X[i]
                if not self.check_x_bounds(x, pars_bounds):
                    pred[i] = -np.inf
        if y_boundaries:
            for i in range(len(X)):
                if not self.check_y_bounds(pred, self.pred_bounds_train):
                    pred[i] = -np.inf
        end = timer()
        prediction_time = (end - start)/len(X)
        if save_log:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.log[timestamp] = {"action": "predicted with tf model",
                                   "batch size": batch_size,
                                   "npoints": len(pred),
                                   "prediction time": prediction_time}
            self.save_log(overwrite=True, verbose=verbose_sub)
        return [pred, prediction_time]

    def model_predict_scalar(self, x, x_boundaries=False, y_boundaries=False, verbose=None):
        """
        Method that returns a prediction on a single input point.
        It calls the method :meth:`DNNLik.model_predict <DNNLikelihood.DNNLik.model_predict>`
        with ``batch_size=1`` and converts the output to a scalar.

        - **Arguments**

            - **x**
            
                One dimensional input vector ``x`` for which a scalar predictions ``y``
                is computed
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,)``

            - **x_boundaries**
            
                Argument passed to the 
                :meth:`DNNLik.model_predict <DNNLikelihood.DNNLik.model_predict>` method.
                    
                    - **type**: ``str``
                    - **default**: ``"none"``

            - **y_boundaries**

                Argument passed to the 
                :meth:`DNNLik.model_predict <DNNLikelihood.DNNLik.model_predict>` method.

                    - **type**: ``bool``
                    - **default**: ``"False"``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        pred = self.model_predict(x, batch_size=1, steps=None, x_boundaries=x_boundaries, y_boundaries=y_boundaries, save_log=False, verbose=False)[0][0]
        return pred

    def __set_optimizer_minimization(self, optimizer):
        """
        Bla bla bla.
        """
        if type(optimizer) != dict:
            raise Exception("Could not set optimizer. The optimizer argument does not have a valid format.")
        if optimizer == {}:
            opt_string = "optimizers.SGD(learning_rate=1)"
        else:
            name = optimizer["name"]
        if name == "scipy" or name == "Scipy":
            opt_string = "scipy.optimize"
        else:
            try:
                string = name+"("
                for key, value in utils.dic_minus_keys(optimizer,["name","options"]).items():
                    if type(value) == str:
                        value = "'"+value+"'"
                    string = string+str(key)+"="+str(value)+", "
                opt_string = str("optimizers."+string+")").replace(", )", ")")
            except:
                raise Exception("Could not set optimizer. The optimizer argument does not have a valid format.")
        return [opt_string,eval(opt_string)]

    def compute_maximum_model(self,
                              pars_init=None,
                              optimizer={},
                              x_boundaries=False, 
                              y_boundaries=False,
                              timestamp = None,
                              save=True,
                              verbose=None):
        """
        Method that computes the maximum of the DNNLikelihood predictor with an optimizer given by the user.
        The two optimizer that are supported are |scipy_link| and |tf_keras_link|. They work as follows.

        - |scipy_link|

            If the argument ``optimizer`` is ``"scipy"``, then the method uses the |scipy_optimize_minimize_link| method to minimize
            minus times the :meth:``DNNLik.model_predict_scalar <DNNLikelihood.DNNLik.model_predict_scalar> function. Optimization is unconstrained
            since constraints determined by the ``x_boundaries`` and ``y_boundaries`` inputs are already applied on the predictor function.
            By default the method uses the |scipy_optimize_minimize_powell_link| method and passes the ``maxiter`` and ``tolerance`` inputs 
            to the analog ``ftol`` and ``maxiter`` input of the |scipy_optimize_minimize_powell_link| method, respectively. 
            Additional (or different) arguments can be passed to the 
            |scipy_optimize_minimize_link| method through the input dictionary ``scipy_options``.

        - |tf_keras_link|

            If the argument ``optimizer`` is not ``scipy``, then |tf_keras_link| is used for the optimization. When ``optimizer``
            is ``None`` the |tf_keras_optimizers_link_2| is set to |tf_keras_optimizer_SGD| with ``learning_rate=1``.
            On the other hand, the user can pass a custom ``optimizer`` in the same way as optimizers are passed to the 
            :class:``DNNLik <DNNLikelihood.DNNLik>` object, see the documentation of :argument:`model_optimizer_inputs`.
            The optimization is done for a maximum of ``maxiter`` iterations, 500 iterations per time. 
            Every ``run_length`` iterations the prediction is compared with the previous one
            and, when the difference is smaller than ``tolerance`` the optimization is stopped. If ``tolerance`` is not reached
            within ``maxiter`` iterations, then the result is returned and a warning message is printed.
            If the optimization starts to deteriorate (worse result after the next 500 iterations), then the ``learning_rate`` of the
            |tf_keras_optimizers_link_2| is reduces by a factor of two.

        The method saves the result of the optimization in the :attr:`DNNLik.model_max <DNNLikelihood.DNNLik.model_max>` dictionary,
        containing the ``X`` (type: ``numpy.ndarray``) and ``Y`` (type: ``float``) items.

        - **Arguments**

            - **pars_init**
            
                Starting point for the optimization. If not specified (``None``), then
                it is set to the parameters central value :attr:`DNNLik.pars_central <DNNLikelihood.DNNLik.pars_central>`.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(ndim,)``
                    - **default**: ``None`` (automatically modified to :attr:`DNNLik.pars_central <DNNLikelihood.DNNLik.pars_central>`)

            - **maxiter**
            
                Maximum number of iteration for the optimization. When using the |tf_keras_link| for optimization this is quantized
                in groups of 500 iterations. Even if ``tolerance`` is not reached, otpimization ends after ``maxiter`` iterations.
                    
                    - **type**: ``int``
                    - **default**: - **default**: ``None`` (automatically modified to ``1000`` times :attr:`DNNLik.ndims <DNNLikelihood.DNNLik.ndims>`)

            - **tolerance**

                Relative difference between the prediction at iteration i and prediction at iteration i+niter, where niter is
                one for |scipy_link| and 500 for |tf_keras_link| optimization.

                    - **type**: ``float``
                    - **default**: ``0.0001``

            - **optimizer**

                It coule be either the string ``"scipy"`` for |scipy_link| optimization, or a string or a dictionary for |tf_keras_link| optimization.
                In the latter case it can be input in the same way as :argument:`model_optimizer_inputs` class argument.

                    - **type**: ``str`` or ``dict``
                    - **default**: ``None`` (automatically modified to ``"tf.keras.optimizers.SGD(learning_rate=1)``)

            - **scipy_options**

                Arguments passed as additional ``"options"`` (dictionary) to the 
                |scipy_optimize_link| method.

                    - **type**: ``dict``
                    - **default**: ``{}``

            - **run_length**

                In the case of optimization with |tf_keras_link|, it represents the 
                number of iterations before evaluating the prediction and comparing it with 
                the previous one to estimate the tolerance.

                    - **type**: ``int``
                    - **default**: ``500``

            - **x_boundaries**

                Implements boundaries on the input vector. If an ``x`` point has a parameter that falls outside
                ``pars_bounds``, the corresponding prediction is set to ``-np.inf`` for |scipy_link| optimization
                and to the prediction multiplied by ``1.1`` plus the squared modulus of the input vector for |tf_keras_link| 
                optimization. It could have the following values:

                    - ``False``: no bounds are imposed and all predictions are accepted.
                    - ``True``: the parameters bounds of the original likelihood function stored in the 
                        :attr:`DnnLik.pars_bounds <DNNLikelihood.DnnLik.pars_bounds>` are used.
                    - ``"original"``: same as ``True``
                    - ``"train"``: the parameters bounds computed from the maximum and minimum values of the training points
                        stored in the :attr:`DnnLik.pars_bounds_train <DNNLikelihood.DnnLik.pars_bounds_train>` are used.
                    
                    - **type**: ``bool`` or ``str``
                    - **default**: ``False``

                For |scipy_link| optimization the argument is passed directly to the 
                :meth:`DNNLik.model_predict_scalar <DNNLikelihood.DNNLik.model_predict_scalar>` method.

                    - **type**: ``bool`` or ``str``
                    - **default**: ``False``

            - **y_boundaries**

                Argument passed to the 
                :meth:`DNNLik.model_predict_scalar <DNNLikelihood.DNNLik.model_predict_scalar>` method.

                    - **type**: ``bool``
                    - **default**: ``"False"``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if pars_init is None:
            pars_init = np.array(self.pars_central).astype(self.dtype)
        else:
            pars_init = np.array(pars_init).astype(self.dtype)
        if x_boundaries == "original":
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == True:
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == "train":
            pars_bounds = np.array(self.pars_bounds_train)
        elif x_boundaries == False:
            pass
        else:
            print("Invalid input for 'x_boundaries'. Assuming False.")
            x_boundaries = False
        if x_boundaries:
            if not self.check_x_bounds(pars_init, pars_bounds):
                raise Exception("pars_init out of bounds.")
        opt_log, opt = self.__set_optimizer_minimization(optimizer)
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"], ["logpdf_max_model"],
                                                 [{}],verbose=verbose_sub)
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_max_model"], [timestamp],
                                                 [{}],verbose=False)
        if "scipy" in opt_log:
            utils.check_set_dict_keys(optimizer, ["name",
                                                  "method",
                                                  "options"],
                                                 ["scipy","Powell",{}],verbose=verbose_sub)
            method=optimizer["method"]
            options=optimizer["options"]
            print("Optimizing with scipy.optimize.", show=verbose)
            def minus_loglik(x):
                return -self.model_predict_scalar(x, x_boundaries=x_boundaries, y_boundaries=y_boundaries)
            ml = opt.minimize(minus_loglik, pars_init, method=method, options=options)
            self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp] = {"x": np.array(ml["x"]), "y": -ml["fun"]}
            end = timer()
            print("Optimized in", str(end-start), "s.", show=verbose)
        else:
            print("Optimizing with tensorflow.", show=verbose)
            utils.check_set_dict_keys(optimizer, ["options"],
                                                 [{}],verbose=verbose_sub)
            utils.check_set_dict_keys(optimizer["options"], ["maxiter",
                                                             "tolerance"],
                                                            [1000*self.ndims,0.0001],
                                                            verbose=verbose_sub)
            utils.check_set_dict_keys(optimizer["options"], ["run_length"],
                                                            [np.min([500,optimizer["options"]["maxiter"]])],
                                                            verbose=verbose_sub)
            maxiter = optimizer["options"]["maxiter"]
            run_length = optimizer["options"]["run_length"]
            tolerance = optimizer["options"]["tolerance"]
            pars_init_scaled = self.scalerX.transform(pars_init.reshape(1,-1))
            x_var = tf.Variable(pars_init_scaled.reshape(1,-1), dtype=self.dtype)
            if x_boundaries:
                pars_bounds_scaled = np.vstack([self.scalerX.transform(np.nan_to_num(pars_bounds[:,0].reshape(1,-1))).reshape(-1,),
                                                self.scalerX.transform(np.nan_to_num(pars_bounds[:,1].reshape(1,-1))).reshape(-1,)]).T
                x_bounds = tf.reshape(tf.Variable(pars_bounds_scaled, dtype=self.dtype),(-1,2))
                y_bounds = self.scalerY.transform(np.array(self.pred_bounds_train).reshape(-1,1)).reshape(-1,)
            if y_boundaries:
                y_bounds = self.scalerY.transform(np.array(self.pred_bounds_train).reshape(-1,1)).reshape(-1,)
            if not x_boundaries and not y_boundaries:
                @tf.function
                def f():
                    return tf.reshape(-1*(self.model(x_var)), [])
            elif x_boundaries and not y_boundaries:
                @tf.function
                def f():
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(tf.reshape(x_var, (-1,)),x_bounds[:,0]),
                                                               tf.math.greater(x_bounds[:,1],tf.reshape(x_var, (-1,))))),
                                  lambda: tf.reshape(-1*(self.model(x_var)),[]),
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(x_var, x_var, axes=2)))
                    return res
            elif not x_boundaries and y_boundaries:
                @tf.function
                def f():
                    res = tf.reshape(-1*(self.model(x_var)), [])
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(res, y_bounds[0]),
                                                               tf.math.greater(y_bounds[1],res))),
                                  lambda: res,
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(x_var, x_var, axes=2)))
                    return res
            elif x_boundaries and y_boundaries:
                @tf.function
                def f():
                    res = tf.reshape(-1*(self.model(x_var)), [])
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(tf.reshape(x_var, (-1,)),x_bounds[:,0]),
                                                               tf.math.greater(x_bounds[:,1],tf.reshape(x_var, (-1,))))),
                                  lambda: tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(res, y_bounds[0]),
                                                                               tf.math.greater(y_bounds[1], res))),
                                                  lambda: res,
                                                  lambda: -y_bounds[0]*(1 + tf.tensordot(x_var, x_var, axes=2))),
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(x_var, x_var, axes=2)))
                    return res
            nruns = int(maxiter/run_length)
            run_steps = np.array([[i*run_length, (i+1)*run_length] for i in range(nruns)])
            last_run_length = maxiter-run_length*nruns
            if last_run_length != 0:
                nruns = nruns+1
                run_steps = np.concatenate((run_steps, np.array([[run_steps[-1][1]+1,run_steps[-1][1]+last_run_length]])))
            for i in range(nruns):
                step_before = run_steps[i][0]
                value_before = self.scalerY.inverse_transform([-f().numpy()])[0]
                for _ in range(run_steps[i][0], run_steps[i][1]):
                    opt.minimize(f, var_list=[x_var])
                step_after = run_steps[i][1]
                value_after = self.scalerY.inverse_transform([-f().numpy()])[0]
                variation = (value_before-value_after)/value_before
                print("Step:",step_before,"Value:",value_before,"-- Step:",step_after,"Value:",value_after,r"-- % Variation",variation, show=verbose)
                if variation > 0 and variation < tolerance:
                    end = timer()
                    print("Converged to tolerance",tolerance,"in",str(end-start),"s.", show=verbose)
                    break
                if value_after<value_before:
                    lr = opt._hyper['learning_rate']#.numpy()
                    opt._hyper['learning_rate'] = lr/2
                    print("Optimizer learning rate reduced from",
                          lr.numpy(), "to", (lr/2).numpy(), ".", show=verbose)
            x_final = self.scalerX.inverse_transform(x_var.numpy())[0]
            y_final = self.scalerY.inverse_transform([-f().numpy()])[0]
            self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp] = {"x": np.array(x_final), "y": y_final}
            end = timer()
            print("Did not converge to tolerance",tolerance,"using",maxiter,"steps.", show=verbose)
            print("Best tolerance",variation,"reached in",str(end-start),"s.", show=verbose)
        if y_boundaries and self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"] < self.pred_bounds_train[0]: 
            print("Warning: the model maximum (",self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"],") is smaller than the minimum y value in the training data (",self.pred_bounds_train[0],").")
        if y_boundaries and self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"] > self.pred_bounds_train[1]:
            print("Warning: the model maximum (",self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"],") is larger than the maximum y value in the training data (",self.pred_bounds_train[1],").")
        self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["pars_init"] = pars_init
        self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["optimizer"] = optimizer
        self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["x_boundaries"] = x_boundaries
        self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y_boundaries"] = y_boundaries
        self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["optimization_time"] = end-start
        self.log[timestamp] = {"action": "computed maximum model",
                               "optimizer": opt_log,
                               "optimization time": end-start,
                               "x": self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["x"],
                               "y": self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"]}
        if save:
            self.save_predictions(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)
        print("Maximum of DNN computed in", str(end-start),"s.", show=verbose)

    def compute_profiled_maximum_model(self,
                                       pars=None, 
                                       pars_val=None,
                                       pars_init=None,
                                       optimizer={},
                                       x_boundaries=False,
                                       y_boundaries=False,
                                       timestamp=None,
                                       save=True,
                                       verbose=None):
        """
        Bla bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            multiple=False
        else:
            multiple=True
        if pars is None:
            raise Exception("The 'pars' input argument cannot be empty.")
        if pars_val is None:
            raise Exception("The 'pars_val' input argument cannot be empty.")
        if len(pars)!=len(pars_val):
            raise Exception("The input arguments 'pars' and 'pars_val' should have the same length.")
        start = timer()
        pars = np.array(pars)
        pars_string = str(pars.tolist())
        pars_insert = pars - range(len(pars))
        pars_val = np.array(pars_val)
        if pars_init is None:
            pars_init = np.array(self.pars_central).astype(self.dtype)
        else:
            pars_init = np.array(pars_init).astype(self.dtype)
        for i in range(len(pars)):
            pars_init[pars[i]] = pars_val[i]
        if x_boundaries == "original": 
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == True:
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == "train":
            pars_bounds = np.array(self.pars_bounds_train)
        elif x_boundaries == False:
            pass
        else:
            print("Invalid input for 'x_boundaries'. Assuming False.")
            x_boundaries = False
        if x_boundaries:
            if not self.check_x_bounds(pars_init, pars_bounds):
                raise Exception("pars_init out of bounds.")
        opt_log, opt = self.__set_optimizer_minimization(optimizer)
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"],
                                  ["logpdf_profiled_max_model"],
                                  [{}],verbose=verbose_sub)
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"],
                                  [timestamp],
                                  [{}], verbose=verbose_sub)
        if "scipy" in opt_log:
            utils.check_set_dict_keys(optimizer, ["method",
                                                  "options"],
                                                 ["Powell",{}],verbose=verbose_sub)
            method = optimizer["method"]
            options = optimizer["options"]
            print("Optimizing with scipy.optimize.", show=verbose)
            pars_init_reduced = np.delete(pars_init, pars)
            def minus_loglik(x):
                return -self.model_predict_scalar(np.insert(x, pars_insert, pars_val), x_boundaries=x_boundaries, y_boundaries=y_boundaries)
            ml = opt.minimize(minus_loglik, pars_init_reduced, method=method, options=options)
            x_final = np.insert(ml["x"], pars_insert, pars_val)
            y_final = -ml["fun"]
            end = timer()
            print("Optimized in", str(end-start), "s.", show=verbose)
        else:
            print("Optimizing with tensorflow.", show=verbose)
            utils.check_set_dict_keys(optimizer, ["options"],
                                                 [{}],verbose=verbose_sub)
            utils.check_set_dict_keys(optimizer["options"], ["maxiter",
                                                             "tolerance"],
                                                            [1000*self.ndims,0.0001],
                                                            verbose=verbose_sub)
            utils.check_set_dict_keys(optimizer["options"], ["run_length"],
                                                            [np.min([500,optimizer["options"]["maxiter"]])],
                                                            verbose=verbose_sub)
            maxiter = optimizer["options"]["maxiter"]
            run_length = optimizer["options"]["run_length"]
            tolerance = optimizer["options"]["tolerance"]
            pars_init_scaled = self.scalerX.transform(pars_init.reshape(1,-1))
            x_var = tf.reshape(tf.Variable(pars_init_scaled.reshape(1,-1), dtype=self.dtype),(-1,))
            idx_reduced = np.reshape(np.delete(np.array(list(range(self.ndims))),pars),(-1,1))
            x_var_reduced = tf.Variable(tf.gather_nd(x_var,idx_reduced))
            idx_0 = np.reshape(pars,(-1,1))
            x_var_0 = tf.Variable(tf.gather_nd(x_var,idx_0))
            idx_resort = np.reshape(np.argsort(np.reshape(np.concatenate((idx_0,idx_reduced)),(-1,))),(-1,1))
            if x_boundaries:
                pars_bounds_scaled = np.vstack([self.scalerX.transform(np.nan_to_num(pars_bounds[:, 0].reshape(1, -1))).reshape(-1,),
                                                self.scalerX.transform(np.nan_to_num(pars_bounds[:, 1].reshape(1, -1))).reshape(-1,)]).T
                x_bounds = tf.reshape(tf.Variable(
                    pars_bounds_scaled, dtype=self.dtype), (-1, 2))
                y_bounds = self.scalerY.transform(
                    np.array(self.pred_bounds_train).reshape(-1, 1)).reshape(-1,)
            if y_boundaries:
                y_bounds = self.scalerY.transform(np.array(self.pred_bounds_train).reshape(-1,1)).reshape(-1,)
            if not x_boundaries and not y_boundaries:
                @tf.function
                def f():
                    var = tf.concat((x_var_0,x_var_reduced),axis=0)
                    var = tf.reshape(tf.gather_nd(var,idx_resort),(1,-1))
                    return tf.reshape(-1*(self.model(var)), [])
            elif x_boundaries and not y_boundaries:
                @tf.function
                def f():
                    var = tf.concat((x_var_0,x_var_reduced),axis=0)
                    var = tf.reshape(tf.gather_nd(var,idx_resort),(1,-1))
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(tf.reshape(var, (-1,)),x_bounds[:,0]),
                                                               tf.math.greater(x_bounds[:,1],tf.reshape(var, (-1,))))),
                                  lambda: tf.reshape(-1*(self.model(var)),[]),
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(var, var, axes=2)))
                    return res
            elif not x_boundaries and y_boundaries:
                @tf.function
                def f():
                    var = tf.concat((x_var_0,x_var_reduced),axis=0)
                    var = tf.reshape(tf.gather_nd(var,idx_resort),(1,-1))
                    res = tf.reshape(-1*(self.model(var)), [])
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(res, y_bounds[0]),
                                                               tf.math.greater(y_bounds[1],res))),
                                  lambda: res,
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(var, var, axes=2)))
                    return res
            elif x_boundaries and y_boundaries:
                @tf.function
                def f():
                    var = tf.concat((x_var_0, x_var_reduced), axis=0)
                    var = tf.reshape(tf.gather_nd(var,idx_resort),(1,-1))
                    res = tf.reshape(-1*(self.model(var)), [])
                    res = tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(tf.reshape(var, (-1,)),x_bounds[:,0]),
                                                               tf.math.greater(x_bounds[:,1],tf.reshape(var, (-1,))))),
                                  lambda: tf.cond(tf.reduce_all(tf.logical_and(tf.math.greater(res, y_bounds[0]),
                                                                               tf.math.greater(y_bounds[1], res))),
                                                  lambda: res,
                                                  lambda: -y_bounds[0]*(1 + tf.tensordot(var, var, axes=2))),
                                  lambda: -y_bounds[0]*(1 + tf.tensordot(var, var, axes=2)))
                    return res
            nruns = int(maxiter/run_length)
            run_steps = np.array([[i*run_length, (i+1)*run_length] for i in range(nruns)])
            last_run_length = maxiter-run_length*nruns
            if last_run_length != 0:
                nruns = nruns+1
                run_steps = np.concatenate((run_steps, np.array([[run_steps[-1][1]+1,run_steps[-1][1]+last_run_length]])))
            for i in range(nruns):
                step_before = run_steps[i][0]
                value_before = self.scalerY.inverse_transform([-f().numpy()])[0]
                for _ in range(run_steps[i][0], run_steps[i][1]):
                    opt.minimize(f, var_list=[x_var_reduced])
                step_after = run_steps[i][1]
                value_after = self.scalerY.inverse_transform([-f().numpy()])[0]
                variation = (value_before-value_after)/value_before
                print("Step:",step_before,"Value:",value_before,"-- Step:",step_after,"Value:",value_after,r"-- % Variation",variation, show=verbose)
                if variation > 0 and variation < tolerance:
                    end = timer()
                    print("Converged to tolerance",tolerance,"in",str(end-start),"s.", show=verbose)
                    break
                if value_after<value_before:
                    lr = opt._hyper['learning_rate']#.numpy()
                    opt._hyper['learning_rate'] = lr/2
                    print("Optimizer learning rate reduced from",lr.numpy(),"to",(lr/2).numpy(),".", show=verbose)
            var = tf.concat((x_var_0,x_var_reduced),axis=0)
            var = tf.reshape(tf.gather_nd(var,idx_resort),(1,-1))
            x_final = self.scalerX.inverse_transform(var.numpy())[0]
            y_final = self.scalerY.inverse_transform([-f().numpy()])[0]
            end = timer()
            print("Did not converge to tolerance",tolerance,"using",maxiter,"steps.", show=verbose)
            print("Best tolerance",variation,"reached in",str(end-start),"s.", show=verbose)
        ## Check for existing global maximum
        try:
            y_max=self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"]
            if self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["pars_init"] != pars_init:
                print("Warning: existing global maximum has been computed with a different parameters initialization 'pars_init'.", show=verbose_sub)
            if self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["optimizer"] != optimizer:
                print("Warning: existing global maximum has been computed with a different optimizer and or different optimization parameters.", show=verbose_sub)
            if self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["x_boundaries"] != x_boundaries:
                print("Warning: existing global maximum has been computed with different 'x_boundaries'.", show=verbose_sub)
            if self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y_boundaries"] != y_boundaries:
                print("Warning: existing global maximum has been computed with different 'y_boundaries'.", show=verbose_sub)
        except:
            print("Computing global maximum to estimate tmu test statistics.",show=verbose)
            self.compute_maximum_model(pars_init=pars_init,
                                       optimizer=optimizer,
                                       x_boundaries=x_boundaries, 
                                       y_boundaries=y_boundaries,
                                       timestamp=timestamp,
                                       save=True,
                                       verbose=verbose_sub)
            y_max = self.predictions["Frequentist_inference"]["logpdf_max_model"][timestamp]["y"]
        tmu_final = np.array([x_final[pars][0], -2*(y_final-y_max)])
        if multiple:
            try:
                self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["X"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["X"], [x_final]))
                self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["Y"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["Y"], [y_final]))
                self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["tmu"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["tmu"], [tmu_final]))
            except:
                self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp] = {"X": np.array([x_final]), "Y": np.array([y_final]), "tmu": np.array([tmu_final])}
        else:
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp] = {"X": np.array([x_final]), "Y": np.array([y_final]), "tmu": np.array([tmu_final])}
        if not multiple:
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["pars"] = pars
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["pars_init"] = pars_init
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["optimizer"] = optimizer
        if not multiple:
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["x_boundaries"] = x_boundaries
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["y_boundaries"] = y_boundaries
        try:    
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["optimization_times"].append(end-start)
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["global_optimization_time"]=np.array(self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["optimization_times"]).sum()
        except:
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["optimization_times"]=[end-start]
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["global_optimization_time"]=end-start
        self.log[timestamp] = {"action": "computed profiled maximum model",
                               "optimizer": opt_log,
                               "optimization time": end-start,
                               "pars": pars,
                               "x": self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["X"][-1],
                               "y": self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["Y"][-1]}
        if save:
            self.save_predictions(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)
        print("Profiled maxima of DNN computed in", str(end-start),"s.", show=verbose)

    def compute_profiled_maxima_model(self,
                                      pars=None,
                                      pars_ranges=None,
                                      pars_init=None,
                                      optimizer={},
                                      spacing="grid",
                                      x_boundaries=False,
                                      y_boundaries=False,
                                      timestamp = None,
                                      progressbar=True,
                                      save=True,
                                      verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if pars is None:
            raise Exception("The 'pars' input argument cannot be empty.")
        if pars_ranges is None:
            raise Exception("The 'pars_val' input argument cannot be empty.")
        if len(pars)!=len(pars_ranges):
            raise Exception("The input arguments 'pars' and 'pars_ranges' should have the same length.")
        start = timer()
        pars_string = str(np.array(pars).tolist())
        if progressbar:
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                print("If you want to show a progress bar please install the ipywidgets package.",show=verbose)
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
        if x_boundaries == "original": 
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == True:
            pars_bounds = np.array(self.pars_bounds)
        elif x_boundaries == "train":
            pars_bounds = np.array(self.pars_bounds_train)
        elif x_boundaries == False:
            pass
            #pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        else:
            print("Invalid input for 'x_boundaries'. Assuming False.")
            x_boundaries = False
        if x_boundaries:
            for i in range(len(pars_vals)):
                if self.check_x_bounds(pars_vals, pars_bounds):
                    pars_vals_bounded.append(pars_vals[i])
        else:
            pars_vals_bounded = pars_vals
        if len(pars_vals) != len(pars_vals_bounded):
            print("Deleted", str(len(pars_vals)-len(pars_vals_bounded)),"points outside the parameters allowed range.",show=verbose)
        res = []
        for pars_val in pars_vals_bounded:
            print("Optimizing for parameters:",pars," - values:",pars_val.tolist(),".",show=verbose)
            self.compute_profiled_maximum_model(pars=pars,
                                                pars_val=pars_val,
                                                pars_init=pars_init,
                                                optimizer=optimizer,
                                                x_boundaries=x_boundaries,
                                                y_boundaries=y_boundaries,
                                                timestamp=timestamp,
                                                save=False,
                                                verbose=verbose_sub)
            if progressbar:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(pars_vals_bounded))
        end = timer()
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["pars"] = pars
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["pars_ranges"] = pars_ranges
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["pars_init"] = pars_init
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["spacing"] = spacing
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["x_boundaries"] = x_boundaries
        self.predictions["Frequentist_inference"]["logpdf_profiled_max_model"][timestamp]["y_boundaries"] = y_boundaries
        self.log[timestamp] = {"action": "computed profiled maxima model",
                               "pars": pars, 
                               "pars_ranges": pars_ranges, 
                               "total time": end-start,
                               "spacing": spacing,
                               "npoints": len(pars_vals_bounded)}
        if save:
            self.save_predictions(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)
        print("Profiled maxima of DNN for",len(pars_vals_bounded),"points computed in",str(end-start),"s.", show=verbose)
        
    def compute_maximum_sample(self,
                               samples = "train",
                               timestamp = None,
                               save = True,
                               verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        samples = np.array([samples]).flatten()
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"], ["logpdf_max_sample"],
                                                    [{}],verbose=verbose_sub)
        for sample in samples:
            start = timer()
            if sample in ["train", "val", "test"]:
                if eval("len(self.X_"+sample+")") <= 1:
                    print("Generating",sample,"data", show=verbose)
                    if sample == "test":
                        self.generate_test_data(verbose=verbose_sub)
                    else:
                        self.generate_train_data(verbose=verbose_sub)
                X = eval("self.X_"+sample)
                Y = eval("self.Y_"+sample)
            else:
                raise Exception("Invalid sample input. It should be one of: 'train', 'val', or 'test'.")
            res = inference.compute_maximum_sample(X=X, Y=Y)
            utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_max_sample"],
                                                   [timestamp],
                                                   [{}],verbose=verbose_sub)
            end = timer()
            self.predictions["Frequentist_inference"]["logpdf_max_sample"][timestamp][sample] = {"x": np.array(res[0]), "y": res[1], "x_abs_err": np.array(res[2]), "y_abs_err": res[3], "optimization_time": end-start, "data_file": self.input_data_file, "idx_file": self.output_idx_h5_file}
            self.log[timestamp] = {"action": "computed maximum sample",
                                   "sample": sample,
                                   "total time": end-start,
                                   "x": self.predictions["Frequentist_inference"]["logpdf_max_sample"][timestamp][sample]["x"],
                                   "y": self.predictions["Frequentist_inference"]["logpdf_max_sample"][timestamp][sample]["y"]}
            print("Maximum of",sample,"sample computed in", str(end-start),"s.", show=verbose)
        if save:
            self.save_predictions(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    def compute_profiled_maxima_sample(self, 
                                       pars,
                                       pars_ranges,
                                       samples = "train", 
                                       spacing="grid",
                                       binwidths = "auto",
                                       x_boundaries=False,
                                       timestamp = None,
                                       progressbar=True,
                                       save=True,
                                       verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        samples = np.array([samples]).flatten()
        pars_string = str(np.array(pars).tolist())
        if progressbar:
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                print("If you want to show a progress bar please install the ipywidgets package.", show=verbose)
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                "width": "500px", "height": "14px",
                "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"], ["logpdf_profiled_max_sample"],
                                                    [{}],verbose=verbose_sub)
        utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"],
                                  [timestamp],
                                  [{}],verbose=verbose_sub)
        for sample in samples:
            start = timer()
            utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp],
                                  [sample],
                                  [{}], verbose=verbose_sub)
            if sample in ["train", "val", "test"]:
                if eval("len(self.X_"+sample+")") <= 1:
                    print("Generating",sample,"data", show=verbose)
                    if sample == "test":
                        self.generate_test_data(verbose=verbose_sub)
                    else:
                        self.generate_train_data(verbose=verbose_sub)
                X = eval("self.X_"+sample)
                Y = eval("self.Y_"+sample)
            else:
                raise Exception("Invalid sample input. It should be one of: 'train', 'val', or 'test'.")
            pars_vals = utils.get_sorted_grid(pars_ranges=pars_ranges, spacing=spacing)
            print("Total number of points:", len(pars_vals),".",show=verbose)
            pars_vals_bounded = []
            if x_boundaries == "original": 
                pars_bounds = np.array(self.pars_bounds)
            elif x_boundaries == True:
                pars_bounds = np.array(self.pars_bounds)
            elif x_boundaries == "train":
                pars_bounds = np.array(self.pars_bounds_train)
            elif x_boundaries == False:
                pass
                #pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
            else:
                print("Invalid input for 'x_boundaries'. Assuming False.")
                x_boundaries = False
            if x_boundaries:
                for i in range(len(pars_vals)):
                    if self.check_x_bounds(pars_vals, pars_bounds):
                        pars_vals_bounded.append(pars_vals[i])
            else:
                pars_vals_bounded = pars_vals
            if len(pars_vals) != len(pars_vals_bounded):
                print("Deleted", str(len(pars_vals)-len(pars_vals_bounded)),"points outside the parameters allowed range.",show=verbose)
            for pars_val in pars_vals_bounded:
                print("Optimizing for parameters:",pars," - values:",pars_val.tolist(),".",show=verbose)
                start_sub = timer()
                tmp = inference.compute_profiled_maximum_sample(pars=pars,
                                                                pars_val=pars_val,
                                                                X=X,
                                                                Y=Y,
                                                                binwidths=binwidths)
                x_final = tmp[0]
                y_final = tmp[1]
                x_abs_err_final = tmp[2]
                y_abs_err_final = tmp[3]
                try:
                    y_max = self.predictions["Frequentist_inference"]["logpdf_max_sample"][timestamp][sample]["y"]
                except:
                    print("Computing global maximum to estimate tmu test statistics.",show=verbose)
                    self.compute_maximum_sample(samples = sample,
                                                timestamp = timestamp,
                                                save = True,
                                                verbose = verbose_sub)
                    y_max = self.predictions["Frequentist_inference"]["logpdf_max_sample"][timestamp][sample]["y"]
                tmu_final = np.array([x_final[pars][0], -2*(y_final-y_max)])
                end_sub = timer()
                try:
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["X"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["X"], [x_final]))
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["Y"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["Y"], [y_final]))
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["tmu"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["tmu"], [tmu_final]))
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["X_abs_err"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["X_abs_err"], [tmp[2]]))
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["Y_abs_err"] = np.concatenate((self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["Y_abs_err"], [tmp[3]]))
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["optimization_times"].append(end_sub-start_sub)
                except:
                    self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample] = {"X": np.array([x_final]), "Y": np.array([y_final]), "tmu": np.array([tmu_final]), "X_abs_err": np.array([tmp[2]]), "Y_abs_err": np.array([tmp[3]]), "optimization_times": [end_sub-start_sub]}
                self.log[timestamp] = {"action": "computed profiled maximum model",
                                       "sample": sample,
                                       "pars": pars,
                                       "pars_val": pars_val,
                                       "optimization time": end_sub-start_sub}
                if progressbar:
                    iterator = iterator + 1
                    overall_progress.value = float(iterator)/(len(pars_vals_bounded))
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["pars"] = pars
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["pars_ranges"] = pars_ranges
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["data_file"] = self.input_data_file
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["idx_file"] = self.output_idx_h5_file
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["global_optimization_time"] = np.array(self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["optimization_times"]).sum()
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["spacing"] = spacing
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["binwidths"] = binwidths
            self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][sample]["x_boundaries"] = x_boundaries            
            end = timer()
            self.log[timestamp] = {"action": "computed profiled maxima sample",
                                   "sample": sample,
                                   "pars": pars,
                                   "pars_ranges": pars_ranges,
                                   "spacing": spacing,
                                   "npoints": len(pars_vals_bounded),
                                   "total time": end-start}
            print("Profiled maxima of",sample,"sample for",len(pars_vals_bounded),"points computed in",str(end-start),"s.", show=verbose)
        if save:
            self.save_predictions(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)
        
    def model_evaluate(self, X, Y, batch_size=None, steps=None, verbose=None):
        """
        Method that evaluates the :attr:`DNNLik.model <DNNLikelihood.DNNLik.model>` model on the 
        :attr:`DNNLik.metrics <DNNLikelihood.DNNLik.metrics>` by calling the 
        |tf_keras_model_evaluate_link| method.

        - **Arguments**

            - **X**
            
                |Numpy_link| array of X points for 
                model evaluation.
                    
                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``

            - **Y**
            
                |Numpy_link| array of corresponding Y points sfor 
                model evaluation.
                    
                    - **type**: ``numpy.ndarray``
                    - **default**: ``(ndims,)``

            - **batch_size**

                Batch size used for evaluation. If ``None`` (default), then it is set to the value of
                :attr:`DNNLik.batch_size <DNNLikelihood.DNNLik.batch_size>`.

                    - **type**: ``int``
                    - **default**: ``None`` (automatically modified to :attr:`DNNLik.batch_size <DNNLikelihood.DNNLik.batch_size>`)

            - **steps**
            
                Total number of batches on which the evaluation is made. If ``None`` (default), predictions are computed
                for all ``X``. See the documentation of |tf_keras_model_evaluate_link| for more details.
                    
                    - **type**: ``int``
                    - **default**: ``None``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None``

        - **Produces file**

            - :attr:`DnnLik.output_log_file <DNNLikelihood.DnnLik.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Scale data
        start = timer()
        if batch_size == None:
            batch_size = self.batch_size
        print("Scaling data.", show=verbose)
        X = self.scalerX.transform(X)
        Y = self.scalerY.transform(Y.reshape(-1, 1)).reshape(len(Y))
        pred = self.model.evaluate(X, Y, batch_size=batch_size, verbose=verbose_sub)
        end = timer()
        prediction_time = (end - start)/len(X)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "evaluated tf model",
                               "npoints": len(Y),
                               "evaluation time": prediction_time}
        self.save_log(overwrite=True, verbose=verbose_sub)
        return [pred, prediction_time]

    def generate_fig_base_title(self):
        """
        Generates a common title for figures including information on the model and saved in the 
        :attr:`DNNLik.fig_base_title <DNNLikelihood.DNNLik.fig_base_title>` attribute.
        """
        title = "Ndim: " + str(self.ndims) + " - "
        title = title + "Nevt: " + "%.E" % Decimal(str(self.npoints_train)) + " - "
        title = title + "Layers: " + str(len(self.hidden_layers)) + " - "
        title = title + "Nodes: " + str(self.hidden_layers[0][0]) + " - "
        title = title.replace("+", "") + "Loss: " + str(self.loss_string)
        self.fig_base_title = title

    def update_figures(self,figure_filename=None,overwrite=False):
        if figure_filename is None:
            raise Exception("figure_filename input argument of update_figures method needs to be specified while it is None.")
        else:
            if not overwrite:
                # search figure
                timestamp=None
                for k, v in self.predictions["Figures"].items():
                    if figure_filename in v:
                        timestamp = k
                    old_figure_filename = utils.check_rename_file(figure_filename,timestamp=timestamp)
                    if old_figure_filename is not None:
                        self.predictions["Figures"][k] = [q.replace(figure_filename,old_figure_filename) for q in v]

    def plot_training_history(self, metrics=["loss"], yscale="log", show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        metrics = np.unique(metrics)
        for metric in metrics:
            start = timer()
            metric = utils.metric_name_unabbreviate(metric)
            val_metric = "val_"+ metric
            figure_filename = self.output_figures_base_file+"_training_history_" + metric+".pdf"
            self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
            plt.plot(self.history[metric])
            plt.plot(self.history[val_metric])
            plt.yscale(yscale)
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.title(r"%s" % self.fig_base_title, fontsize=10)
            plt.xlabel(r"epoch")
            ylabel = (metric.replace("_", "-"))
            plt.ylabel(r"%s" % ylabel)
            plt.legend([r"training", r"validation"])
            plt.tight_layout()
            ax = plt.axes()
            #x1, x2, y1, y2 = plt.axis()
            plt.text(0.967, 0.2, r"%s" % self.summary_text, fontsize=7, bbox=dict(facecolor="green", alpha=0.15,
                                                                              edgecolor="black", boxstyle="round,pad=0.5"), ha="right", ma="left", transform=ax.transAxes)
            plt.savefig(r"%s" % (figure_filename))
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": path.split(figure_filename)[-1], 
                                   "file path": figure_filename}
            print(r"%s" % (figure_filename), "created and saved in", str(end-start), "s.", show=verbose)
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions
    
    def plot_tmu_sources_1d(self,
                            sources=["model"],
                            timestamp=None,
                            show_plot=False, 
                            overwrite=False, 
                            verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        pars_dict = {}
        tmu_dict = {}
        for source in sources:
            if source in ["train","val","test"]:
                pars_dict[source] = self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][source]["pars"]
                tmu_dict[source] = self.predictions["Frequentist_inference"]["logpdf_profiled_max_sample"][timestamp][source]["tmu"]
            else:
                pars_dict[source] = self.predictions["Frequentist_inference"]["logpdf_profiled_max_"+source][timestamp]["pars"]
                tmu_dict[source] = self.predictions["Frequentist_inference"]["logpdf_profiled_max_"+source][timestamp]["tmu"]
        tmp = list(pars_dict.values())
        if tmp.count(tmp[0]) == len(tmp) and len(tmp[0]) == 1:
            par=tmp[0][0]
        else:
            raise Exception("Parameters should be  should be the same for the different tmu sources.")
        figure_filename = self.output_figures_base_file+"_tmu_"+str(par) +".pdf"
        self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
        for k, v in tmu_dict.items():
            plt.plot(v[:,0],v[:,-1], label=r"%s"%k)
        plt.title(r"%s" % self.fig_base_title, fontsize=10)
        plt.xlabel(r"$t_{\mu}$(%s)"%(self.pars_labels[par]))
        plt.ylabel(r"%s"%(self.pars_labels[par]))
        plt.legend()
        plt.tight_layout()
        plt.savefig(r"%s" %figure_filename)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure", 
                               "file name": path.split(figure_filename)[-1], 
                               "file path": figure_filename}
        print(r"%s" % (figure_filename), "created and saved in", str(end-start), "s.", show=verbose)

    def plot_pars_coverage(self, pars=None, loglik=True, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        if pars == None:
            pars = self.pars_pos_poi
        else:
            pars = pars
        for par in pars:
            start=timer()
            if loglik:
                figure_filename = self.output_figures_base_file+"_par_loglik_coverage_" + str(par) +".pdf"
            else:
                figure_filename = self.output_figures_base_file+"_par_lik_coverage_" + str(par) +".pdf"
            self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
            nnn = min(1000,len(self.X_train),len(self.X_test))
            rnd_idx_train = np.random.choice(np.arange(self.npoints_train), nnn, replace=False)
            rnd_idx_test = np.random.choice(np.arange(self.npoints_test), nnn, replace=False)
            Y_pred_test, _ = self.model_predict(self.X_test[rnd_idx_test], batch_size=self.batch_size)
            if loglik:
                curve_train = np.array([self.X_train[rnd_idx_train,par],self.Y_train[rnd_idx_train]]).transpose()
                curve_test = np.array([self.X_test[rnd_idx_test,par],self.Y_test[rnd_idx_test]]).transpose()
                curve_test_pred = np.array([self.X_test[rnd_idx_test,par],Y_pred_test]).transpose()
            else:
                curve_train = np.array([self.X_train[rnd_idx_train,par],np.exp(self.Y_train[rnd_idx_train])]).transpose()
                curve_test = np.array([self.X_test[rnd_idx_test,par],np.exp(self.Y_test[rnd_idx_test])]).transpose()
                curve_test_pred = np.array([self.X_test[rnd_idx_test,par],np.exp(Y_pred_test)]).transpose()
            curve_train = curve_train[curve_train[:, 0].argsort()]
            curve_test = curve_test[curve_test[:, 0].argsort()]
            curve_test_pred = curve_test_pred[curve_test_pred[:,0].argsort()]
            plt.plot(curve_train[:,0], curve_train[:,1], color="green", marker="o", linestyle="dashed", linewidth=2, markersize=3, label=r"train")
            plt.plot(curve_test[:,0], curve_test[:,1], color="blue", marker="o", linestyle="dashed", linewidth=2, markersize=3, label=r"test")
            plt.plot(curve_test_pred[:,0], curve_test_pred[:,1], color="red", marker="o", linestyle="dashed", linewidth=2, markersize=3, label=r"pred")
            plt.title(r"%s" % self.fig_base_title, fontsize=10)
            plt.xlabel(r"%s"%(self.pars_labels[par]))
            if loglik:
                plt.ylabel(r"logprob ($\log\mathcal{L}+\log\mathcal{P}$)")
            else:
                plt.yscale("log")
                plt.ylabel(r"prob ($\mathcal{L}\cdot\mathcal{P}$)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(r"%s" %figure_filename)
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            self.log[timestamp] = {"action": "saved figure",
                                   "file name": path.split(figure_filename)[-1],
                                   "file path": figure_filename}
            print(r"%s" %figure_filename,"created and saved in",str(end-start),"s.", show=verbose)
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions

    def plot_lik_distribution(self, loglik=True, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        if loglik:
            figure_filename = self.output_figures_base_file+"_loglik_distribution.pdf"
        else:
            figure_filename = self.output_figures_base_file+"_lik_distribution.pdf"
        self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
        nnn = min(10000, len(self.X_train), len(self.X_test))
        rnd_idx_train = np.random.choice(np.arange(self.npoints_train), nnn, replace=False)
        rnd_idx_test = np.random.choice(np.arange(self.npoints_test), nnn, replace=False)
        Y_pred_test, _ = self.model_predict(self.X_test[rnd_idx_test], batch_size=self.batch_size)
        if loglik:
            bins = np.histogram(self.Y_train[rnd_idx_train], 50)[1]
            counts, _ = np.histogram(self.Y_train[rnd_idx_train], bins)
            integral = 1  # counts.sum()
            plt.step(bins[:-1], counts/integral, where="post",color="green", label=r"train")
            counts, _ = np.histogram(self.Y_test[rnd_idx_test], bins)
            integral = 1  # counts.sum()
            plt.step(bins[:-1], counts/integral,where="post", color="blue", label=r"val")
            counts, _ = np.histogram(Y_pred_test, bins)
            integral = 1  # counts.sum()
        else:
            bins = np.exp(np.histogram(self.Y_train[rnd_idx_train], 50)[1])
            counts, _ = np.histogram(np.exp(self.Y_train[rnd_idx_train]), bins)
            integral = 1  # counts.sum()
            plt.step(bins[:-1], counts/integral, where="post",color="green", label=r"train")
            counts, _ = np.histogram(np.exp(self.Y_test[rnd_idx_test]), bins)
            integral = 1  # counts.sum()
            plt.step(bins[:-1], counts/integral,where="post", color="blue", label=r"val")
            counts, _ = np.histogram(np.exp(Y_pred_test), bins)
            integral = 1  # counts.sum()
        plt.step(bins[:-1], counts/integral,where="post", color="red", label=r"pred")
        plt.title(r"%s" % self.fig_base_title, fontsize=10)
        if loglik:
            plt.xlabel(r"logprob ($\log\mathcal{L}+\log\mathcal{P}$)")
        else:
            plt.xlabel(r"prob ($\mathcal{L}\cdot\mathcal{P}$)")
            plt.xscale("log")
        plt.ylabel(r"counts")
        plt.legend()
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(r"%s" % figure_filename)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_filename)[-1],
                               "file path": figure_filename}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions
        print(r"%s" %figure_filename,"created and saved in",str(end-start),"s.", show=verbose)

    def plot_corners_1samp(self, X, W=None, pars=None, max_points=None, nbins=50, pars_labels="original",
                           HPDI_dic={"sample": "train", "type": "true"},
                           ranges_extend=None, title = None, color="green",
                           plot_title="Params contours", legend_labels=None, legend_loc="upper right",
                           figure_filename=None, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        linewidth = 1.3
        intervals = inference.CI_from_sigma([1, 2, 3])
        if ranges_extend == None:
            ranges = extend_corner_range(X, X, pars, 0)
        else:
            ranges = extend_corner_range(X, X, pars, ranges_extend)
        pars_labels = self.__set_pars_labels(pars_labels)
        labels = np.array(pars_labels)[pars].tolist()
        self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
        nndims = len(pars)
        if max_points != None:
            if type(max_points) == list:
                nnn = np.min([len(X), max_points[0]])
            else:
                nnn = np.min([len(X), max_points])
        else:
            nnn = len(X)
        rnd_idx = np.random.choice(np.arange(len(X)), nnn, replace=False)
        samp = X[rnd_idx][:,pars]
        if W is not None:
            W = W[rnd_idx]
        try:
            HPDI = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI_dic["type"]][HPDI_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
            #print(np.shape(HPDI1),np.shape(HPDI2))
        except:
            print("HPDI not present in predictions. Computing them.")
            HPDI = [[inference.HPDI(samp[:,i], intervals = intervals, weights=W, nbins=nbins, print_hist=False, optimize_binning=True)[interval]["Intervals"] for i in range(nndims)] for interval in intervals]
        levels = np.array([[np.sort(inference.HPD_quotas(samp[:,[i,j]], nbins=nbins, intervals = inference.CI_from_sigma([1, 2, 3]), weights=W)).tolist() for j in range(nndims)] for i in range(nndims)])
        fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
        figure = corner(samp, bins=nbins, weights=W, labels=[r"%s" % s for s in labels], 
                        fig=fig, max_n_ticks=6, color=color, plot_contours=True, smooth=True, 
                        smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False, 
                        fill_contours=False, normalize1d=True, hist_kwargs={"color": color, "linewidth": "1.5"}, 
                        label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18}, 
                        levels_lists=levels, data_kwargs={"alpha": 1}, 
                        contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
                        no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color], "alpha": 1})  
                        # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        axes = np.array(figure.axes).reshape((nndims, nndims))
        for i in range(nndims):
            title_i = ""
            ax = axes[i, i]
            #ax.axvline(value1[i], color="green",alpha=1)
            #ax.axvline(value2[i], color="red",alpha=1)
            ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=16)
            hists_1d = get_1d_hist(i, samp, nbins=nbins, ranges=ranges, weights=W, normalize1d=True)[0]  # ,intervals=HPDI681)
            HPDI68 = HPDI[i][0]
            HPDI95 = HPDI[i][1]
            HPDI3s = HPDI[i][2]
            for j in HPDI3s:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI95:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI68:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(0,1,0,.5))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color, alpha=1, linestyle="--", linewidth=linewidth)
                title_i = r"%s"%title + ": ["+"{0:1.2e}".format(j[0])+","+"{0:1.2e}".format(j[1])+"]"
            if i == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.set_title(title_i, fontsize=10)
        for yi in range(nndims):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    x1, x2, _, _ = ax.axis()
                    ax.set_xlim(x1*1.3, x2)
                ax.grid(True, linestyle="--", linewidth=1)
                ax.tick_params(axis="both", which="major", labelsize=16)
        fig.subplots_adjust(top=0.85,wspace=0.25, hspace=0.25)
        fig.suptitle(r"%s" % (plot_title+"\n\n"+self.fig_base_title), fontsize=26)
        #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
        colors = [color, "black", "black", "black"]
        red_patch = matplotlib.patches.Patch(color=colors[0])  # , label="The red data")
        #blue_patch = matplotlib.patches.Patch(color=colors[1])  # , label="The blue data")
        line1 = matplotlib.lines.Line2D([0], [0], color=colors[0], lw=int(7+2*nndims))
        line2 = matplotlib.lines.Line2D([0], [0], color=colors[1], linewidth=3, linestyle="--")
        line3 = matplotlib.lines.Line2D([0], [0], color=colors[2], linewidth=3, linestyle="-.")
        line4 = matplotlib.lines.Line2D([0], [0], color=colors[3], linewidth=3, linestyle=":")
        lines = [line1, line2, line3, line4]
        fig.legend(lines, legend_labels, fontsize=int(7+2*nndims), loc=legend_loc, bbox_to_anchor=(0.95, 0.8))#(1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        #plt.tight_layout()
        plt.savefig(figure_filename)#, dpi=200)  # ,dpi=200)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_filename)[-1],
                               "file path": figure_filename}
        print(r"%s" % figure_filename, "created and saved in", str(end-start), "s.", show=verbose)
        print("Plot done and saved in", end-start, "s.", show=verbose)

    def plot_corners_2samp(self, X1, X2, W1=None, W2=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                     HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                     ranges_extend=None, title1 = None, title2 = None,
                     color1="green", color2="red", 
                     plot_title="Params contours", legend_labels=None, legend_loc="upper right",
                     figure_filename=None, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        linewidth = 1.3
        intervals = inference.CI_from_sigma([1, 2, 3])
        if ranges_extend == None:
            ranges = extend_corner_range(X1, X2, pars, 0)
        else:
            ranges = extend_corner_range(X1, X2, pars, ranges_extend)
        pars_labels = self.__set_pars_labels(pars_labels)
        labels = np.array(pars_labels)[pars].tolist()
        self.update_figures(figure_filename=figure_filename,overwrite=overwrite)
        nndims = len(pars)
        if max_points != None:
            if type(max_points) == list:
                nnn1 = np.min([len(X1), max_points[0]])
                nnn2 = np.min([len(X2), max_points[1]])
            else:
                nnn1 = np.min([len(X1), max_points])
                nnn2 = np.min([len(X2), max_points])
        else:
            nnn1 = len(X1)
            nnn2 = len(X2)
        rnd_idx_1 = np.random.choice(np.arange(len(X1)), nnn1, replace=False)
        rnd_idx_2 = np.random.choice(np.arange(len(X2)), nnn2, replace=False)
        samp1 = X1[rnd_idx_1][:,pars]
        samp2 = X2[rnd_idx_2][:,pars]
        if W1 is not None:
            W1 = W1[rnd_idx_1]
        if W2 is not None:
            W2 = W2[rnd_idx_2]
        try:
            HPDI1 = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI1_dic["type"]][HPDI1_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
            HPDI2 = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI2_dic["type"]][HPDI2_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
            #print(np.shape(HPDI1),np.shape(HPDI2))
        except:
            print("HPDI not present in predictions. Computing them.")
            HPDI1 = [[inference.HPDI(samp1[:,i], intervals = intervals, weights=W1, nbins=nbins, print_hist=False, optimize_binning=True)[interval]["Intervals"] for i in range(nndims)] for interval in intervals]
            HPDI2 = [[inference.HPDI(samp2[:, i], intervals=intervals, weights=W2, nbins=nbins, print_hist=False, optimize_binning=True)[interval]["Intervals"] for i in range(nndims)] for interval in intervals]
        levels1 = np.array([[np.sort(inference.HPD_quotas(samp1[:,[i,j]], nbins=nbins, intervals = inference.CI_from_sigma([1, 2, 3]), weights=W1)).tolist() for j in range(nndims)] for i in range(nndims)])
        levels2 = np.array([[np.sort(inference.HPD_quotas(samp2[:, [i, j]], nbins=nbins, intervals=inference.CI_from_sigma([1, 2, 3]), weights=W2)).tolist() for j in range(nndims)] for i in range(nndims)])
        fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
        figure1 = corner(samp1, bins=nbins, weights=W1, labels=[r"%s" % s for s in labels], 
                         fig=fig, max_n_ticks=6, color=color1, plot_contours=True, smooth=True, 
                         smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False, 
                         fill_contours=False, normalize1d=True, hist_kwargs={"color": color1, "linewidth": "1.5"}, 
                         label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18}, 
                         levels_lists=levels1, data_kwargs={"alpha": 1}, 
                         contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(HPDI1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI1[0])]},
                         no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color1], "alpha": 1})  
                         # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        figure2 = corner(samp2, bins=nbins, weights=W2, labels=[r"%s" % s for s in labels], 
                         fig=fig, max_n_ticks=6, color=color2, plot_contours=True, smooth=True, 
                         range=ranges, smooth1d=True, plot_datapoints=True, plot_density=False, 
                         fill_contours=False, normalize1d=True, hist_kwargs={"color": color2, "linewidth": "1.5"}, 
                         label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels2, data_kwargs={"alpha": 1}, 
                         contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][0:len(HPDI2[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI2[0])]},
                         no_fill_contours=False, contourf_kwargs={"colors": ["white", "tomato", color2], "alpha": 1})  
                         # , quantiles = (0.16, 0.84), levels=(0.393,0.68,)), levels=[300],levels_lists=levels2)#,levels=[120])
        axes = np.array(figure1.axes).reshape((nndims, nndims))
        for i in range(nndims):
            ax = axes[i, i]
            title = ""
            #ax.axvline(value1[i], color="green",alpha=1)
            #ax.axvline(value2[i], color="red",alpha=1)
            ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=16)
            HPDI681 = HPDI1[i][0]
            HPDI951 = HPDI1[i][1]
            HPDI3s1 = HPDI1[i][2]
            HPDI682 = HPDI2[i][0]
            HPDI952 = HPDI2[i][1]
            HPDI3s2 = HPDI2[i][2]
            hists_1d_1 = get_1d_hist(i, samp1, nbins=nbins, ranges=ranges, weights=W1, normalize1d=True)[0]  # ,intervals=HPDI681)
            hists_1d_2 = get_1d_hist(i, samp2, nbins=nbins, ranges=ranges, weights=W2, normalize1d=True)[0]  # ,intervals=HPDI682)
            for j in HPDI3s1:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0], color=color1, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1], color=color1, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI3s2:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="tomato", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0], color=color2, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1], color=color2, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI951:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0], color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1], color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI952:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="tomato", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0], color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1], color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI681:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(0,1,0,.5))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0], color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1], color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                title = title+r"%s"%title1 + ": ["+"{0:1.2e}".format(j[0])+","+"{0:1.2e}".format(j[1])+"]"
            title = title+"\n"
            for j in HPDI682:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(1,0,0,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0], color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1], color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                title = title+r"%s"%title2 + ": ["+"{0:1.2e}".format(j[0])+","+"{0:1.2e}".format(j[1])+"]"
            if i == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.set_title(title, fontsize=10)
        for yi in range(nndims):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    x1, x2, _, _ = ax.axis()
                    ax.set_xlim(x1*1.3, x2)
                ax.grid(True, linestyle="--", linewidth=1)
                ax.tick_params(axis="both", which="major", labelsize=16)
        fig.subplots_adjust(top=0.85,wspace=0.25, hspace=0.25)
        fig.suptitle(r"%s" % (plot_title+"\n\n"+self.fig_base_title), fontsize=26)
        #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
        colors = [color1, color2, "black", "black", "black"]
        red_patch = matplotlib.patches.Patch(color=colors[0])  # , label="The red data")
        blue_patch = matplotlib.patches.Patch(color=colors[1])  # , label="The blue data")
        line1 = matplotlib.lines.Line2D([0], [0], color=colors[0], lw=int(7+2*nndims))
        line2 = matplotlib.lines.Line2D([0], [0], color=colors[1], lw=int(7+2*nndims))
        line3 = matplotlib.lines.Line2D([0], [0], color=colors[2], linewidth=3, linestyle="--")
        line4 = matplotlib.lines.Line2D([0], [0], color=colors[3], linewidth=3, linestyle="-.")
        line5 = matplotlib.lines.Line2D([0], [0], color=colors[4], linewidth=3, linestyle=":")
        lines = [line1, line2, line3, line4, line5]
        # (1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        fig.legend(lines, legend_labels, fontsize=int(
            7+2*nndims), loc=legend_loc, bbox_to_anchor=(0.95, 0.8))
        #plt.tight_layout()
        plt.savefig(figure_filename, dpi=50)  # ,dpi=200)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_filename)
        self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"])
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_filename)[-1],
                               "file path": figure_filename}
        print(r"%s" % figure_filename, "created and saved in", str(end-start), "s.", show=verbose)
        print("Plot done and saved in", end-start, "s.", show=verbose)

    def corner_select_data(self,string, weights):
        [W_train, W_val, W_test] = weights
        X = eval("self.X_"+string.split("_")[0])
        if "true" in string:
            W = np.ones(len(X))
        if "pred" in string:
            W = eval("W_"+string.split("_")[0])
        return [X, W]

    def reset_predictions(self, delete_figures=False, verbose=None):
        """
        Re-initializes the :attr:`Lik.predictions <DNNLikelihood.Lik.predictions>` dictionary to

         .. code-block:: python

            predictions = {"Model_evaluation": {},
                           "Bayesian_inference": {},
                           "Frequentist_inference": {},
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
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if delete_figures:
            utils.check_delete_all_files_in_path(self.output_figures_folder)
            figs = {}
            print("All predictions and figures have been deleted and the 'predictions' attribute has been initialized.")
        else:
            figs = utils.check_figures_dic(self.predictions["Figures"])
            print("All predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.")
        self.predictions = {"Model_evaluation": {},
                            "Bayesian_inference": {},
                            "Frequentist_inference": {},
                            "Figures": figs}

    def model_compute_predictions(self, 
                                  CI=inference.CI_from_sigma([inference.sigma_from_CI(0.5), 1, 2, 3]), 
                                  pars=None,
                                  batch_size=None,
                                  model_predictions={"Model_evaluation": True, 
                                                     "Bayesian_inference": False, 
                                                     "Frequentist_inference": False},
                                  plots={"plot_training_history": True,
                                         "plot_pars_coverage": True,
                                         "plot_lik_distribution": True,
                                         "plot_corners_1samp": True, # ["train_true","train_pred","val_true","val_pred","test_true","test_pred"],
                                         "plot_corners_2samp": True, #[["train_true", "train_pred"], ["train_true", "test_pred"]]},
                                         "plot_tmu_sources_1d": True},
                                  model_predict_kwargs={}, # batch_size=None, steps=None, x_boundaries=False, y_boundaries=False, save_log=True, verbose=None
                                  HPDI_kwargs={}, # intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True
                                  frequentist_inference_options={"input_likelihood_file": None,
                                                                 "use_reference_likelihood": False,
                                                                 "compute_maximum_likelihood_kwargs": False,
                                                                 "compute_profiled_maxima_likelihood_kwargs": False,
                                                                 "compute_maximum_sample_kwargs": False,
                                                                 "compute_profiled_maxima_sample_kwargs": False,
                                                                 "compute_maximum_model_kwargs": False,
                                                                 "compute_profiled_maxima_model_kwargs": False
                                                                },
                                  plot_training_history_kwargs = {}, # metrics=["loss"], yscale="log", show_plot=False, overwrite=False, verbose=None
                                  plot_pars_coverage_kwargs = {}, # pars=None, loglik=True, show_plot=False, overwrite=False, verbose=None
                                  plot_lik_distribution_kwargs = {}, # loglik=True, show_plot=False, overwrite=False, verbose=None
                                  plot_corners_1samp_kwargs={},   # W=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                                                                  # HPDI_dic={"sample": "train", "type": "true"},
                                                                  # ranges_extend=None, title = None, color="green",
                                                                  # plot_title="Params contours", legend_labels=None, 
                                                                  # figure_filename=None, show_plot=False, overwrite=False, verbose=None
                                  plot_corners_2samp_kwargs={},   # W1=None, W2=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                                                                  # HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                                                                  # ranges_extend=None, title1 = None, title2 = None,
                                                                  # color1="green", color2="red", 
                                                                  # plot_title="Params contours", legend_labels=None, 
                                                                  # figure_filename=None, show_plot=False, overwrite=False, verbose=None
                                  # sources=None, timestamp=None, show_plot=False, overwrite=False, verbose=None
                                  plot_tmu_sources_1d_kwargs={},  # sources=["likelihood","model","train","val","test"], timestamp=None, show_plot=False,overwrite=False, verbose=None
                                  overwrite=False,
                                  verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if not np.any(list(model_predictions.values())):
            print("No predictions to compute. Plese select one or more through the 'model_predictions' input argument.")
            return
        def model_predict_sub(X):
            return self.model_predict(X, batch_size=self.batch_size, verbose=verbose_sub,**model_predict_kwargs)
        def HPDI_sub(X,CI,weights=None):
            return inference.HPDI(X, CI, weights=weights, **HPDI_kwargs)
        start_global = timer()
        if pars == None:
            pars = self.data.pars_pos_poi.tolist()
        else:
            pars = pars
        if batch_size == None:
            batch_size = self.batch_size
        # Determine which predictions are requested
        utils.check_set_dict_keys(model_predictions, ["Model_evaluation",
                                                      "Bayesian_inference",
                                                      "Frequentist_inference"],
                                                     [True,False,False],verbose=verbose_sub)
        # Save model for predictions

        # Metrics evaluation
        if model_predictions["Model_evaluation"]:
            print("Performing Model evaluation", show=verbose)
            if len(self.X_train) <= 1:
                print("Generating train/val data", show=verbose)
                self.generate_train_data(verbose=verbose_sub)
            if len(self.X_test) <= 1:
                print("Generating test data", show=verbose)
                self.generate_test_data(verbose=verbose_sub)
                self.save_data_indices(overwrite=True,verbose=verbose_sub)
            start = timer()
            utils.check_set_dict_keys(self.predictions["Model_evaluation"],
                                      ["Metrics_on_scaled_data","Prediction_time","Metrics_on_unscaled_data"], 
                                      [{},{},{}], verbose=verbose_sub)
            print("Evaluating all metrics on (scaled) train/val/test using best models", show=verbose)
            metrics_names = self.model.metrics_names
            metrics_names_train = [i+"_best" for i in self.model.metrics_names]
            metrics_names_val = ["val_"+i+"_best" for i in self.model.metrics_names]
            metrics_names_test = ["test_"+i+"_best" for i in self.model.metrics_names]
            metrics_train = self.model_evaluate(self.X_train, self.Y_train, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
            metrics_val = self.model_evaluate(self.X_val, self.Y_val, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
            metrics_test = self.model_evaluate(self.X_test, self.Y_test, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
            metrics_true = {**dict(zip(metrics_names_train, metrics_train)), **dict(zip(metrics_names_val, metrics_val)), **dict(zip(metrics_names_test, metrics_test))}
            self.predictions["Model_evaluation"]["Metrics_on_scaled_data"][timestamp] = metrics_true
            print("Predicting Y for train/val/test samples", show=verbose)
            Y_pred_train, prediction_time1 = model_predict_sub(self.X_train)
            Y_pred_val, prediction_time2 = model_predict_sub(self.X_val)
            Y_pred_test, prediction_time3 = model_predict_sub(self.X_test)
            self.predictions["Model_evaluation"]["Prediction_time"][timestamp] = (prediction_time1+prediction_time2+prediction_time3)/3
            print("Evaluating all metrics on (un-scaled) trai[timestamp]n/val/test using best models", show=verbose)
            metrics_names_train = [i+"_best_unscaled" for i in self.model.metrics_names]
            metrics_names_val = ["val_"+i+"_best_unscaled" for i in self.model.metrics_names]
            metrics_names_test = ["test_"+i +"_best_unscaled" for i in self.model.metrics_names]
            metrics_train_unscaled = [metric(self.Y_train,Y_pred_train).numpy() for metric in [self.loss]+self.metrics]
            metrics_val_unscaled = [metric(self.Y_val,Y_pred_val).numpy() for metric in [self.loss]+self.metrics]
            metrics_test_unscaled = [metric(self.Y_test,Y_pred_test).numpy() for metric in [self.loss]+self.metrics]
            metrics_unscaled = {**dict(zip(metrics_names_train, metrics_train_unscaled)), **dict(zip(metrics_names_val, metrics_val_unscaled)), **dict(zip(metrics_names_test, metrics_test_unscaled))}
            self.predictions["Model_evaluation"]["Metrics_on_unscaled_data"][timestamp] = metrics_unscaled
            end = timer()
            print("Prediction on ("+str(self.npoints_train)+","+str(self.npoints_val)+","+str(self.npoints_test)+")", "(train,val,test) points done in", str(end-start), "s.", show=verbose)
        # Bayesian inference
        if model_predictions["Bayesian_inference"]:
            print("Computing Bayesian inference benchmarks", show=verbose)
            if len(self.X_train) <= 1:
                print("Generating train/val data", show=verbose)
                self.generate_train_data(verbose=verbose_sub)
            if len(self.X_test) <= 1:
                print("Generating test data", show=verbose)
                self.generate_test_data(verbose=verbose_sub)
                self.save_data_indices(overwrite=True, verbose=verbose_sub)
            start = timer()
            self.predictions["Bayesian_inference"][timestamp] = {}
            try:
                [Y_pred_train, Y_pred_val, Y_pred_test]
            except:
                utils.check_set_dict_keys(self.predictions["Model_evaluation"],
                                          ["Prediction_time"], 
                                          [{}], verbose=verbose_sub)
                print("Predicting Y for train/val/test samples", show=verbose)
                Y_pred_train, prediction_time1 = model_predict_sub(self.X_train)
                Y_pred_val, prediction_time2 = model_predict_sub(self.X_val)
                Y_pred_test, prediction_time3 = model_predict_sub(self.X_test)
                self.predictions["Model_evaluation"]["Prediction_time"][timestamp] = (prediction_time1+prediction_time2+prediction_time3)/3
            print("Computing exp(Y_true) and exp(Y_pred) for train/val/test samples", show=verbose)
            [Y_train_exp, Y_val_exp, Y_test_exp, Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [np.exp(Y) for Y in [self.Y_train, self.Y_val, self.Y_test, Y_pred_train, Y_pred_val, Y_pred_test]]
            print("Computing weights (pred vs true) for reweighting of distributions", show=verbose)
            [W_train, W_val, W_test] = [utils.normalize_weights(W) for W in [Y_pred_train_exp/Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]]
            print("Computing HPDI (pred vs true) using reweighted distributions", show=verbose)
            #(data, intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True)
            HPDI_result = {}
            for par in pars:
                [HPDI_train, HPDI_val, HPDI_test] = [HPDI_sub(X, CI) for X in [self.X_train[:, par], self.X_val[:, par], self.X_test[:, par]]]
                [HPDI_pred_train, HPDI_pred_val, HPDI_pred_test] = [HPDI_sub(self.X_train[:, par], CI, W_train), HPDI_sub(self.X_val[:, par], CI, W_val), HPDI_sub(self.X_test[:, par], CI, W_test)]
                HPDI_result[par] = {"true": {"train": HPDI_train, "val": HPDI_val, "test": HPDI_test}, "pred":{"train": HPDI_pred_train, "val": HPDI_pred_val, "test": HPDI_pred_test}}
            HDPI_error = inference.HPDI_error(HPDI_result)
            utils.check_set_dict_keys(self.predictions["Bayesian_inference"], 
                                      ["HPDI","HPDI_error","KS","KS_medians"], 
                                      [{},{},{},{}], verbose=verbose_sub)
            self.predictions["Bayesian_inference"]["HPDI"][timestamp] = HPDI_result
            self.predictions["Bayesian_inference"]["HPDI_error"][timestamp] = HDPI_error
            print("Computing KS test between one-dimensional distributions (pred vs true) using reweighted distributions", show=verbose)
            KS_test_pred_train = [inference.ks_w(self.X_test[:, q], self.X_train[:, q], np.ones(len(self.X_test)), W_train) for q in range(len(self.X_train[0]))]
            KS_test_pred_val = [inference.ks_w(self.X_test[:, q], self.X_val[:, q], np.ones(len(self.X_test)), W_val) for q in range(len(self.X_train[0]))]
            KS_val_pred_test = [inference.ks_w(self.X_val[:, q], self.X_test[:, q], np.ones(len(self.X_val)), W_test) for q in range(len(self.X_train[0]))]
            KS_train_pred_train = [inference.ks_w(self.X_train[:, q], self.X_train[:, q], np.ones(len(self.X_train)), W_train) for q in range(len(self.X_train[0]))]
            KS_test_pred_train_median = np.median(np.array(KS_test_pred_train)[:, 1]).tolist()
            KS_test_pred_val_median = np.median(np.array(KS_test_pred_val)[:, 1]).tolist()
            KS_val_pred_test_median = np.median(np.array(KS_val_pred_test)[:, 1]).tolist()
            KS_train_pred_train_median = np.median(np.array(KS_train_pred_train)[:, 1]).tolist()
            self.predictions["Bayesian_inference"]["KS"][timestamp] = {"Test_vs_pred_on_train": KS_test_pred_train,
                                                                       "Test_vs_pred_on_val": KS_test_pred_val,
                                                                       "Val_vs_pred_on_test": KS_val_pred_test,
                                                                       "Train_vs_pred_on_train": KS_train_pred_train}
            self.predictions["Bayesian_inference"]["KS_medians"][timestamp] = {"Test_vs_pred_on_train": KS_test_pred_train_median,
                                                                               "Test_vs_pred_on_val": KS_test_pred_val_median,
                                                                               "Val_vs_pred_on_test": KS_val_pred_test_median,
                                                                               "Train_vs_pred_on_train": KS_train_pred_train_median}                                          
            end = timer()
            print("Bayesian inference benchmarks computed in", str(end-start), "s.", show=verbose)
        # Frequentist inference
        if model_predictions["Frequentist_inference"]:
            # Determine required frequentist inference
            utils.check_set_dict_keys(frequentist_inference_options, ["input_likelihood_file",
                                                                      "compute_maximum_likelihood_kwargs",
                                                                      "compute_profiled_maxima_likelihood_kwargs",
                                                                      "compute_maximum_sample_kwargs",
                                                                      "compute_profiled_maxima_sample_kwargs",
                                                                      "compute_maximum_model_kwargs",
                                                                      "compute_profiled_maxima_model_kwargs"],
                                                                      [None,False,False,False,False,False,False],verbose=False)
            if frequentist_inference_options["compute_maximum_likelihood_kwargs"] is True:
                frequentist_inference_options["compute_maximum_likelihood_kwargs"] = {}
            if frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"] is True:
                frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"] = {}
            if frequentist_inference_options["compute_maximum_sample_kwargs"] is True:
                frequentist_inference_options["compute_maximum_sample_kwargs"] = {}
            if frequentist_inference_options["compute_profiled_maxima_sample_kwargs"] is True:
                frequentist_inference_options["compute_profiled_maxima_sample_kwargs"] = {}
            if frequentist_inference_options["compute_maximum_model_kwargs"] is True:
                frequentist_inference_options["compute_maximum_model_kwargs"] = {}
            if frequentist_inference_options["compute_profiled_maxima_model_kwargs"] is True:
                frequentist_inference_options["compute_profiled_maxima_model_kwargs"] = {}
            print("Computing Frequentist inference benchmarks based on 'frequentist_inference_options'.", show=verbose)
            start = timer()
            # Frequentist Inference from reference Likelihood object
            if frequentist_inference_options["compute_maximum_likelihood_kwargs"] is not False or frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"] is not False:
                # Check for a reference likelihood
                if self.likelihood is not None:
                    print("A reference Likelihood object is already available. frequentist_inference_options[\"input_likelihood_file\"] input will be ignored.", show=verbose)
                else:
                    if self.input_likelihood_file is None:
                        self.input_likelihood_file = frequentist_inference_options["input_likelihood_file"]
                    self.__set_likelihood(verbose=verbose_sub)
                # Compute maximum Likelihood
                if self.likelihood is not None:
                    print("Computing Frequentist inference benchmarks for reference Likelihood", show=verbose)
                    if frequentist_inference_options["compute_maximum_likelihood_kwargs"] is not False:
                        print("Computing global maximum for reference Likelihood", show=verbose)
                        utils.check_set_dict_keys(frequentist_inference_options["compute_maximum_likelihood_kwargs"], 
                                                  ["pars_init",
                                                   "optimizer",
                                                   "timestamp",
                                                   "save",
                                                   "verbose"],
                                                  [None,{},timestamp,False,verbose_sub],verbose=verbose_sub)
                        string = "self.likelihood.compute_maximum_logpdf("
                        for key, value in frequentist_inference_options["compute_maximum_likelihood_kwargs"].items():
                            if type(value) == str:
                                value = "'"+value+"'"
                            string = string + str(key)+"="+str(value)+", "
                        string = str(string + ")").replace(", )", ")")
                        print(string)
                        eval(string)
                        utils.check_set_dict_keys(self.predictions["Frequentist_inference"],
                                                  ["logpdf_max_likelihood"],
                                                  [{}],verbose=verbose_sub)
                        utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_max_likelihood"],
                                                  [timestamp],
                                                  [{}], verbose=verbose_sub)
                        self.predictions["Frequentist_inference"]["logpdf_max_likelihood"][timestamp] = self.likelihood.predictions["logpdf_max"][timestamp]
                    if frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"] is not False:
                        print("Computing profiled maxima for reference Likelihood", show=verbose)
                        utils.check_set_dict_keys(frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"],
                                                  ["pars",
                                                   "pars_ranges",
                                                   "pars_init",
                                                   "pars_bounds",
                                                   "spacing",
                                                   "optimizer",
                                                   "timestamp",
                                                   "progressbar",
                                                   "save",
                                                   "verbose"],
                                                  [pars,np.full([len(pars),3],[-1,1,2]).tolist(),
                                                   None,None,"grid",{},timestamp,True,False,verbose_sub],verbose=verbose_sub)
                        string = "self.likelihood.compute_profiled_maxima_logpdf("
                        for key, value in frequentist_inference_options["compute_profiled_maxima_likelihood_kwargs"].items():
                            if type(value) == str:
                                value = "'"+value+"'"
                            string = string + str(key)+"="+str(value)+", "
                        string = str(string + ")").replace(", )", ")")
                        print(string)
                        eval(string)
                        utils.check_set_dict_keys(self.predictions["Frequentist_inference"],
                                                  ["logpdf_profiled_max_likelihood"],
                                                  [{}],verbose=verbose_sub)
                        utils.check_set_dict_keys(self.predictions["Frequentist_inference"]["logpdf_profiled_max_likelihood"],
                                                  [timestamp],
                                                  [{}],verbose=verbose_sub)
                        self.predictions["Frequentist_inference"]["logpdf_profiled_max_likelihood"][timestamp] = self.likelihood.predictions["logpdf_profiled_max"][timestamp]
            # Frequentist Inference from samples
            if frequentist_inference_options["compute_maximum_sample_kwargs"] is not False or frequentist_inference_options["compute_profiled_maxima_sample_kwargs"] is not False:
                print("Computing Frequentist inference benchmarks from samples", show=verbose)
                if frequentist_inference_options["compute_maximum_sample_kwargs"] is not False:
                    print("Computing global maximum from samples", show=verbose)
                    utils.check_set_dict_keys(frequentist_inference_options["compute_maximum_sample_kwargs"], 
                                              ["samples",
                                               "timestamp",
                                               "save",
                                               "verbose"],
                                              [["train","val","test"],timestamp,False,verbose_sub],verbose=verbose_sub)
                    string = "self.compute_maximum_sample("
                    for key, value in frequentist_inference_options["compute_maximum_sample_kwargs"].items():
                        if type(value) == str:
                            value = "'"+value+"'"
                        string = string + str(key)+"="+str(value)+", "
                    string = str(string + ")").replace(", )", ")")
                    print(string)
                    eval(string)
                if frequentist_inference_options["compute_profiled_maxima_sample_kwargs"] is not False:
                    print("Computing profiled maxima from samples", show=verbose)
                    utils.check_set_dict_keys(frequentist_inference_options["compute_profiled_maxima_sample_kwargs"],
                                              ["pars",
                                               "pars_ranges",
                                               "samples",
                                               "spacing",
                                               "binwidths",
                                               "x_boundaries",
                                               "timestamp",
                                               "progressbar",
                                               "save",
                                               "verbose"],
                                              [pars,np.full([len(pars),3],[-1,1,2]).tolist(),["train", "val", "test"],"grid","auto",False,timestamp,True,False, verbose_sub],verbose=verbose_sub)
                    string = "self.compute_profiled_maxima_sample("
                    for key, value in frequentist_inference_options["compute_profiled_maxima_sample_kwargs"].items():
                        if type(value) == str:
                            value = "'"+value+"'"
                        string = string + str(key)+"="+str(value)+", "
                    string = str(string + ")").replace(", )", ")")
                    print(string)
                    eval(string)
            # Frequentist Inference from model
            if frequentist_inference_options["compute_maximum_model_kwargs"] is not False or frequentist_inference_options["compute_profiled_maxima_model_kwargs"] is not False:
                print("Computing Frequentist inference benchmarks from model", show=verbose)
                if frequentist_inference_options["compute_maximum_model_kwargs"] is not False:
                    print("Computing global maximum from model", show=verbose)
                    utils.check_set_dict_keys(frequentist_inference_options["compute_maximum_model_kwargs"], 
                                              ["pars_init",
                                               "optimizer",
                                               "x_boundaries",
                                               "y_boundaries",
                                               "timestamp",
                                               "save",
                                               "verbose"],
                                              [None,{},False,False,timestamp,False,verbose_sub],verbose=verbose_sub)
                    string = "self.compute_maximum_model("
                    for key, value in frequentist_inference_options["compute_maximum_model_kwargs"].items():
                        if type(value) == str:
                            value = "'"+value+"'"
                        string = string + str(key)+"="+str(value)+", "
                    string = str(string + ")").replace(", )", ")")
                    print(string)
                    eval(string)
                if frequentist_inference_options["compute_profiled_maxima_model_kwargs"] is not False:
                    print("Computing profiled maxima from model", show=verbose)
                    utils.check_set_dict_keys(frequentist_inference_options["compute_profiled_maxima_model_kwargs"],
                                              ["pars",
                                               "pars_ranges",
                                               "pars_init",
                                               "optimizer",
                                               "spacing",
                                               "x_boundaries",
                                               "y_boundaries",
                                               "timestamp",
                                               "progressbar",
                                               "save",
                                               "verbose"],
                                              [pars,np.full([len(pars),3],[-1,1,2]).tolist(),None,{},"grid",False,False,timestamp,True,False,verbose_sub],verbose=verbose_sub)
                    string = "self.compute_profiled_maxima_model("
                    for key, value in frequentist_inference_options["compute_profiled_maxima_model_kwargs"].items():
                        if type(value) == str:
                            value = "'"+value+"'"
                        string = string + str(key)+"="+str(value)+", "
                    string = str(string + ")").replace(", )", ")")
                    print(string)
                    eval(string)
            end = timer()
            print("Frequentist inference benchmarks computed in", str(end-start), "s.", show=verbose)
        # Sort nested dictionary by keys
        self.predictions = utils.sort_dict(self.predictions)
        self.save_predictions(timestamp=timestamp,overwrite=overwrite,verbose=verbose_sub)
        self.generate_summary_text(model_predictions=model_predictions,timestamp=timestamp)
        self.generate_fig_base_title()
        if np.any(list(plots.values())):
            start = timer()
            ## Check plots input arguments
            utils.check_set_dict_keys(plots, ["plot_training_history",
                                              "plot_pars_coverage",
                                              "plot_lik_distribution",
                                              "plot_corners_1samp",
                                              "plot_corners_2samp",
                                              "plot_tmu_sources_1d"],
                                             [True,True,True,True,True,False],verbose=verbose_sub)
            if model_predictions["Model_evaluation"]:
                # Make plots for model evaluation
                print("Making plots for model evaluation.", show=verbose)
                if plots["plot_training_history"]:
                    self.plot_training_history(timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub, **plot_training_history_kwargs)
                if plots["plot_pars_coverage"]:
                    self.plot_pars_coverage(pars=pars, timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub, **plot_pars_coverage_kwargs)
                if plots["plot_lik_distribution"]:
                    self.plot_lik_distribution(timestamp=timestamp, overwrite=overwrite,verbose=verbose_sub, **plot_lik_distribution_kwargs)
            if model_predictions["Bayesian_inference"]:
                # Make corner plots for Bayesian inference
                print("Making corner plots for Bayesian inference.", show=verbose)
                #### 1sample corner plots
                ## **corners_kwargs should include ranges_extend, max_points, nbins, show_plot, overwrite
                if type(plots["plot_corners_1samp"]) == bool:
                    if plots["plot_corners_1samp"]:
                        plots["plot_corners_1samp"] = ["train_true","train_pred","val_true","val_pred","test_true","test_pred"]
                    else:
                        plots["plot_corners_1samp"] = []
                if type(plots["plot_corners_2samp"]) == bool:
                    if plots["plot_corners_2samp"]:
                        plots["plot_corners_2samp"] = [["train_true", "train_pred"],["test_true", "test_pred"], ["train_true", "test_pred"]]
                    else:
                        plots["plot_corners_2samp"] = []
                if len(plots["plot_corners_1samp"])>0:
                    for string in plots["plot_corners_1samp"]:
                        [X, W] = self.corner_select_data(string,[W_train, W_val, W_test])
                        self.plot_corners_1samp(X, W=W,
                                                HPDI_dic={"sample": string.split("_")[0], "type": string.split("_")[1]},
                                                pars = pars, pars_labels = "original",
                                                title="$68\%$ HPDI - sample: "+string.split("_")[0], color="green",
                                                plot_title="Sample: "+string,
                                                legend_labels=["Sample: " + string + r" ($%s$ points)" % utils.latex_float(len(X)),
                                                               r"$68.27\%$ HPDI", 
                                                               r"$95.45\%$ HPDI", 
                                                               r"$99.73\%$ HPDI"],
                                                figure_filename=self.output_figures_base_file+"_corner_pars_"+string+".pdf",
                                                timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub, **plot_corners_1samp_kwargs)
                #### 2sample corner plots
                if len(plots["plot_corners_2samp"])>0:
                    for strings in plots["plot_corners_2samp"]:
                        string1=strings[0]
                        string2=strings[1]
                        [X1, W1] = self.corner_select_data(string1,[W_train, W_val, W_test])
                        [X2, W2] = self.corner_select_data(string2,[W_train, W_val, W_test])
                        self.plot_corners_2samp(X1, X2, W1=W1, W2=W2,
                                                HPDI1_dic={"sample": string1.split("_")[0], "type": string1.split("_")[1]}, 
                                                HPDI2_dic={"sample": string2.split("_")[0], "type": string2.split("_")[1]},
                                                pars = pars, pars_labels = "original",
                                                title1 = "$68\%$ HPDI - sample: "+string1.split("_")[0], 
                                                title2 ="$68\%$ HPDI - sample: "+string2.split("_")[0],
                                                color1 = "green", color2 = "red",
                                                plot_title = "Samples: "+string1+" vs "+string2,
                                                legend_labels = ["Sample: " + string1 + r" ($%s$ points)" % utils.latex_float(len(X1)),
                                                                 "Sample: " + string2 + r" ($%s$ points)" % utils.latex_float(len(X2)),
                                                                 r"$68.27\%$ HPDI", 
                                                                 r"$95.45\%$ HPDI", 
                                                                 r"$99.73\%$ HPDI"],
                                                figure_filename=self.output_figures_base_file+"_corner_pars_"+string1+"_vs_"+string2+".pdf",
                                                timestamp=timestamp, overwrite=overwrite, verbose=verbose_sub, **plot_corners_2samp_kwargs)
            if model_predictions["Frequentist_inference"]:
                # Make tmu plot for Frequentist inference
                print("Making t_mu plot for Frequentist inference.", show=verbose)
                if plots["plot_tmu_sources_1d"]:
                    self.plot_tmu_sources_1d(timestamp=timestamp,overwrite=overwrite,verbose=verbose_sub,**plot_tmu_sources_1d_kwargs)
            end = timer()
            print("All plots done in", str(end-start), "s.", show=verbose)
        self.predictions["Figures"] = utils.check_figures_dic(
            self.predictions["Figures"])
        #self.save_summary_json(overwrite=overwrite, verbose=verbose_sub)
        end_global = timer()
        self.log[timestamp] = {"action": "computed predictions",
                               "CI": CI,
                               "pars": pars,
                               "batch size": batch_size,
                               "model_predictions": model_predictions,
                               "plots": plots,
                               "model_predict_kwargs": model_predict_kwargs,
                               "HPDI_kwargs": HPDI_kwargs,
                               "frequentist_inference_options": frequentist_inference_options,
                               "plot_training_history_kwargs": plot_training_history_kwargs,
                               "plot_pars_coverage_kwargs": plot_pars_coverage_kwargs,
                               "plot_lik_distribution_kwargs": plot_lik_distribution_kwargs,
                               "plot_corners_1samp_kwargs": plot_corners_1samp_kwargs,
                               "plot_corners_2samp_kwargs": plot_corners_2samp_kwargs,
                               "plot_tmu_sources_1d_kwargs": plot_tmu_sources_1d_kwargs}
        self.save(timestamp=timestamp,overwrite=overwrite, verbose=verbose_sub)
        print("All predictions done in",end_global-start_global,"s.", show=verbose)
        #[tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02,
        #    tmu_err_mean] = ["None", "None", "None", "None", "None", "None", "None"]
        #if FREQUENTISTS_RESULTS:
        #     print("Estimating frequentist inference")
        #     start_tmu = timer()
        #     blst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #     tmuexact = np.array(list(map(tmu, blst)))
        #     tmuDNN = np.array(list(map(lambda x: tmu_DNN(x, model, scalerX, scalerY), blst)))
        #     [tmusample001, tmusample005, tmusample01, tmusample02] = [np.array(list(map(lambda x: tmu_sample(x, X_train, Y_train, binsize), blst))) for binsize in [0.01, 0.05, 0.1, 0.2]]
        #     tmu_err_mean = np.mean(np.abs(tmuexact[:, -1]-tmuDNN[:, -1]))
        #     end_tmu = timer()
        #     print("Frequentist inference done in", start_tmu-end_tmu, "s.")
        #end_global = timer()
        #print("Total time for predictions:", end_global-start_global, "s")
        #return [metrics_true, metrics_scaled,
        #        #mean_error_train, mean_error_val, mean_error_test, min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
        #        #quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
        #        #one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
        #        #central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test,
        #        HPDI_train, HPDI_val, HPDI_test, HPDI_pred_train, HPDI_pred_val, HPDI_pred_test, one_sigma_HPDI_rel_err_train, one_sigma_HPDI_rel_err_val, one_sigma_HPDI_rel_err_test, one_sigma_HPDI_rel_err_train_test,
        #        KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_pred_train, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_pred_train_median,
        #        tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time]

    def save_log(self, timestamp=None, overwrite=False, verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_log_file = utils.generate_dump_file_path(self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print("DnnLik log file", output_log_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("DnnLik log file", output_log_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("DnnLik log file dump", output_log_file, "saved in", str(end-start), "s.", show=verbose)

    def save_data_indices(self, timestamp=None, overwrite=False, verbose=None):
        """ Save indices to member_n_idx.h5 as h5 file
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_idx_h5_file = self.output_idx_h5_file
            if not overwrite:
                utils.check_rename_file(output_idx_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_idx_h5_file = utils.generate_dump_file_path(self.output_idx_h5_file, timestamp=timestamp)
        #self.close_opened_dataset(verbose=verbose_sub)
        utils.check_delete_files(output_idx_h5_file)
        h5_out = h5py.File(output_idx_h5_file, "w")
        h5_out.require_group(self.name)
        data = h5_out.require_group("idx")
        data["idx_train"] = self.idx_train
        data["idx_val"] = self.idx_val
        data["idx_test"] = self.idx_test
        h5_out.close()
        end = timer()
        self.log[timestamp] = {"action": "saved indices",
                               "file name": path.split(output_idx_h5_file)[-1],
                               "file path": output_idx_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Idx h5 file", output_idx_h5_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Idx h5 file", output_idx_h5_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Idx h5 file dump", output_idx_h5_file, "saved in", str(end-start), "s.", show=verbose)

    def save_model_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save model to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_tf_model_json_file = self.output_tf_model_json_file
            if not overwrite:
                utils.check_rename_file(output_tf_model_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_tf_model_json_file = utils.generate_dump_file_path(self.output_tf_model_json_file, timestamp=timestamp)
        try:
            model_json = self.model.to_json()
        except:
            print("Model not defined. No file is saved.")
            return
        with open(output_tf_model_json_file, "w") as json_file:
            json_file.write(model_json)
        end = timer()
        
        self.log[timestamp] = {"action": "saved tf model json",
                               "file name": path.split(output_tf_model_json_file)[-1],
                               "file path": output_tf_model_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Mode json file", output_tf_model_json_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Mode json file", output_tf_model_json_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Mode json file dump", output_tf_model_json_file, "saved in", str(end-start), "s.", show=verbose)

    def save_model_h5(self, timestamp=None, overwrite=False, verbose=None):
        """ Save model to h5
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_tf_model_h5_file = self.output_tf_model_h5_file
            if not overwrite:
                utils.check_rename_file(output_tf_model_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_tf_model_h5_file = utils.generate_dump_file_path(self.output_tf_model_h5_file, timestamp=timestamp)
        try:
            self.model.save(output_tf_model_h5_file)
        except:
            print("Model not defined. No file is saved.")
            return
        end = timer()
        self.log[timestamp] = {"action": "saved tf model h5",
                               "file name": path.split(output_tf_model_h5_file)[-1],
                               "file path": output_tf_model_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Mode h5 file", output_tf_model_h5_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Mode h5 file", output_tf_model_h5_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Mode h5 file dump", output_tf_model_h5_file, "saved in", str(end-start), "s.", show=verbose)

    def save_model_onnx(self, timestamp=None, overwrite=False, verbose=None):
        """ Save model to onnx
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_tf_model_onnx_file = self.output_tf_model_onnx_file
            if not overwrite:
                utils.check_rename_file(output_tf_model_onnx_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_tf_model_onnx_file = utils.generate_dump_file_path(self.output_tf_model_onnx_file, timestamp=timestamp)
        try:
            onnx_model = keras2onnx.convert_keras(self.model, self.name)
        except:
            print("Model not defined. No file is saved.")
            return
        onnx.save_model(onnx_model, output_tf_model_onnx_file)
        end = timer()
        self.log[timestamp] = {"action": "saved tf model onnx",
                               "file name": path.split(output_tf_model_onnx_file)[-1],
                               "file path": output_tf_model_onnx_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Mode onnx file", output_tf_model_onnx_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Mode onnx file", output_tf_model_onnx_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Mode onnx file dump", output_tf_model_onnx_file, "saved in", str(end-start), "s.", show=verbose)

    def save_history_json(self, timestamp=None,overwrite=False,verbose=None):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_history_json_file = self.output_history_json_file
            if not overwrite:
                utils.check_rename_file(output_history_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_history_json_file = utils.generate_dump_file_path(self.output_history_json_file, timestamp=timestamp)
        #for key in list(history.keys()):
        #    self.history[utils.metric_name_abbreviate(key)] = self.history.pop(key)
        dictionary = utils.convert_types_dict(self.history)
        with codecs.open(output_history_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), sort_keys=True, indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved history json",
                               "file name": path.split(output_history_json_file)[-1],
                               "file path": output_history_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Mode history file", output_history_json_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Mode history file", output_history_json_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Mode history file dump", output_history_json_file, "saved in", str(end-start), "s.", show=verbose)

    def save_summary_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_summary_json_file = self.output_summary_json_file
            if not overwrite:
                utils.check_rename_file(output_summary_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_summary_json_file = utils.generate_dump_file_path(self.output_summary_json_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__,["_DnnLik__resources_inputs",
                                                         "callbacks","data","history",
                                                         "idx_test","idx_train","idx_val",
                                                         "input_files_base_name","input_history_json_file",
                                                         "input_idx_h5_file","input_log_file",
                                                         "input_predictions_h5_file",
                                                         "input_scalers_pickle_file","input_summary_json_file",
                                                         "input_tf_model_h5_file","likelihood","load_on_RAM",
                                                         "log","loss","metrics","model","optimizer",
                                                         "predictions", "scalerX","scalerY","verbose",
                                                         "X_test","X_train","X_val","Y_test","Y_train","Y_val","W_train"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(output_summary_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved summary json",
                               "file name": path.split(output_summary_json_file)[-1],
                               "file path": output_summary_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Summary json file", output_summary_json_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Summary json file", output_summary_json_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Summary json file dump", output_summary_json_file, "saved in", str(end-start), "s.", show=verbose)

    def generate_summary_text(self,model_predictions={},timestamp=None,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        utils.check_set_dict_keys(model_predictions, ["Model_evaluation",
                                                      "Bayesian_inference",
                                                      "Frequentist_inference"],
                                  [False, False, False], verbose=verbose_sub)
        summary_text = "Sample file: " + str(path.split(self.input_data_file)[1].replace("_", r"$\_$")) + "\n"
        summary_text = summary_text + "Layers: " + utils.string_add_newline_at_char(str(self.hidden_layers),",") + "\n"
        summary_text = summary_text + "Pars: " + str(self.ndims) + "\n"
        summary_text = summary_text + "Trainable pars: " + str(self.model_trainable_params) + "\n"
        summary_text = summary_text + "Scaled X/Y: " + str(self.scalerX_bool) +"/"+ str(self.scalerY_bool) + "\n"
        summary_text = summary_text + "Dropout: " + str(self.dropout_rate) + "\n"
        summary_text = summary_text + "AF out: " + self.act_func_out_layer + "\n"
        summary_text = summary_text + "Batch norm: " + str(self.batch_norm) + "\n"
        summary_text = summary_text + "Loss: " + self.loss_string + "\n"
        summary_text = summary_text + "Optimizer: " + utils.string_add_newline_at_char(self.optimizer_string,",").replace("_", "-") + "\n"
        summary_text = summary_text + "Batch size: " + str(self.batch_size) + "\n"
        summary_text = summary_text + "Epochs: " + str(self.epochs_available) + "\n"
        summary_text = summary_text + "GPU(s): " + utils.string_add_newline_at_char(str(self.training_device),",") + "\n"
        if model_predictions["Model_evaluation"]:
            try:
                metrics_scaled = self.predictions["Model_evaluation"]["Metrics_on_scaled_data"][timestamp]
                summary_text = summary_text + "Best losses: " + "[" + "{0:1.2e}".format(metrics_scaled["loss_best"]) + "," + \
                                                                      "{0:1.2e}".format(metrics_scaled["val_loss_best"]) + "," + \
                                                                      "{0:1.2e}".format(metrics_scaled["test_loss_best"]) + "]" + "\n"
            except:
                pass
            try:
                metrics_unscaled = self.predictions["Model_evaluation"]["Metrics_on_unscaled_data"][timestamp]
                summary_text = summary_text + "Best losses scaled: " + "[" + "{0:1.2e}".format(metrics_unscaled["loss_best_unscaled"]) + "," + \
                                                                             "{0:1.2e}".format(metrics_unscaled["val_loss_best_unscaled"]) + "," + \
                                                                             "{0:1.2e}".format(metrics_unscaled["test_loss_best_unscaled"]) + "]" + "\n"
            except:
                pass
        if model_predictions["Bayesian_inference"]:
            try:
                ks_medians = self.predictions["Bayesian_inference"]["KS_medians"][timestamp]
                summary_text = summary_text + "KS $p$-median: " + "[" + "{0:1.2e}".format(ks_medians["Test_vs_pred_on_train"]) + "," + \
                                                                    "{0:1.2e}".format(ks_medians["Test_vs_pred_on_val"]) + "," + \
                                                                    "{0:1.2e}".format(ks_medians["Val_vs_pred_on_test"]) + "," + \
                                                                    "{0:1.2e}".format(ks_medians["Train_vs_pred_on_train"]) + "]" + "\n"
            except:
                pass
        if model_predictions["Frequentist_inference"]:
            summary_text = summary_text + "Average error on tmu: "
        #if FREQUENTISTS_RESULTS:
        #    summary_text = summary_text + "Mean error on tmu: "+ str(summary_log["Frequentist mean error on tmu"]) + "\n"
        summary_text = summary_text + "Train time per epoch: " + str(round(self.training_time,1)) + "s" + "\n"
        if model_predictions["Model_evaluation"] or model_predictions["Bayesian_inference"]:
            try:
                summary_text = summary_text + "Pred time per point: " + str(round(self.predictions["Model_evaluation"][timestamp]["Prediction_time"],1)) + "s"
            except:
                pass
        self.summary_text = summary_text

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
            output_predictions_json_file = utils.generate_dump_file_path(self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1],
                               "file path": output_predictions_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Predictions json file", output_predictions_json_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Predictions json file", output_predictions_json_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Predictions json file dump", output_predictions_json_file, "saved in", str(end-start), "s.", show=verbose)
    
    def save_predictions_h5(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions h5
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_h5_file = self.output_predictions_h5_file
            if not overwrite:
                utils.check_rename_file(output_predictions_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_h5_file = utils.generate_dump_file_path(self.output_predictions_h5_file, timestamp=timestamp)
        dictionary = dict(self.predictions)
        dd.io.save(output_predictions_h5_file, dictionary)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_h5_file)[-1],
                               "file path": output_predictions_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Predictions h5 file", output_predictions_h5_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Predictions h5 file", output_predictions_h5_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Predictions h5 file dump", output_predictions_h5_file, "saved in", str(end-start), "s.", show=verbose)

    def save_predictions(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions h5 and json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.save_predictions_json(timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_predictions_h5(timestamp=timestamp, overwrite=overwrite, verbose=verbose)

    def save_scalers(self, timestamp=None, overwrite=False, verbose=None):
        """ 
        Save scalers to pickle
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_scalers_pickle_file = self.output_scalers_pickle_file
            if not overwrite:
                utils.check_rename_file(output_scalers_pickle_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_scalers_pickle_file = utils.generate_dump_file_path(self.output_scalers_pickle_file, timestamp=timestamp)
        pickle_out = open(output_scalers_pickle_file, "wb")
        pickle.dump(self.scalerX, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.scalerY, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        end = timer()
        self.log[timestamp] = {"action": "saved scalers h5",
                               "file name": path.split(output_scalers_pickle_file)[-1],
                               "file path": output_scalers_pickle_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Scalers pickle file", output_scalers_pickle_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Scalers pickle file", output_scalers_pickle_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Scalers pickle file dump", output_scalers_pickle_file, "saved in", str(end-start), "s.", show=verbose)

    def save_model_graph_pdf(self, timestamp=None, overwrite=False, verbose=None):
        """ Save model graph to pdf
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_tf_model_graph_pdf_file = self.output_tf_model_graph_pdf_file
            if not overwrite:
                utils.check_rename_file(output_tf_model_graph_pdf_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_tf_model_graph_pdf_file = utils.generate_dump_file_path(self.output_tf_model_graph_pdf_file, timestamp=timestamp)
        png_file = path.splitext(output_tf_model_graph_pdf_file)[0]+".png"
        try:
            plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=png_file)
        except:
            print("Model not defined. No file is saved.")
            return
        utils.make_pdf_from_img(png_file)
        try:
            remove(png_file)
        except:
            try:
                time.sleep(1)
                remove(png_file)
            except:
                print("Cannot remove png file",png_file,".", show=verbose)
        end = timer()
        self.log[timestamp] = {"action": "saved model graph pdf",
                               "file name": path.split(output_tf_model_graph_pdf_file)[-1],
                               "file path": output_tf_model_graph_pdf_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print("Model graph pdf file", output_tf_model_graph_pdf_file, "updated in", str(end-start), "s.", show=verbose)
            else:
                print("Model graph pdf file", output_tf_model_graph_pdf_file, "saved in", str(end-start), "s.", show=verbose)
        elif overwrite == "dump":
            print("Model graph pdf file dump", output_tf_model_graph_pdf_file, "saved in", str(end-start), "s.", show=verbose)

    def save(self, timestamp=None, overwrite=False, verbose=None):
        """ Save all model information
        - data indices as hdf5 dataset
        - model in json format
        - model in h5 format (with weights)
        - model in onnx format
        - history, including summary log as json
        - scalers to jlib file
        - model graph to pdf
        """
        verbose, _ = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.save_data_indices(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_model_json(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_model_h5(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_model_onnx(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_history_json(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_predictions(timestamp=timestamp,overwrite=overwrite, verbose=verbose)
        self.save_summary_json(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_scalers(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_model_graph_pdf(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_log(timestamp=timestamp,overwrite=overwrite, verbose=verbose)

    def show_figures(self,fig_list,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        fig_list = np.array(fig_list).flatten().tolist()
        for fig in fig_list:
            try:
                from os import startfile
                startfile(r"%s"%fig)
                print("File", fig, "opened.", show=verbose)
            except:
                print("File", fig,"not found.", show=verbose)

    def mean_error(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        ME_model = K.mean(y_true-y_pred)
        return K.abs(ME_model)

    def mean_percentage_error(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MPE_model = K.mean((y_true-y_pred)/(K.sign(y_true)*K.clip(K.abs(y_true),
                                                                  K.epsilon(),
                                                                  None)))
        return 100. * K.abs(MPE_model)

    def R2_metric(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MSE_model =  K.sum(K.square( y_true-y_pred )) 
        MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return (1 - MSE_model/(MSE_baseline + K.epsilon()))

    def save_script(self, 
                    model_predict_kwargs = {},
                    verbose=True):
        """
        Saves the file :attr:`DnnLik.script_file <DNNLikelihood.DnnLik.script_file>`. 

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Creates file**

            - :attr:`DnnLik.script_file <DNNLikelihood.DnnLik.script_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if model_predict_kwargs == {}:
            model_predict_kwargs = {"batch_size": self.batch_size, "steps": None, "x_boundaries": False, "y_boundaries": False, "save_log": False, "verbose": False}
        with open(self.script_file, "w") as out_file:
            out_file.write("import DNNLikelihood\n"+"\n" +
                           "dnnlik = DNNLikelihood.DnnLik(name=None,\n" +
                           "\tinput_file="+r"'" + r"%s" % ((self.output_summary_json_file).replace(sep, '/'))+"', \n" +
                           "verbose = "+str(self.verbose)+")"+"\n"+"\n" +
                           "name = dnnlik.name\n" +
                           "def logpdf(x_pars,*args,**kwargs):\n" +
                           "\treturn dnnlik.model_predict(x_pars,*args,**kwargs)\n" +
                           "logpdf_args = None\n" +
                           "logpdf_kwargs = %s\n" % str(model_predict_kwargs) +
                           "pars_pos_poi = dnnlik.pars_pos_poi\n" +
                           "pars_pos_nuis = dnnlik.pars_pos_nuis\n" +
                           "pars_central = dnnlik.pars_central\n" +
                           "pars_init_vec = lik.logpdf_profiled_max['X']\n" +
                           "pars_labels = dnnlik.pars_labels\n" +
                           "pars_bounds = dnnlik.pars_bounds\n" +
                           "ndims = dnnlik.ndims\n" +
                           "output_folder = dnnlik.output_folder"
                           )
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.script_file)[-1],
                               "file path": self.script_file}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("File", self.script_file, "correctly generated.", show=verbose)

