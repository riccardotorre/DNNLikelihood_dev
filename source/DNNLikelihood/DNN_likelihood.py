__all__ = ["DNN_likelihood"]

import builtins
import codecs
import json
import matplotlib
import multiprocessing
import pickle
import re
import time
from datetime import datetime
from decimal import Decimal
from os import path, sep, stat, remove, startfile
from timeit import default_timer as timer

import h5py
import keras2onnx
import matplotlib.pyplot as plt
import numpy as np
import onnx
import tensorflow as tf
from scipy import optimize
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, losses, metrics, optimizers
from tensorflow.keras.layers import (AlphaDropout, BatchNormalization, Dense,
                                     Dropout, InputLayer)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

try:
    from livelossplot import PlotLossesTensorFlowKeras as PlotLossesKeras
except:
    print("No module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.")

import seaborn as sns
sns.set()
kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

from . import inference, utils
from .corner import corner, get_1d_hist, extend_corner_range
from .data import Data
from .show_prints import print
from .resources import Resources

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")


class DNN_likelihood(Resources): #show_prints.Verbosity inherited from resources.Resources
    def __init__(self,
                 name=None,
                 data=None,
                 input_data_file=None,
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
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #### Set input files
        self.input_summary_json_file = input_summary_json_file
        self.input_data_file = input_data_file
        self.__check_define_input_files()  
        ############ Check wheather to create a new DNNLik object from inputs or from files
        if self.input_files_base_name is None:
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
            self.__check_define_ensemble_name_folder(verbose=verbose_sub)
            ### Set model hyperparameters parameters
            self.__set_model_hyperparameters()
        else:
            ############ Initialize input parameters from file
            #### Load summary_log dictionary
            print("When providing DNNLik input folder all arguments but data, load_on_RAM and dtype are ignored and the object is constructed from saved data",show=verbose)
            self.__load_summary_json_and_log(verbose=verbose_sub)
            self.data = None
            #### Set main inputs and DataSample
            self.load_on_RAM = load_on_RAM
            if dtype is not None:
                self.dtype = dtype
            if seed is not None:
                self.seed = seed
            ### Set name, folders and files names
            if output_folder is not None:
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
        if self.input_files_base_name is not None:
            self.__load_model(verbose=verbose_sub)
            self.__load_history(verbose=verbose_sub)
            self.__load_scalers(verbose=verbose_sub)
            self.__load_data_indices(verbose=verbose_sub)
            self.__load_predictions(verbose=verbose_sub)
        else:
            self.epochs_available = 0
            self.idx_train, self.idx_val, self.idx_test = [np.array([], dtype="int"),np.array([], dtype="int"),np.array([], dtype="int")]
            self.scalerX, self.scalerY = [None,None]
            self.model = None
            self.history = {}
            self.predictions = {}
            self.figures_list = []
        self.X_train, self.Y_train, self.W_train = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype),np.array([], dtype=self.dtype)]
        self.X_val, self.Y_val = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype)]
        self.X_test, self.Y_test = [np.array([[]], dtype=self.dtype),np.array([], dtype=self.dtype)]
        ### Save object
        if self.input_files_base_name is None:
            self.save_summary_json(overwrite=False, verbose=verbose_sub)
            self.save_log(overwrite=False, verbose=verbose_sub)
        else:
            self.save_summary_json(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    def __set_resources(self,verbose=None):
        """
        Bla bla bla
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

    def __check_define_input_files(self):
        """
        Sets the attributes corresponding to input files
        :attr:`Likelihood.input_json_file <DNNLikelihood.Likelihood.input_json_file>`,
        :attr:`Likelihood.input_log_file <DNNLikelihood.Likelihood.input_log_file>`, and
        :attr:`Likelihood.input_pickle_file <DNNLikelihood.Likelihood.input_pickle_file>`
        depending on the value of the 
        :attr:`Likelihood.input_file <DNNLikelihood.Likelihood.input_file>` attribute.
        """
        if self.input_summary_json_file is None:
            self.input_log_file = self.input_summary_json_file
            self.input_files_base_name = self.input_summary_json_file
            self.input_history_json_file = self.input_files_base_name
            self.input_predictions_json_file = self.input_files_base_name
            self.input_idx_h5_file = self.input_files_base_name
            self.input_tf_model_h5_file = self.input_files_base_name
            self.input_scalers_pickle_file = self.input_files_base_name
        else:
            self.input_files_base_name = path.abspath(path.splitext(self.input_summary_json_file)[0].replace("_summary",""))
            self.input_log_file = self.input_files_base_name+".log"
            self.input_history_json_file = self.input_files_base_name+"_history.json"
            self.input_predictions_json_file = self.input_files_base_name+"_predictions.json"
            self.input_idx_h5_file = self.input_files_base_name+"_idx.h5"
            self.input_tf_model_h5_file = self.input_files_base_name+"_model.h5"
            self.input_scalers_pickle_file = self.input_files_base_name+"_scalerX.pickle"
        if self.input_data_file is not None:
            self.input_data_file = path.abspath(path.splitext(self.input_data_file)[0])

    def __check_define_output_files(self):
        """
        Sets attributes for output files
        """
        if self.input_files_base_name is None:
            if self.output_folder is None:
                self.output_folder = ""
            self.output_folder = path.abspath(self.output_folder)
        self.output_figures_folder = path.join(self.output_folder,"figures")
        self.output_files_base_name = path.join(self.output_folder, self.name)
        self.output_log_file = self.output_files_base_name+".log"
        self.output_summary_json_file = self.output_files_base_name+"_summary.json"
        self.output_history_json_file = self.output_files_base_name+"_history.json"
        self.output_predictions_json_file = self.output_files_base_name+"_predictions.json"
        self.output_idx_h5_file = self.output_files_base_name+"_idx.h5"
        self.output_tf_model_json_file = self.output_files_base_name+"_model.json"
        self.output_tf_model_h5_file = self.output_files_base_name+"_model.h5"
        self.output_tf_model_onnx_file = self.output_files_base_name+"_model.onnx"
        self.output_scalers_pickle_file = self.output_files_base_name+"_scalerX.pickle"
        self.output_model_graph_pdf_file = self.output_files_base_name+"_model_graph.pdf"
        self.output_figures_base_file = self.output_files_base_name+"_figure"
        utils.check_create_folder(self.output_folder)
        utils.check_create_folder(self.output_figures_folder)
        self.output_checkpoints_folder = None
        self.output_checkpoints_files = None
        self.output_figure_plot_losses_keras_file = None
        self.output_tensorboard_log_dir = None

    def __check_define_name(self):
        """
        Bla bla
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.name = "model_"+timestamp+"_DNNlikelihood"

    def __check_npoints(self):
        available_points_tot = self.data.npoints
        available_points_train = (1-self.data.test_fraction)*available_points_tot
        available_points_test = self.data.test_fraction*available_points_tot
        if self.npoints_train + self.npoints_val > available_points_train:
            raise Exception("npoints_train+npoints_val larger than the available number of points in data.\
                Please reduce npoints_train+npoints_val or change test_fraction in the Data object.")
        if self.npoints_test > available_points_test:
            raise Exception("npoints_test larger than the available number of points in data.\
                Please reduce npoints_test or change test_fraction in the Data object.")

    def __check_define_model_data_inputs(self):
        try:
            self.__model_data_inputs["npoints"]
        except:
            raise Exception("model_data_inputs dictionary should contain at least a key 'npoints'.")
        if self.__model_data_inputs["npoints"][1] <= 1:
            self.__model_data_inputs["npoints"][1] = round(self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][1])
        if self.__model_data_inputs["npoints"][2] <= 1:
            self.__model_data_inputs["npoints"][2] = round(self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][2])
        try:
            self.__model_data_inputs["scaleX"]
        except:
            self.__model_data_inputs["scaleX"] = False
        try:
            self.__model_data_inputs["scaleY"]
        except:
            self.__model_data_inputs["scaleY"] = False
        try:
            self.__model_data_inputs["weighted"]
        except:
            self.__model_data_inputs["weighted"] = False

    def __check_define_model_define_inputs(self):
        try:
            self.__model_define_inputs["hidden_layers"]
        except:
            raise Exception("model_define_inputs dictionary should contain at least a key 'hidden_layers'.")
        try:
            self.__model_define_inputs["act_func_out_layer"]
        except:
            self.__model_define_inputs["act_func_out_layer"] = "linear"
        try:
            self.__model_define_inputs["dropout_rate"]
        except:
            self.__model_define_inputs["dropout_rate"] = 0
        try:
            self.__model_define_inputs["batch_norm"]
        except:
            self.__model_define_inputs["batch_norm"] = False

    def __check_define_model_compile_inputs(self,verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        if self.__model_compile_inputs is None:
            self.__model_compile_inputs = {}
        try:
            self.__model_compile_inputs["loss"]
        except:
            self.__model_compile_inputs["loss"] = "mse"
            print("No loss has been specified. Using 'mse'.")
        try:
            self.__model_compile_inputs["metrics"]
        except:
            self.__model_compile_inputs["metrics"] = ["mse","mae","mape","msle"]

    def __check_define_model_train_inputs(self):
        try:
            self.__model_train_inputs["epochs"]
        except:
            raise Exception("model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")
        try:
            self.__model_train_inputs["batch_size"]
        except:
            raise Exception("model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")

    def __check_define_ensemble_name_folder(self,verbose=None):
        """
        Set ensemble attributes if an ensemble name is passed.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.ensemble_name is None:
            self.ensemble_folder = None
            self.standalone = True
            print("This is a 'standalone' DNNLikelihood and does not belong to a DNNLikelihood_ensemble. The attributes 'ensemble_name' and 'ensemble_folder' are therefore been set to None.",show=verbose)
        else:
            self.enseble_folder = path.abspath(path.join(self.output_folder,".."))
            self.standalone = False

    def __set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __set_dtype(self):
        if self.dtype is None:
            self.dtype = "float64"
        K.set_floatx(self.dtype)

    def __set_data(self,verbose=None):
        _, verbose_sub = self.set_verbosity(verbose)
        if self.data is None and self.input_data_file is None:
            raise Exception(
                "At least one of the arguments 'data' and 'input_data_file' should be specified.\nPlease specify one and retry.")
        elif self.data is not None and self.input_data_file is None:
            self.input_data_file = self.data.input_file
        else:
            if self.data is not None:
                print("Both the arguments 'data' and 'input_data_file' have been specified. 'data' will be ignored and the Data object will be set from 'input_data_file'.")
            self.data = Data(name=None,
                             data_X=None,
                             data_Y=None,
                             dtype=self.dtype,
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
        self.__check_npoints(verbose=verbose_sub)
        self.__set_pars_info()

    def __check_npoints(self,verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        self.npoints_available = self.data.npoints
        self.npoints_train_val_available = int((1-self.data.test_fraction)*self.npoints_available)
        self.npoints_test_available = int(self.data.test_fraction*self.npoints_available)
        required_points_train_val = self.npoints_train+self.npoints_val
        required_points_test = self.npoints_test
        if required_points_train_val > self.npoints_train_val_available:
            print("Requiring more training points than available in data. Either reduce npoints_train+npoints_val or change test_fraction in Data (and call Data._Data__define_test_fraction()).",show=verbose)
        if required_points_test > self.npoints_test_available:
            print("Requiring more test points than available in data. Either reduce npoints_test or change test_fraction in Data (and call Data._Data__define_test_fraction()).",show=verbose)

    def __set_pars_info(self):
        self.pars_pos_poi = self.data.pars_pos_poi
        self.pars_pos_nuis = self.data.pars_pos_nuis
        self.pars_labels = self.data.pars_labels
        self.generic_pars_labels = self.data.generic_pars_labels
        self.pars_bounds = self.data.pars_bounds

    def __set_model_hyperparameters(self):
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
        _, verbose_sub = self.set_verbosity(verbose)
        self.__set_optimizer(verbose=verbose_sub)  # this defines the string optimizer_string and object optimizer
        self.__set_loss(verbose=verbose_sub)  # this defines the string loss_string and the object loss
        self.__set_metrics(verbose=verbose_sub)  # this defines the lists metrics_string and metrics
        self.__set_callbacks(verbose=verbose_sub)  # this defines the lists callbacks_strings and callbacks

    def __load_summary_json_and_log(self,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        with open(self.input_summary_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded summary json",
                               "files names": [path.split(self.input_summary_json_file)[-1],
                                               path.split(self.input_log_file)[-1]],
                               "files paths": [self.input_summary_json_file,
                                               self.input_log_file]}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood summary json and log files loaded in", str(end-start), ".", show=verbose)

    def __load_history(self,verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            with open(self.input_history_json_file) as json_file:
                self.history = json.load(json_file)
        except:
            self.history = {}
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded history json",
                               "file name": path.split(self.input_history_json_file)[-1],
                               "file path": self.input_history_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood history json file loaded in", str(end-start), ".", show=verbose)

    def __load_model(self,verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            self.model = load_model(self.input_tf_model_h5_file, custom_objects={"mean_error": self.mean_error,
                                                                                 "mean_percentage_error": self.mean_percentage_error,
                                                                                 "R2_metric": self.R2_metric, 
                                                                                 "Rt_metric": self.Rt_metric})
        except:
            self.model = None
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded tf model h5",
                               "file name": path.split(self.input_tf_model_h5_file)[-1],
                               "file path": self.input_tf_model_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood tf model h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_scalers(self,verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            pickle_in = open(self.input_scalers_pickle_file, "rb")
            self.scalerX = pickle.load(pickle_in)
            self.scalerY = pickle.load(pickle_in)
            pickle_in.close()
        except:
            self.scalerX = None
            self.scalerY = None
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded scalers h5",
                               "file name": path.split(self.input_scalers_pickle_file)[-1],
                               "file path": self.input_scalers_pickle_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood scalers h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_data_indices(self,verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        h5_in = h5py.File(self.input_idx_h5_file, "r")
        data = h5_in.require_group("idx")
        self.idx_train = data["idx_train"][:]
        self.idx_val = data["idx_val"][:]
        self.idx_test = data["idx_test"][:]
        self.data.data_dictionary["idx_train"] = self.idx_train
        #self.data.data_dictionary["X_train"] = self.data.data_X[self.idx_train].astype(self.dtype)
        #self.data.data_dictionary["Y_train"] = self.data.data_Y[self.idx_train].astype(self.dtype)
        self.data.data_dictionary["idx_val"] = self.idx_val
        #self.data.data_dictionary["X_val"] = self.data.data_X[self.idx_val].astype(self.dtype)
        #self.data.data_dictionary["Y_val"] = self.data.data_Y[self.idx_val].astype(self.dtype)
        self.data.data_dictionary["idx_test"] = self.idx_test
        #self.data.data_dictionary["X_test"] = self.data.data_X[self.idx_test].astype(self.dtype)
        #self.data.data_dictionary["Y_test"] = self.data.data_Y[self.idx_test].astype(self.dtype)
        h5_in.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded data indices h5",
                               "file name": path.split(self.input_idx_h5_file)[-1],
                               "file path": self.input_idx_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood data indices h5 file loaded in", str(end-start), ".", show=verbose)

    def __load_predictions(self,verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            with open(self.input_predictions_json_file) as json_file: 
                self.predictions = json.load(json_file)
        except:
            self.predictions = {}
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded predictions json",
                               "file name": path.split(self.input_predictions_json_file)[-1],
                               "file path": self.input_predictions_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print("DNN_likelihood predictions json file loaded in",str(end-start), ".", show=verbose)

    def __set_optimizer(self,verbose=None):
        """
        Set Keras Model optimizer. Uses parameters from the dictionary "self.__model_optimizer_inputs".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if type(self.__model_optimizer_inputs) is str:
            self.optimizer_string = self.__model_optimizer_inputs
            self.optimizer = self.optimizer_string
        if type(self.__model_optimizer_inputs) is dict:
            name = self.__model_optimizer_inputs["name"]
            string = name+"("
            for key, value in utils.dic_minus_keys(self.__model_optimizer_inputs,["name"]).items():
                if type(value) is str:
                    value = "'"+value+"'"
                string = string+str(key)+"="+str(value)+", "
            optimizer_string = str("optimizers."+string+")").replace(", )", ")")
            self.optimizer_string = optimizer_string
            self.optimizer = eval(optimizer_string)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "optimizer set",
                               "optimizer": self.optimizer_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__
        print("Optimizer set to:", self.optimizer_string, show=verbose)

    def __set_loss(self, verbose=None):
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loss set",
                               "loss": self.loss_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_metrics(self, verbose=None):
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "metrics set",
                               "metrics": self.metrics_string}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_callbacks(self, verbose=None):
        """
        Set Keras Model callbacks. Uses parameters from the dictionary "self.__model_callbacks_inputs".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        callbacks_strings = []
        callbacks_string = [cb for cb in self.__model_callbacks_inputs if type(cb) is str]
        callbacks_dict = [cb for cb in self.__model_callbacks_inputs if type(cb) is dict]
        print("Setting callbacks")
        for cb in callbacks_string:
            if cb == "PlotLossesKeras":
                self.output_figure_plot_losses_keras_file = self.output_files_base_name+"_figure_plot_losses_keras.pdf"
                string = "PlotLossesKeras(fig_path='" + self.output_figure_plot_losses_keras_file+"')"
            elif cb == "ModelCheckpoint":
                self.output_checkpoints_folder = path.join(self.output_folder, "checkpoints")
                self.output_checkpoints_files = path.join(self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                utils.check_create_folder(self.output_checkpoints_folder)
                string = "callbacks.ModelCheckpoint(filepath='" + self.output_checkpoints_files+"')"
            elif cb == "TensorBoard":
                self.output_tensorboard_log_dir = path.join(self.output_folder, "logs")
                utils.check_create_folder(self.output_tensorboard_log_dir)
                utils.check_create_folder(path.join(self.output_folder, "logs/fit"))
                string = "callbacks.TensorBoard(log_dir='" + self.output_tensorboard_log_dir+"')"
            else:
                string = "callbacks."+cb+"()"
            callbacks_strings.append(string)
            print("\tAdded callback:", string, show=verbose)
        for cb in callbacks_dict:
            name = cb["name"]
            if name == "PlotLossesKeras":
                self.output_figure_plot_losses_keras_file = self.output_files_base_name+"_figure_plot_losses_keras.pdf"
                string = "fig_path = '"+self.output_figure_plot_losses_keras_file + "', "
                name = "callbacks."+name
            elif name == "ModelCheckpoint":
                self.output_checkpoints_folder = path.join(self.output_folder, "checkpoints")
                self.output_checkpoints_files = path.join(self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                utils.check_create_folder(self.output_checkpoints_folder)
                string = "filepath = '"+self.output_checkpoints_files+"', "
                name = "callbacks."+name
            elif name == "TensorBoard":
                self.output_tensorboard_log_dir = path.join(self.output_folder, "logs")
                utils.check_create_folder(self.output_tensorboard_log_dir)
                utils.check_create_folder(path.join(self.output_folder, "logs/fit"))
                string = "log_dir = '"+self.output_tensorboard_log_dir+"', "
                name = "callbacks."+name
            else:
                string = ""
                name = "callbacks."+name
            for key, value in utils.dic_minus_keys(cb,["name"]).items():
                if key == "monitor" and type(value) is str:
                    if "val_" in value:
                        value = value.split("val_")[1]
                    if value == "loss":
                        value = "val_loss"
                    else:
                        value = "val_" + utils.metric_name_unabbreviate(value)
                if type(value) is str:
                    value = "'"+value+"'"
                if not "path" in key:
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "callbacks set",
                               "callbacks": self.callbacks_strings}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of __init__

    def __set_epochs_to_run(self, verbose=None):
        """
        Private method that returns the number of steps to run computed as the difference between the value of
        :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` and the number of steps available in 
        :attr:`Sampler.backend <DNNLikelihood.Sampler.backend>`.
        If this difference is negative, a warning message asking to increase the value of 
        :attr:`Sampler.nsteps_required <DNNLikelihood.Sampler.nsteps_required>` is printed.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, _ = self.set_verbosity(verbose)
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
                or :attr:`Sampler.generic_pars_labels <DNNLikelihood.Sampler.generic_pars_labels>`, respectively,
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

    def compute_sample_weights(self, bins=100, power=1, verbose=None):
        _, verbose_sub = self.set_verbosity(verbose)
        self.W_train = self.data.compute_sample_weights(self.Y_train, bins=bins, power=power,verbose=verbose_sub).astype(self.dtype)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "computed sample weights"}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def define_scalers(self, verbose=None):
        _, verbose_sub = self.set_verbosity(verbose)
        self.scalerX, self.scalerY = self.data.define_scalers(self.X_train, self.Y_train, self.scalerX_bool, self.scalerY_bool, verbose=verbose_sub)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "defined scalers",
                               "scaler X": self.scalerX_bool,
                               "scaler Y": self.scalerY_bool}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved by generate_train_data

    def generate_train_data(self, verbose=None):
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
        # Define scalers
        self.define_scalers(verbose=verbose_sub)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "generated train data",
                               "data": ["idx_train", "X_train", "Y_train", "idx_val", "X_val", "Y_val"],
                               "npoints train": self.npoints_train,
                               "npoints val": self.npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def generate_test_data(self, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Generate data
        self.data.generate_test_data(self.npoints_test, verbose=verbose)
        self.idx_test = self.data.data_dictionary["idx_test"][:self.npoints_train]
        self.X_test = self.data.data_dictionary["X_test"][:self.npoints_test].astype(self.dtype)
        self.Y_test = self.data.data_dictionary["Y_test"][:self.npoints_test].astype(self.dtype)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "generated test data",
                               "data": ["idx_test", "X_test", "Y_test"],
                               "npoints test": self.npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_define(self,verbose=None):
        """
        Define the Keras Model "self.model" and the attributes "self.model_params", "self.model_trainable_params", and
        "self.model_non_trainable_params". Uses parameters from the dictionary "self.__model_define_inputs".
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x.replace("\"","'")))
        self.log[timestamp] = {"action": "defined tf model",
                               "model summary": summary_list}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for DNNLikelihood",self.name,"defined in", str(end-start), "s.",show=verbose)
        print(self.model.summary(), show=verbose)

    def model_compile(self,verbose=None):
        """
        Compile the Keras Model "self.model". Uses parameters from the dictionary "self.__model_compile_inputs".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Compile model
        start = timer()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "compiled tf model"}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for DNNLikelihood",self.name,"compiled in",str(end-start),"s.",show=verbose)

    def model_build(self, gpu="auto", verbose=None):
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
        if not create and not compile:
            print("Model already built.", show=verbose)
            return
        if self.gpu_mode:
            if gpu is "auto":
                gpu = 0
            elif gpu > len(self.available_gpus):
                print("gpu", gpu,
                      "does not exist. Continuing on first gpu.", show=verbose)
                gpu = 0
            self.training_device = self.available_gpus[gpu]
            device_id = self.training_device[0]
        else:
            if gpu is not "auto":
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "built tf model",
                               "gpu mode": self.gpu_mode,
                               "device id": device_id}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_train(self,verbose=None):
        """
        Train the Keras Model "self.model". Uses parameters from the dictionary "self.__model_train_inputs".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Scale data
        start = timer()
        epochs_to_run = self.__set_epochs_to_run(verbose=verbose_sub)
        #print("Checking data",show=verbose)
        if len(self.X_train) <= 1:
            print("Generating train data",show=verbose)
            self.generate_train_data()
        print("Scaling training data.", show=verbose)
        X_train = self.scalerX.transform(self.X_train)
        X_val = self.scalerX.transform(self.X_val)
        Y_train = self.scalerY.transform(self.Y_train.reshape(-1, 1)).reshape(len(self.Y_train))
        Y_val = self.scalerY.transform(self.Y_val.reshape(-1, 1)).reshape(len(self.Y_val))
        print([type(X_train),type(X_val),type(Y_train),type(Y_train)],show=verbose)
        # If PlotLossesKeras is in callbacks set plot style
        if "PlotLossesKeras" in str(self.callbacks_strings):
            plt.style.use(mplstyle_path)
        # Train model
        print("Start training of model for DNNLikelihood",self.name, ".",show=verbose)
        if self.weighted:
            # Compute weights
            if len(self.W_train) <= 1:
                print("In order to compute sample weights with the desired parameters please run the function\
                       self.compute_sample_weights(bins=100, power=1) before training.\n Proceding with sample weights\
                       computed with default parameters (bins=100 and power=1).", show=verbose)
                self.compute_sample_weights()
            # Train
            history = self.model.fit(X_train, Y_train, sample_weight=self.W_train, epochs=epochs_to_run, batch_size=self.batch_size, verbose=verbose_tf,
                    validation_data=(X_val, Y_val), callbacks=self.callbacks)
        else:
            history = self.model.fit(X_train, Y_train, epochs=epochs_to_run, batch_size=self.batch_size, verbose=verbose_sub,
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
        if "PlotLossesKeras" in str(self.callbacks_strings):
            plt.close()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "trained tf model",
                               "epochs run": epochs_to_run,
                               "epochs total": self.epochs_available,
                               "batch size": self.batch_size,
                               "training time": self.training_time}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for DNNLikelihood", self.name, "successfully trained for",
              epochs_to_run, "epochs in", self.training_time, "s.", show=verbose)

    def model_predict(self, X, batch_size=None, steps=None, save_log=True, verbose=None):
        """
        Predict with the Keras Model "self.model".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Scale data
        if batch_size is None:
            batch_size = self.batch_size
        start = timer()
        print("Scaling data.", show=verbose)
        X = self.scalerX.transform(X)
        pred = self.scalerY.inverse_transform(self.model.predict(X, batch_size=batch_size, steps=steps, verbose=verbose_sub)).reshape(len(X))
        end = timer()
        prediction_time = (end - start)/len(X)
        if save_log:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.log[timestamp] = {"action": "predicted with tf model",
                                   "batch size": batch_size,
                                   "npoints": len(pred),
                                   "prediction time": prediction_time}
            self.save_log(overwrite=True, verbose=verbose_sub)
        return [pred, prediction_time]

    def model_predict_scalar(self, x, steps=None, verbose=None):
        """
        Predict with the Keras Model "self.model".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        pred = self.model_predict(x, batch_size=1, steps=None, save_log=False, verbose=False)[0][0]
        return pred

    def model_compute_max_logpdf(self,pars_init=None,pars_bounds=None,nsteps=10000,tolerance=0.0001,optimizer=None,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        optimizer_log = optimizer
        start = timer()
        ## Parameters initialization
        if pars_init is None:
            pars_init = np.zeros(self.ndims)
        else:
            pars_init = np.array(pars_init).flatten()
        if optimizer is "scipy":
            print("Optimizing with scipy.optimize.", show=verbose)
            def minus_loglik(x):
                return -self.model_predict_scalar(x)
            if pars_bounds is None:
                ml = optimize.minimize(minus_loglik, pars_init, method="Powell")
            else:
                pars_bounds = np.array(pars_bounds)
                bounds = optimize.Bounds(pars_bounds[:, 0], pars_bounds[:, 1])
                ml = optimize.minimize(minus_loglik, pars_init, bounds=bounds,method="SLSQP")
            x_final, y_final = [ml["x"], ml["fun"]]
            self.X_max_logpdf = x_final
            self.Y_max_logpdf = y_final
            end = timer()
            print("Optimized in", str(end-start), "s.", show=verbose)
        else:
            ## Set optimizer
            def __set_optimizer(optimizer):
                if optimizer is None:
                    lr = 0.1
                    opt = tf.keras.optimizers.SGD(lr)
                    return opt
                elif type(optimizer) is dict:
                    name = list(optimizer.keys())[0]
                    string = name+"("
                    for key, value in optimizer[name].items():
                        if type(value) is str:
                            value = "'"+value+"'"
                        string = string+str(key)+"="+str(value)+", "
                    opt_string = str("optimizers."+string+")").replace(", )", ")")
                    opt = eval(opt_string)
                    return opt
                elif type(optimizer) is str:
                    opt = eval(optimizer)
                    return opt
                else:
                    opt = optimizer
                    return opt
            optimizer = __set_optimizer(optimizer)
            print("Optimizing with tensorflow.", show=verbose)
            ## Scalers
            sX = self.scalerX
            sY = self.scalerY        
            x_var = tf.Variable(sX.transform(pars_init.reshape(1,-1)), dtype=tf.float32)
            f = lambda: tf.reshape(-1*(self.model(x_var)),[])
            ##### Should add the possibility to parse optimizer in different ways #####
            if optimizer is None:
                lr = 0.1
                optimizer = tf.keras.optimizers.SGD(lr)
            run_lenght = 500
            nruns = int(nsteps/run_lenght)
            last_run_length = nsteps-run_lenght*nruns
            if last_run_length != 0:
                nruns = nruns+1
            for i in range(nruns):
                step_before = i*run_lenght+1
                value_before = sY.inverse_transform([-f().numpy()])[0]
                if i+1<nruns:
                    for _ in range(1,run_lenght):
                        optimizer.minimize(f,var_list=[x_var])
                    step_after = (i+1)*run_lenght
                else:
                    for _ in range(1,last_run_length):
                        optimizer.minimize(f,var_list=[x_var])
                    step_after = i*run_lenght+last_run_length
                value_after = sY.inverse_transform([-f().numpy()])[0]
                variation = np.abs(value_before-value_after)/np.abs(value_before)
                if value_after<value_before:
                    lr = optimizer._hyper["learning_rate"]
                    optimizer = tf.keras.optimizers.SGD(lr/2)
                    print("Optimizer learning rate reduced.", show=verbose)
                print("Step:",step_before,"Value:",value_before,"-- Step:",step_after,"Value:",value_after,r"-- % Variation",variation, show=verbose)
                if variation < tolerance:
                    end = timer()
                    print("Converged to tolerance",tolerance,"in",str(end-start),"s.", show=verbose)
                    x_final = sX.inverse_transform(x_var.numpy())[0]
                    y_final = sY.inverse_transform([-f().numpy()])[0]
                    self.X_max_logpdf = x_final
                    self.Y_max_logpdf = y_final
                    break
            end = timer()
            print("Did not converge to tolerance",tolerance,"using",nsteps,"steps.", show=verbose)
            print("Best tolerance",variation,"reached in",str(end-start),"s.", show=verbose)
            self.X_max_logpdf = x_final
            self.Y_max_logpdf = y_final
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "computed maximum logpdf",
                               "optimizer": optimizer_log,
                               "optimization time": end-start}
        self.save_log(overwrite=True, verbose=verbose_sub)

    #def maximum_loglik(self, loglik, npars=None, pars_init=None, pars_bounds=None,verbose=None):
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    def minus_loglik(x): return -loglik(x)
    #    if pars_bounds is None:
    #        print("Optimizing", show=verbose)
    #        ml = optimize.minimize(minus_loglik, pars_init, method="Powell")
    #    else:
    #        pars_bounds = np.array(pars_bounds)
    #        bounds = optimize.Bounds(pars_bounds[:, 0], pars_bounds[:, 1])
    #        ml = optimize.minimize(minus_loglik, pars_init, bounds=bounds)
    #    return [ml["x"], ml["fun"]]

    def model_evaluate(self, X, Y, batch_size=1, steps=None, verbose=None):
        """
        Predict with the Keras Model "self.model".
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        # Scale data
        start = timer()
        print("Scaling data.", show=verbose)
        X = self.scalerX.transform(X)
        Y = self.scalerY.transform(Y.reshape(-1, 1)).reshape(len(Y))
        pred = self.model.evaluate(X, Y, batch_size=batch_size, verbose=verbose_sub)
        end = timer()
        prediction_time = (end - start)/len(X)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "evaluated tf model",
                               "npoints": len(Y),
                               "evaluation time": prediction_time}
        self.save_log(overwrite=True, verbose=verbose_sub)
        return [pred, prediction_time]

    def generate_fig_base_title(self):
        title = "Ndim: " + str(self.ndims) + " - "
        title = title + "Nevt: " + "%.E" % Decimal(str(self.npoints_train)) + " - "
        title = title + "Layers: " + str(len(self.hidden_layers)) + " - "
        title = title + "Nodes: " + str(self.hidden_layers[0][0]) + " - "
        title = title.replace("+", "") + "Loss: " + str(self.loss_string)
        self.fig_base_title = title

    def plot_training_history(self, metrics=["loss"], yscale="log", show_plot=False, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        metrics = np.unique(metrics)
        for metric in metrics:
            start = timer()
            metric = utils.metric_name_unabbreviate(metric)
            val_metric = "val_"+ metric
            figure_filename = self.output_figures_base_file+"_training_history_" + metric+".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
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
            utils.append_without_duplicate(self.figures_list, figure_filename)
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.log[timestamp] = {"action": "saved figure", 
                                   "file name": path.split(figure_filename)[-1], 
                                   "file path": figure_filename}
            print(r"%s" % (figure_filename), "created and saved in", str(end-start), "s.", show=verbose)
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions
            
    def plot_pars_coverage(self, pars=None, loglik=True, show_plot=False, overwrite=False,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        if pars is None:
            pars = self.pars_pos_poi
        else:
            pars = pars
        for par in pars:
            start=timer()
            if loglik:
                figure_filename = self.output_figures_base_file+"_par_loglik_coverage_" + str(par) +".pdf"
            else:
                figure_filename = self.output_figures_base_file+"_par_lik_coverage_" + str(par) +".pdf"
            if not overwrite:
                utils.check_rename_file(figure_filename)
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
            plt.xlabel(r"%s"%(self.pars_labels[par]))
            if loglik:
                plt.ylabel(r"logprob ($\log\mathcal{L}+\log\mathcal{P}$)")
            else:
                plt.yscale("log")
                plt.ylabel(r"prob ($\mathcal{L}\cdot\mathcal{P}$)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(r"%s" %figure_filename)
            utils.append_without_duplicate(self.figures_list, figure_filename)
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.log[timestamp] = {"action": "saved figure",
                                   "file name": path.split(figure_filename)[-1],
                                   "file path": figure_filename}
            print(r"%s" %figure_filename,"created and saved in",str(end-start),"s.", show=verbose)
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions

    def plot_lik_distribution(self, loglik=True, show_plot=False, overwrite=False,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        start = timer()
        if loglik:
            figure_filename = self.output_figures_base_file+"_loglik_distribution.pdf"
        else:
            figure_filename = self.output_figures_base_file+"_lik_distribution.pdf"
        if not overwrite:
                utils.check_rename_file(figure_filename)
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
        utils.append_without_duplicate(self.figures_list, figure_filename)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_filename)[-1],
                               "file path": figure_filename}
        #self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions
        print(r"%s" %figure_filename,"created and saved in",str(end-start),"s.", show=verbose)

    def plot_corners_2samp(self, X1, X2, W1=None, W2=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                     HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                     ranges_extend=None, title1 = None, title2 = None,
                     color1="green", color2="red", 
                     plot_title="Params contours", legend_labels=None, 
                     figure_filename=None, show_plot=False, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        plt.style.use(mplstyle_path)
        start = timer()
        linewidth = 1.3
        intervals = inference.CI_from_sigma([1, 2, 3])
        if ranges_extend is None:
            ranges = extend_corner_range(X1, X2, pars, 0)
        else:
            ranges = extend_corner_range(X1, X2, pars, ranges_extend)
        pars_labels = self._DNN_likelihood__set_pars_labels(pars_labels)
        labels = np.array(pars_labels)[pars].tolist()
        if not overwrite:
            utils.check_rename_file(figure_filename)
        nndims = len(pars)
        if max_points is not None:
            if type(max_points) is list:
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
            HPDI1 = [[self.predictions['HPDI'][str(par)][HPDI1_dic["type"]][HPDI1_dic["sample"]][str(interval)]["Intervals"] for interval in intervals] for par in pars]
            HPDI2 = [[self.predictions['HPDI'][str(par)][HPDI2_dic["type"]][HPDI2_dic["sample"]][str(interval)]["Intervals"] for interval in intervals] for par in pars]
            #print(np.shape(HPDI1),np.shape(HPDI2))
        except:
            print("HPDI not present in predictions. Computing them.")
            HPDI1 = [inference.HPDI(samp1[:,i], intervals = intervals, weights=W1, nbins=nbins, print_hist=False, reduce_binning=True) for i in range(nndims)]
            HPDI2 = [inference.HPDI(samp2[:,i], intervals = intervals, weights=W2, nbins=nbins, print_hist=False, reduce_binning=True) for i in range(nndims)]
        levels1 = np.array([[np.sort(inference.HPD_quotas(samp1[:,[i,j]], nbins=nbins, intervals = inference.CI_from_sigma([1, 2, 3]), weights=W1)).tolist() for j in range(nndims)] for i in range(nndims)])
        levels2 = np.array([[np.sort(inference.HPD_quotas(samp2[:, [i, j]], nbins=nbins, intervals=inference.CI_from_sigma(
            [1, 2, 3]), weights=W2)).tolist() for j in range(nndims)] for i in range(nndims)])
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
        fig.suptitle(r"%s" % plot_title, fontsize=26)
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
        fig.legend(lines, legend_labels, fontsize=int(7+2*nndims), loc="best")#(1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        #plt.tight_layout()
        plt.savefig(figure_filename, dpi=50)  # ,dpi=200)
        utils.append_without_duplicate(self.figures_list, figure_filename)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": path.split(figure_filename)[-1],
                               "file path": figure_filename}
        print(r"%s" % figure_filename, "created and saved in", str(end-start), "s.", show=verbose)
        print("Plot done and saved in", end-start, "s.", show=verbose)

    def model_compute_predictions(self, 
                                  CI=inference.CI_from_sigma([inference.sigma_from_CI(0.5), 1, 2, 3]), 
                                  pars=None,
                                  batch_size=None,
                                  overwrite=False,
                                  verbose=None,
                                  HPDI_kwargs={}, # intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True
                                  plot_training_history_kwargs = {}, # metrics=["loss"], yscale="log", show_plot=False, overwrite=False, verbose=None
                                  plot_pars_coverage_kwargs = {}, # pars=None, loglik=True, show_plot=False, overwrite=False, verbose=None
                                  plot_lik_distribution_kwargs = {}, # loglik=True, show_plot=False, overwrite=False, verbose=None
                                  plot_corners_2samp_kwargs={}):  # W1=None, W2=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                                                                  # HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                                                                  # ranges_extend=None, title1 = None, title2 = None,
                                                                  # color1="green", color2="red", 
                                                                  # plot_title="Params contours", legend_labels=None, 
                                                                  # figure_filename=None, show_plot=False, overwrite=False, verbose=None
        verbose, verbose_sub = self.set_verbosity(verbose)
        start_global = timer()
        start = timer()
        if pars is None:
            pars = self.data.pars_pos_poi.tolist()
        else:
            pars = pars
        if batch_size is None:
            batch_size = self.batch_size
        print("Compute predictions", show=verbose)
        if len(self.X_train) <= 1:
            print("Generating train data", show=verbose)
            self.generate_train_data(verbose=verbose_sub)
        if len(self.X_test) <= 1:
            print("Generating test data", show=verbose)
            self.generate_test_data(verbose=verbose_sub)
            self.save_data_indices(overwrite=True,verbose=verbose_sub)
        print("Evaluate all metrics on (scaled) train/val/test using best models", show=verbose)
        metrics_names = self.model.metrics_names
        metrics_names_train = [i+"_best" for i in self.model.metrics_names]
        metrics_names_val = ["val_"+i+"_best" for i in self.model.metrics_names]
        metrics_names_test = ["test_"+i+"_best" for i in self.model.metrics_names]
        metrics_train = self.model_evaluate(self.X_train, self.Y_train, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
        metrics_val = self.model_evaluate(self.X_val, self.Y_val, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
        metrics_test = self.model_evaluate(self.X_test, self.Y_test, batch_size=self.batch_size,verbose=verbose_sub)[0][0:len(metrics_names)]
        metrics_true = {**dict(zip(metrics_names_train, metrics_train)), **dict(zip(metrics_names_val, metrics_val)), **dict(zip(metrics_names_test, metrics_test))}
        self.predictions = {**self.predictions, **{"Metrics on scaled data": metrics_true}}
        print("Predict Y for train/val/test samples", show=verbose)
        Y_pred_train, prediction_time1 = self.model_predict(self.X_train, batch_size=self.batch_size,verbose=verbose_sub)
        Y_pred_val, prediction_time2 = self.model_predict(self.X_val, batch_size=self.batch_size,verbose=verbose_sub)
        Y_pred_test, prediction_time3 = self.model_predict(self.X_test, batch_size=self.batch_size,verbose=verbose_sub)
        self.predictions = {**self.predictions, **{"Prediction time": (prediction_time1+prediction_time2+prediction_time3)/3}}
        print("Evaluate all metrics on (un-scaled) train/val/test using best models", show=verbose)
        metrics_names_train = [i+"_best_unscaled" for i in self.model.metrics_names]
        metrics_names_val = ["val_"+i+"_best_unscaled" for i in self.model.metrics_names]
        metrics_names_test = ["test_"+i +"_best_unscaled" for i in self.model.metrics_names]
        metrics_train_unscaled = [metric(self.Y_train,Y_pred_train).numpy() for metric in [self.loss]+self.metrics]
        metrics_val_unscaled = [metric(self.Y_val,Y_pred_val).numpy() for metric in [self.loss]+self.metrics]
        metrics_test_unscaled = [metric(self.Y_test,Y_pred_test).numpy() for metric in [self.loss]+self.metrics]
        metrics_unscaled = {**dict(zip(metrics_names_train, metrics_train_unscaled)), **dict(zip(metrics_names_val, metrics_val_unscaled)), **dict(zip(metrics_names_test, metrics_test_unscaled))}
        self.predictions = {**self.predictions, **{"Metrics on unscaled data": metrics_unscaled}}
        print("Compute exp(Y_true) and exp(Y_pred) for train/val/test samples", show=verbose)
        [Y_train_exp, Y_val_exp, Y_test_exp, Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [np.exp(Y) for Y in [self.Y_train, self.Y_val, self.Y_test, Y_pred_train, Y_pred_val, Y_pred_test]]
        end = timer()
        print("Prediction on ("+str(self.npoints_train)+","+str(self.npoints_val)+","+str(self.npoints_test)+")", "(train,val,test) points done in", str(end-start), "s.", show=verbose)
        print("Compute Bayesian inference benchmarks", show=verbose)
        start = timer()
        print("Computing weights (pred vs true) for reweighting of distributions", show=verbose)
        [W_train, W_val, W_test] = [utils.normalize_weights(W) for W in [Y_pred_train_exp/Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]]
        print("Computing HPDI (pred vs true) using reweighted distributions", show=verbose)
        #(data, intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True)
        HPDI_result = {}
        for par in pars:
            [HPDI_train, HPDI_val, HPDI_test] = [inference.HPDI(X, CI, **HPDI_kwargs) for X in [self.X_train[:, par], self.X_val[:, par], self.X_test[:, par]]]
            [HPDI_pred_train, HPDI_pred_val, HPDI_pred_test] = [inference.HPDI(self.X_train[:, par], CI, W_train, **HPDI_kwargs), inference.HPDI(self.X_val[:, par], CI, W_val, **HPDI_kwargs), inference.HPDI(self.X_test[:, par], CI, W_test, **HPDI_kwargs)]
            HPDI_result[str(par)] = {"true": {"train": HPDI_train, "val": HPDI_val, "test": HPDI_test}, "pred":{"train": HPDI_pred_train, "val": HPDI_pred_val, "test": HPDI_pred_test}}
        HDPI_error = inference.HPDI_error(HPDI_result)
        if "HPDI" not in self.predictions:
            self.predictions["HPDI"] = {}
        if "HPDI_error" not in self.predictions:
            self.predictions["HPDI_error"] = {}
        self.predictions["HPDI"] = {**self.predictions["HPDI"], **HPDI_result}
        self.predictions["HPDI_error"] = {**self.predictions["HPDI_error"], **HDPI_error}
        #self.predictions = {**self.predictions, **{"HPDI": HPDI_result},**{"HPDI_error": HDPI_error}}
        print("Computing KS test between one-dimensional distributions (pred vs true) using reweighted distributions", show=verbose)
        KS_test_pred_train = [inference.ks_w(self.X_test[:, q], self.X_train[:, q], np.ones(len(self.X_test)), W_train) for q in range(len(self.X_train[0]))]
        KS_test_pred_val = [inference.ks_w(self.X_test[:, q], self.X_val[:, q], np.ones(len(self.X_test)), W_val) for q in range(len(self.X_train[0]))]
        KS_val_pred_test = [inference.ks_w(self.X_val[:, q], self.X_test[:, q], np.ones(len(self.X_val)), W_test) for q in range(len(self.X_train[0]))]
        KS_train_pred_train = [inference.ks_w(self.X_train[:, q], self.X_train[:, q], np.ones(len(self.X_train)), W_train) for q in range(len(self.X_train[0]))]
        KS_test_pred_train_median = np.median(np.array(KS_test_pred_train)[:, 1]).tolist()
        KS_test_pred_val_median = np.median(np.array(KS_test_pred_val)[:, 1]).tolist()
        KS_val_pred_test_median = np.median(np.array(KS_val_pred_test)[:, 1]).tolist()
        KS_train_pred_train_median = np.median(np.array(KS_train_pred_train)[:, 1]).tolist()
        self.predictions = {**self.predictions, **{"KS": {"Test vs pred on train": KS_test_pred_train,
                                                          "Test vs pred on val": KS_test_pred_val,
                                                          "Val vs pred on test": KS_val_pred_test,
                                                          "Train vs pred on train": KS_train_pred_train}},
                                                **{"KS medians": {"Test vs pred on train": KS_test_pred_train_median,
                                                                  "Test vs pred on val": KS_test_pred_val_median,
                                                                  "Val vs pred on test": KS_val_pred_test_median,
                                                                  "Train vs pred on train": KS_train_pred_train_median}}}
        self.predictions = utils.convert_types_dict(self.predictions)
        # Sort nested dictionary by keys
        self.predictions = utils.sort_dict(self.predictions)
        end = timer()
        print("Bayesian inference benchmarks computed in", str(end-start), "s.", show=verbose)
        self.save_predictions_json(overwrite=overwrite,verbose=verbose_sub)
        self.generate_summary_text()
        self.generate_fig_base_title()
        print("Making plots.", show=verbose)
        start = timer()
        self.plot_training_history(overwrite=overwrite, verbose=verbose_sub, **plot_training_history_kwargs)
        self.plot_pars_coverage(pars=pars, overwrite=overwrite, verbose=verbose_sub, **plot_pars_coverage_kwargs)
        self.plot_lik_distribution(overwrite=overwrite,verbose=verbose_sub, **plot_lik_distribution_kwargs)
        #### TRAIN CORNER
        ## **corners_kwargs should include ranges_extend, max_points, nbins, show_plot, overwrite
        self.plot_corners_2samp(self.X_train, self.X_train, W1=None, W2=W_train,
                                HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "train", "type": "pred"},
                                pars = pars, pars_labels = "original",
                                title1 = "$68\%$ HPDI train", title2 = "$68\%$ HPDI DNN train",
                                color1 = "green", color2 = "red",
                                plot_title = "DNN reweighting train",
                                legend_labels = [r"Train set ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"DNN reweight train ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"$68.27\%$ HPDI", 
                                                 r"$95.45\%$ HPDI", 
                                                 r"$99.73\%$ HPDI"],
                                figure_filename=self.output_figures_base_file+"_corner_pars_train.pdf",
                                overwrite=overwrite, verbose=verbose_sub, **plot_corners_2samp_kwargs)
        #### TEST CORNER
        ## **corners_kwargs should include ranges_extend, max_points, nbins, show_plot, overwrite
        self.plot_corners_2samp(self.X_train, self.X_train, W1=None, W2=W_train,
                                HPDI1_dic={"sample": "test", "type": "true"}, HPDI2_dic={"sample": "test", "type": "pred"},
                                pars = pars, pars_labels = "original",
                                title1 = "$68\%$ HPDI test", title2 = "$68\%$ HPDI DNN test",
                                color1 = "green", color2 = "red",
                                plot_title = "DNN reweighting train",
                                legend_labels = [r"Test set ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"DNN reweight test ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"$68.27\%$ HPDI", 
                                                 r"$95.45\%$ HPDI", 
                                                 r"$99.73\%$ HPDI"],
                                figure_filename=self.output_figures_base_file+"_corner_pars_test.pdf",
                                overwrite=overwrite, verbose=verbose_sub, **plot_corners_2samp_kwargs)
        #### TRAIN vs TEST CORNER
        ## **corners_kwargs should include ranges_extend, max_points, nbins, show_plot, overwrite
        self.plot_corners_2samp(self.X_train, self.X_test, W1=None, W2=None,
                                HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                                pars = pars, pars_labels = "original",
                                title1 = "$68\%$ HPDI train", title2 = "$68\%$ HPDI test",
                                color1 = "green", color2 = "red",
                                plot_title = "Train vs test samples",
                                legend_labels = [r"Train set ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"Test set ($%s$ points)" % utils.latex_float(len(self.X_train)),
                                                 r"$68.27\%$ HPDI", 
                                                 r"$95.45\%$ HPDI", 
                                                 r"$99.73\%$ HPDI"],
                                figure_filename=self.output_figures_base_file+"_corner_pars_train_vs_test.pdf",
                                overwrite=overwrite, verbose=verbose_sub, **plot_corners_2samp_kwargs)
        end = timer()
        print("All plots done in", str(end-start), "s.", show=verbose)
        self.save_summary_json(overwrite=overwrite, verbose=verbose_sub)
        end_global = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "computed predictions",
                               "probability intervals": CI,
                               "pars": pars,
                               "batch size": batch_size}
        self.save_log(overwrite=overwrite, verbose=verbose_sub)
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

    def save_log(self, overwrite=False, verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_log_file, verbose=verbose_sub)
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if overwrite:
            print("DNN_likelihood log file", self.output_log_file, "updated in", str(end-start), "s.", show=verbose)
        else:
            print("DNN_likelihood log file", self.output_log_file, "saved in", str(end-start), "s.", show=verbose)

    def save_data_indices(self, overwrite=False, verbose=None):
        """ Save indices to member_n_idx.h5 as h5 file
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_idx_h5_file, verbose=verbose_sub)
        #self.close_opened_dataset(verbose=verbose_sub)
        utils.check_delete_file(self.output_idx_h5_file)
        h5_out = h5py.File(self.output_idx_h5_file)
        h5_out.require_group(self.name)
        data = h5_out.require_group("idx")
        data["idx_train"] = self.idx_train
        data["idx_val"] = self.idx_val
        data["idx_test"] = self.idx_test
        h5_out.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved indices",
                               "file name": path.split(self.output_idx_h5_file)[-1],
                               "file path": self.output_idx_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_idx_h5_file, "created and saved in", str(end-start), "s.", show=verbose)

    def save_model_json(self, overwrite=False, verbose=None):
        """ Save model to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_tf_model_json_file, verbose=verbose_sub)
        model_json = self.model.to_json()
        with open(self.output_tf_model_json_file, "w") as json_file:
            json_file.write(model_json)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved tf model json",
                               "file name": path.split(self.output_tf_model_json_file)[-1],
                               "file path": self.output_tf_model_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_tf_model_json_file, "created and saved.", str(end-start), "s.", show=verbose)

    def save_model_h5(self, overwrite=False, verbose=None):
        """ Save model to h5
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_tf_model_h5_file, verbose=verbose_sub)
        self.model.save(self.output_tf_model_h5_file)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved tf model h5",
                               "file name": path.split(self.output_tf_model_h5_file)[-1],
                               "file path": self.output_tf_model_h5_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_tf_model_h5_file, "created and saved.",str(end-start), "s.", show=verbose)

    def save_model_onnx(self, overwrite=False, verbose=None):
        """ Save model to onnx
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_tf_model_onnx_file, verbose=verbose_sub)
        onnx_model = keras2onnx.convert_keras(self.model, self.name)
        onnx.save_model(onnx_model, self.output_tf_model_onnx_file)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved tf model onnx",
                               "file name": path.split(self.output_tf_model_onnx_file)[-1],
                               "file path": self.output_tf_model_onnx_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_tf_model_onnx_file,"created and saved.", str(end-start), "s.", show=verbose)

    def save_history_json(self,overwrite=False,verbose=None):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_history_json_file, verbose=verbose_sub)
        history = self.history
        #for key in list(history.keys()):
        #    self.history[utils.metric_name_abbreviate(key)] = self.history.pop(key)
        new_hist = utils.convert_types_dict(history)
        with codecs.open(self.output_history_json_file, "w", encoding="utf-8") as f:
            json.dump(new_hist, f, separators=(",", ":"), sort_keys=True, indent=4)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved history json",
                               "file name": path.split(self.output_history_json_file)[-1],
                               "file path": self.output_history_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_history_json_file, "created and saved.", str(end-start), "s.", show=verbose)

    def save_summary_json(self, overwrite=False, verbose=None):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_summary_json_file, verbose=verbose_sub)
        dictionary = utils.dic_minus_keys(self.__dict__,["_DNN_likelihood__resources_inputs",
                                                         "callbacks","data","history",
                                                         "idx_test","idx_train","idx_val",
                                                         "input_files_base_name","input_history_json_file",
                                                         "input_idx_h5_file","input_log_file",
                                                         "input_predictions_json_file",
                                                         "input_scalers_pickle_file","input_summary_json_file",
                                                         "input_tf_model_h5_file","load_on_RAM",
                                                         "log","loss","metrics","model","optimizer",
                                                         "predictions", "scalerX","scalerY","verbose",
                                                         "X_test","X_train","X_val","Y_test","Y_train","Y_val","W_train"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.output_summary_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved summary json",
                               "file name": path.split(self.output_summary_json_file)[-1],
                               "file path": self.output_summary_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        if overwrite:
            print("DNN_likelihood json file", self.output_summary_json_file, "updated in", str(end-start), "s.", show=verbose)
        else:
            print("DNN_likelihood json file", self.output_summary_json_file, "saved in", str(end-start), "s.", show=verbose)

    def generate_summary_text(self):
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
        summary_text = summary_text + "Best losses: " + "[" + "{0:1.2e}".format(self.predictions["Metrics on scaled data"]["loss_best"]) + "," + \
                                                              "{0:1.2e}".format(self.predictions["Metrics on scaled data"]["val_loss_best"]) + "," + \
                                                              "{0:1.2e}".format(self.predictions["Metrics on scaled data"]["test_loss_best"]) + "]" + "\n"
        summary_text = summary_text + "Best losses scaled: " + "[" + "{0:1.2e}".format(self.predictions["Metrics on unscaled data"]["loss_best_unscaled"]) + "," + \
                                                                     "{0:1.2e}".format(self.predictions["Metrics on unscaled data"]["val_loss_best_unscaled"]) + "," + \
                                                                     "{0:1.2e}".format(self.predictions["Metrics on unscaled data"]["test_loss_best_unscaled"]) + "]" + "\n"
        summary_text = summary_text + "KS $p$-median: " + "[" + "{0:1.2e}".format(self.predictions["KS medians"]["Test vs pred on train"]) + "," + \
                                                                "{0:1.2e}".format(self.predictions["KS medians"]["Test vs pred on val"]) + "," + \
                                                                "{0:1.2e}".format(self.predictions["KS medians"]["Val vs pred on test"]) + "," + \
                                                                "{0:1.2e}".format(self.predictions["KS medians"]["Train vs pred on train"]) + "]" + "\n"
        #if FREQUENTISTS_RESULTS:
        #    summary_text = summary_text + "Mean error on tmu: "+ str(summary_log["Frequentist mean error on tmu"]) + "\n"
        summary_text = summary_text + "Train time per epoch: " + str(round(self.training_time,1)) + "s" + "\n"
        summary_text = summary_text + "Pred time per point: " + str(round(self.predictions["Prediction time"],1)) + "s"
        self.summary_text = summary_text

    def save_predictions_json(self, overwrite=False, verbose=None):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_predictions_json_file, verbose=verbose_sub)
        #history = self.predictions
        #new_hist = utils.convert_types_dict(history)
        with codecs.open(self.output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(self.predictions, f, separators=(
                ",", ":"), sort_keys=True, indent=4)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(self.output_predictions_json_file)[-1],
                               "file path": self.output_predictions_json_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_predictions_json_file, "created and saved.", str(end-start), "s.", show=verbose)

    def save_scalers(self, overwrite=False,verbose=None):
        """ 
        Save scalers to pickle
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_scalers_pickle_file, verbose=verbose_sub)
        pickle_out = open(self.output_scalers_pickle_file, "wb")
        pickle.dump(self.scalerX, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.scalerY, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved scalers h5",
                               "file name": path.split(self.output_scalers_pickle_file)[-1],
                               "file path": self.output_scalers_pickle_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print("DNN_likelihood scalers pickle file", self.output_scalers_pickle_file,"saved in", str(end-start), "s.",show=verbose)

    def save_model_graph_pdf(self, overwrite=False, verbose=None):
        """ Save model graph to pdf
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_model_graph_pdf_file, verbose=verbose_sub)
        png_file = path.splitext(self.output_model_graph_pdf_file)[0]+".png"
        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=png_file)
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
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved model graph pdf",
                               "file name": path.split(self.output_model_graph_pdf_file)[-1],
                               "file path": self.output_model_graph_pdf_file}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by model_store
        print(self.output_model_graph_pdf_file," created and saved in", str(end-start), "s.", show=verbose)

    def model_store(self, overwrite=False, verbose=None):
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
        self.save_data_indices(overwrite=overwrite,verbose=verbose)
        self.save_model_json(overwrite=overwrite,verbose=verbose)
        self.save_model_h5(overwrite=overwrite,verbose=verbose)
        self.save_model_onnx(overwrite=overwrite,verbose=verbose)
        self.save_history_json(overwrite=overwrite,verbose=verbose)
        self.save_summary_json(overwrite=overwrite,verbose=verbose)
        self.save_scalers(overwrite=overwrite,verbose=verbose)
        self.save_model_graph_pdf(overwrite=overwrite,verbose=verbose)
        self.save_log(overwrite=overwrite, verbose=verbose)

    def show_figures(self,fig_list,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        fig_list = np.array(fig_list).flatten().tolist()
        for fig in fig_list:
            try:
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

