__all__ = ["DNNLik"]

import json
#import ndjson as json
import codecs
import h5py
from timeit import default_timer as timer
import time
import multiprocessing
import builtins
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, metrics, optimizers, callbacks, losses
from tensorflow.keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout, InputLayer)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import keras2onnx
import onnx
import os
import pickle
try:
    from jupyterthemes import jtplot
except:
    print("No module named 'jupyterthemes'. Continuing without.\nIf you wish to customize jupyter notebooks please install 'jupyterthemes'.")
try:
    from livelossplot import PlotLossesTensorFlowKeras as PlotLossesKeras

except:
    print("No module named 'livelossplot'. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.")

from .data_sample import Data_sample
from . import utility
from . import set_resources
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

class DNNLik(object):
    """The DNNLik class."""
    def __init__(self,
                 DNNLik_input_folder=None,
                 ensemble_name=None,
                 member_number=0,
                 data_sample=None,
                 data_sample_input_filename=None,
                 ensemble_folder=None,
                 load_on_RAM=False,
                 seed=1,
                 dtype=None,
                 same_data=True,
                 model_data_member_kwargs=None,
                 model_define_member_kwargs=None,
                 model_optimizer_member_kwargs=None,
                 model_compile_member_kwargs=None,
                 model_callbacks_member_kwargs=None,
                 model_train_member_kwargs=None,
                 resources_member_kwargs=None,
                 verbose=True
                 ):
        #### Set global verbosity
        global ShowPrints
        self.member_verbose_mode = verbose
        ShowPrints = self.member_verbose_mode
        #### Set resources
        if resources_member_kwargs is None:
            self.get_available_gpus()
            self.get_available_cpu()
            self.set_gpus(gpus_list="all")
        else:
            self.available_gpus = resources_member_kwargs["available_gpus"]
            self.available_cpu = resources_member_kwargs["available_cpu"]
            self.active_gpus = resources_member_kwargs["active_gpus"]
            self.gpu_mode = resources_member_kwargs["gpu_mode"]
        ############ Check wheather to create a new DNNLik object from inputs or from files
        self.DNNLik_input_folder = DNNLik_input_folder
        if self.DNNLik_input_folder is None:
            ############ Initialize input parameters from arguments
            #### Set main inputs and DataSample
            self.ensemble_name = ensemble_name
            self.data_sample = data_sample
            self.data_sample_input_filename = data_sample_input_filename
            self.__model_data_member_kwargs = model_data_member_kwargs
            self.__model_define_member_kwargs = model_define_member_kwargs
            self.__model_optimizer_member_kwargs = model_optimizer_member_kwargs
            self.__model_compile_member_kwargs = model_compile_member_kwargs
            self.__model_callbacks_member_kwargs = model_callbacks_member_kwargs
            self.__model_train_member_kwargs = model_train_member_kwargs
            self.npoints_train, self.npoints_val, self.npoints_test = self.__model_data_member_kwargs["npoints"]
            self.load_on_RAM = load_on_RAM
            self.seed = seed
            self.dtype = dtype
            self.same_data = same_data

            ### Set name, folders and files names
            self.member_number = member_number
            self.member_name = self.ensemble_name +"_member_"+str(self.member_number)
            self.ensemble_folder = ensemble_folder
            self.ensemble_results_folder = ensemble_folder+"/ensemble"
            self.member_results_folder = self.ensemble_folder + "/member_"+str(self.member_number)
            self.__check_create_ensemble_folder()
            self.__check_create_member_results_folder()
            self.history_json_filename = self.member_results_folder+"/"+self.member_name+"_history.json"
            self.summary_log_json_filename = self.member_results_folder+"/"+self.member_name+"_summary_log.json"
            self.predictions_json_filename = self.member_results_folder+"/"+self.member_name+"_predictions.json"
            self.idx_filename = self.member_results_folder+"/"+self.member_name+"_idx.h5"
            self.model_json_filename = self.member_results_folder+"/"+self.member_name+"_model.json"
            self.model_h5_filename = self.member_results_folder+"/"+self.member_name+"_model.h5"
            self.model_onnx_filename = self.member_results_folder+"/"+self.member_name+"_model.onnx"
            self.scalerX_jlib_filename = self.member_results_folder+"/"+self.member_name+"_scalerX.jlib"
            self.scalerY_jlib_filename = self.member_results_folder+"/"+self.member_name+"_scalerY.jlib"
            self.model_graph_pdf_filename = self.member_results_folder+"/"+self.member_name+"_model_graph.pdf"

        else:
            ############ Initialize input parameters from file
            #### Load summary_log dictionary
            print("When providing DNNLik input folder all arguments but data_sample, load_on_RAM and dtype are ignored and the object is constructed from saved data")
            summary_log = self.__load_summary_log()
            
            #### Set main inputs and DataSample
            self.ensemble_name = summary_log['ensemble_name']
            self.data_sample = data_sample
            self.data_sample_input_filename = summary_log['data_sample_input_filename']
            self.__model_data_member_kwargs = summary_log['_DNNLik__model_data_member_kwargs']
            self.__model_define_member_kwargs = summary_log['_DNNLik__model_define_member_kwargs']
            self.__model_optimizer_member_kwargs = summary_log['_DNNLik__model_optimizer_member_kwargs']
            self.__model_compile_member_kwargs = summary_log['_DNNLik__model_compile_member_kwargs']
            self.__model_callbacks_member_kwargs = summary_log['_DNNLik__model_callbacks_member_kwargs']
            self.__model_train_member_kwargs = summary_log['_DNNLik__model_train_member_kwargs']
            self.npoints_train, self.npoints_val, self.npoints_test = self.__model_data_member_kwargs["npoints"]
            self.load_on_RAM = load_on_RAM
            self.seed = summary_log['seed']
            self.dtype = dtype
            if self.dtype is None:
                self.dtype = summary_log['dtype']
            self.same_data = summary_log['same_data']

            ### Set name, folders and files names
            self.member_number = summary_log['member_number']
            self.member_name = summary_log['member_name']
            self.ensemble_folder = summary_log['ensemble_folder']
            self.ensemble_results_folder = self.ensemble_folder+"/ensemble"
            self.member_results_folder = summary_log['member_results_folder']
            self.history_json_filename = summary_log['history_json_filename']
            self.summary_log_json_filename = summary_log['summary_log_json_filename']
            self.predictions_json_filename = summary_log['predictions_json_filename']
            self.idx_filename = summary_log['idx_filename']
            self.model_json_filename = summary_log['model_json_filename']
            self.model_h5_filename = summary_log['model_h5_filename']
            self.model_onnx_filename = summary_log['model_onnx_filename']
            self.scalerX_jlib_filename = summary_log['scalerX_jlib_filename']
            self.scalerY_jlib_filename = summary_log['scalerY_jlib_filename']
            self.model_graph_pdf_filename = summary_log['model_graph_pdf_filename']

        #### Set additional inputs
        self.__set_seed()
        self.__set_dtype()
        self.__set_data_sample()
        self.ndim = self.data_sample.data_X.shape[1]

        ### Set parameters
        self.scalerX_bool = self.__model_data_member_kwargs["scalerX"]
        self.scalerY_bool = self.__model_data_member_kwargs["scalerY"]
        self.weighted = self.__model_data_member_kwargs["weighted"]
        self.hid_layers = self.__model_define_member_kwargs["hid_layers"]
        self.act_func_out_layer = self.__model_define_member_kwargs["act_func_out_layer"]
        self.dropout_rate = self.__model_define_member_kwargs["dropout_rate"]
        self.batch_norm = self.__model_define_member_kwargs["batch_norm"]
        self.kernel_initializer = self.__model_define_member_kwargs["kernel_initializer"]
        self.required_epochs = self.__model_train_member_kwargs["epochs"]
        self.batch_size = self.__model_train_member_kwargs["batch_size"]
        self.set_optimizer() # this defines the string optimizer_string and object optimizer
        self.set_loss() # this defines the string loss_string and the object loss
        self.set_metrics() # this defines the lists metrics_string and metrics
        self.set_callbacks() # this defines the lists callbacks_strings and callbacks

        ### Predictions and history dictionaries
        self.predictions = {}
        self.history = {}

        ### Reconstruct object state if loading from folder
        if self.DNNLik_input_folder is not None:
            self.__load_history()
            self.__load_model()
            self.__load_scalers()
            self.__load_indices()
            self.__load_predictions()
            self.model_params = summary_log['model_params']
            self.model_trainable_params = summary_log['model_trainable_params']
            self.model_non_trainable_params = summary_log['model_non_trainable_params']
            self.training_time = summary_log['training_time']
            self.final_epochs = summary_log['final_epochs']

    def __set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __set_dtype(self):
        if self.dtype is None:
            self.dtype = "float64"
        K.set_floatx(self.dtype)

    def __set_data_sample(self):
        if self.data_sample is None and self.data_sample_input_filename is None:
            raise Exception(
                "Either a DataSample object or a dataset input file name should be passed while you passed none.\nPlease input one and retry.")
        elif self.data_sample is None and self.data_sample_input_filename is not None:
            self.data_sample = Data_sample(data_X=None,
                                           data_Y=None,
                                           dtype=self.dtype,
                                           pars_pos_poi=None,
                                           pars_pos_nuis=None,
                                           pars_labels=None,
                                           test_fraction=None,
                                           name=None,
                                           data_sample_input_filename=self.data_sample_input_filename,
                                           data_sample_output_filename=None,
                                           load_on_RAM=self.load_on_RAM)
        if self.ensemble_name is None:
            self.ensemble_name = "DNNLikEnsemble_"+self.data_sample.name
        self.__check_npoints()

    def __check_npoints(self):
        available_points_tot = self.data_sample.npoints
        available_points_train = (1-self.data_sample.test_fraction)*available_points_tot
        available_points_test = self.data_sample.test_fraction*available_points_tot
        required_points_train = self.npoints_train+self.npoints_val
        required_points_test = self.npoints_test
        if required_points_train > available_points_train:
            raise Exception("Requiring more training points than available in data_sample. Please reduce npoints_train+npoints_val.")
        if required_points_test > available_points_test:
            raise Exception("Requiring more test points than available in data_sample. Please reduce npoints_test")
    
    def __check_create_ensemble_folder(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        utility.check_create_folder(self.ensemble_folder)

    def __check_create_member_results_folder(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        utility.check_create_folder(self.member_results_folder)

    def __load_summary_log(self):
        summary_log_files = []
        for _, _, f in os.walk(self.DNNLik_input_folder):
            for file in f:
                if "summary_log.json" in file:
                    summary_log_files.append(file)
        if len(summary_log_files) > 0:
            summary_log_file = os.path.join(self.DNNLik_input_folder, summary_log_files[-1])
            with open(summary_log_file) as json_file:
                summary_log = json.load(json_file)
            return summary_log
        else:
            return None

    def __load_history(self):
        try:
            with open(self.history_json_filename) as json_file:
                self.history = json.load(json_file)
        except:
            self.history = {}

    def __load_model(self):
        self.model = load_model(self.model_h5_filename, custom_objects={'R2_metric': self.R2_metric, 'Rt_metric': self.Rt_metric})

    def __load_scalers(self):
        self.scalerX = joblib.load(self.scalerX_jlib_filename)
        self.scalerY = joblib.load(self.scalerY_jlib_filename)

    def __load_indices(self):
        h5_in = h5py.File(self.idx_filename, 'r')
        data = h5_in.require_group("idx")
        self.idx_train = data["idx_train"][:]
        self.idx_val = data["idx_val"][:]
        self.data_sample.data_dictionary['idx_train'] = self.idx_train
        self.data_sample.data_dictionary['X_train'] = self.data_sample.data_X[self.idx_train].astype(self.dtype)
        self.data_sample.data_dictionary['Y_train'] = self.data_sample.data_Y[self.idx_train].astype(self.dtype)
        self.data_sample.data_dictionary['idx_val'] = self.idx_val
        self.data_sample.data_dictionary['X_val'] = self.data_sample.data_X[self.idx_val].astype(self.dtype)
        self.data_sample.data_dictionary['Y_val'] = self.data_sample.data_Y[self.idx_val].astype(self.dtype)
        try:
            self.idx_test = data["idx_test"][:]
            self.data_sample.data_dictionary['idx_test'] = self.idx_test
            self.data_sample.data_dictionary['X_test'] = self.data_sample.data_X[newlik.idx_test].astype(self.dtype)
            self.data_sample.data_dictionary['Y_test'] = self.data_sample.data_Y[newlik.idx_test].astype(self.dtype)
        except:
            pass
        h5_in.close()

    def __load_predictions(self):
        try:
            with open(self.predictions_json_filename) as json_file: 
                self.predictions = json.load(json_file)
        except:
            self.predictions = {}

    def get_available_gpus(self,verbose=False):
        self.available_gpus = set_resources.get_available_gpus(verbose=verbose)

    def get_available_cpu(self,verbose=False):
        self.available_cpu_cores = set_resources.get_available_cpu(verbose=verbose)

    def set_gpus(self, gpus_list, verbose=False):
        self.active_gpus = set_resources.set_gpus(gpus_list, verbose=verbose)
        if self.active_gpus != []:
            self.gpu_mode = True
        else:
            self.gpu_mode = False

    def compute_sample_weights(self, bins=100, power=1):
        # Generate weights
        self.W_train = self.data_sample.compute_sample_weights(self.Y_train, bins=bins, power=power).astype(self.dtype)
        self.W_val = self.data_sample.compute_sample_weights(self.Y_val, bins=bins, power=power).astype(self.dtype)
        #self.W_test = self.data_sample.compute_sample_weights(self.Y_test, bins=bins, power=power)
        #self.W_train = np.full(len(self.Y_train), 1)
        #self.W_val = np.full(len(self.Y_val), 1)
        #self.W_test = np.full(len(self.Y_test), 1)

    def define_scalers(self, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.scalerX, self.scalerY = self.data_sample.define_scalers(self.X_train, self.Y_train, self.scalerX_bool, self.scalerY_bool, verbose=verbose)

    def generate_train_data(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        # Generate data
        if self.same_data:
            self.data_sample.update_train_data(self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        else:
            self.data_sample.generate_train_data(self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        self.idx_train = self.data_sample.data_dictionary["idx_train"][:self.npoints_train]
        self.X_train = self.data_sample.data_dictionary["X_train"][:self.npoints_train].astype(self.dtype)
        self.Y_train = self.data_sample.data_dictionary["Y_train"][:self.npoints_train].astype(self.dtype)
        self.idx_val = self.data_sample.data_dictionary["idx_val"][:self.npoints_train]
        self.X_val = self.data_sample.data_dictionary["X_val"][:self.npoints_val].astype(self.dtype)
        self.Y_val = self.data_sample.data_dictionary["Y_val"][:self.npoints_val].astype(self.dtype)
        # Define scalers
        self.define_scalers()

    def generate_test_data(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        # Generate data
        self.data_sample.generate_test_data(self.npoints_test, verbose=verbose)
        self.idx_test = self.data_sample.data_dictionary["idx_test"][:self.npoints_train]
        self.X_test = self.data_sample.data_dictionary["X_test"][:self.npoints_test].astype(self.dtype)
        self.Y_test = self.data_sample.data_dictionary["Y_test"][:self.npoints_test].astype(self.dtype)

    def model_define(self,verbose=False):
        """
        Define the Keras Model 'self.model' and the attributes 'self.model_params', 'self.model_trainable_params', and
        'self.model_non_trainable_params'. Uses parameters from the dictionary 'self.__model_define_member_kwargs'.
        """
        global ShowPrints
        ShowPrints = verbose
        # Define model
        start = timer()
        inputLayer = Input(shape=(self.ndim,))
        if self.batch_norm:
            x = BatchNormalization()(inputLayer)
        if self.hid_layers[0][1] == 'selu':
            x = Dense(self.hid_layers[0][0], activation=self.hid_layers[0]
                      [1], kernel_initializer='lecun_normal')(inputLayer)
        else:
            x = Dense(self.hid_layers[0][0], activation=self.hid_layers[0]
                      [1], kernel_initializer=self.kernel_initializer)(inputLayer)
        if self.batch_norm:
            x = BatchNormalization()(x)
        if self.dropout_rate != 0:
            if self.hid_layers[0][1] == 'selu':
                x = AlphaDropout(self.dropout_rate)(x)
            else:
                x = Dropout(self.dropout_rate)(x)
        if len(self.hid_layers) > 1:
            for i in self.hid_layers[1:]:
                if i[1] == 'selu':
                    x = Dense(i[0], activation=i[1], kernel_initializer='lecun_normal')(x)
                else:
                    x = Dense(i[0], activation=i[1], kernel_initializer=self.kernel_initializer)(x)
                if self.batch_norm:
                    x = BatchNormalization()(x)
                if self.dropout_rate != 0:
                    if i[1] == 'selu':
                        x = AlphaDropout(self.dropout_rate)(x)
                    else:
                        x = Dropout(self.dropout_rate)(x)
        outputLayer = Dense(1, activation=self.act_func_out_layer)(x)
        self.model = Model(inputs=inputLayer, outputs=outputLayer)
        self.model_params = int(self.model.count_params())
        self.model_trainable_params = int(np.sum([K.count_params(p) for p in self.model.trainable_weights]))
        self.model_non_trainable_params = int(np.sum([K.count_params(p) for p in self.model.non_trainable_weights]))
        end = timer()
        print("Model for member",self.member_number,"defined in", end-start, "s.")
        if ShowPrints > 0:
            print(self.model.summary())

    def set_optimizer(self,verbose=False):
        """
        Set Keras Model optimizer. Uses parameters from the dictionary 'self.__model_optimizer_member_kwargs'.
        """
        global ShowPrints
        ShowPrints = verbose
        if type(self.__model_optimizer_member_kwargs["optimizer"]) is str:
            self.optimizer_string = self.__model_optimizer_member_kwargs["optimizer"]
            self.optimizer = self.optimizer_string
        if type(self.__model_optimizer_member_kwargs["optimizer"]) is dict:
            name = list(
                self.__model_optimizer_member_kwargs["optimizer"].keys())[0]
            string = name+"("
            for key, value in self.__model_optimizer_member_kwargs["optimizer"][name].items():
                if type(value) is str:
                    value = "'"+value+"'"
                string = string+str(key)+"="+str(value)+", "
            optimizer_string = str("optimizers."+string+")").replace(", )", ")")
            self.optimizer_string = optimizer_string
            self.optimizer = eval(optimizer_string)
        print("Optimizer set to:", self.optimizer_string)

    def set_loss(self, verbose=False):
        loss_string = self.__model_compile_member_kwargs["loss"]
        try:
            loss_obj = losses.deserialize(loss_string)
            print(loss_string, "ok.")
        except:
            print("Exception for",loss_string,".")
            loss_obj = eval("self."+loss_string)
        self.loss_string = loss_string
        self.loss = loss_obj

    def set_metrics(self, verbose=False):
        metrics_string = self.__model_compile_member_kwargs["metrics"]
        metrics_obj = list(range(len(metrics_string)))
        for i in range(len(metrics_string)):
            try:
                metrics_obj[i] = metrics.deserialize(metrics_string[i])
                print(metrics_string[i], "ok.")
            except:
                print("Exception for", metrics_string[i], ".")
                metrics_obj[i] = eval("self."+metrics_string[i])
        self.metrics_string = metrics_string
        self.metrics = metrics_obj

    def model_compile(self,verbose=False):
        """
        Compile the Keras Model 'self.model'. Uses parameters from the dictionary 'self.__model_compile_member_kwargs'.
        """
        global ShowPrints
        ShowPrints = verbose
        # Compile model
        start = timer()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        end = timer()
        print("Model for member",self.member_number,"compiled in",end-start,"s.")

    def set_callbacks(self, verbose=False):
        """
        Set Keras Model callbacks. Uses parameters from the dictionary 'self.__model_callbacks_member_kwargs'.
        """
        global ShowPrints
        ShowPrints = verbose
        callbacks_strings = []
        callbacks_name = [
            cb for cb in self.__model_callbacks_member_kwargs["callbacks"] if type(cb) is str]
        callbacks_dict = [
            cb for cb in self.__model_callbacks_member_kwargs["callbacks"] if type(cb) is dict]
        for cb in callbacks_name:
            if cb == "PlotLossesKeras":
                self.model_plot_losses_keras_filename = self.member_results_folder+"/"+self.member_name +"_figure_plot_losses_keras.pdf"
                string = "PlotLossesKeras(fig_path='" + self.model_plot_losses_keras_filename+"')"
            elif cb == "ModelCheckpoint":
                self.model_checkpoints_filename = self.member_results_folder+"/"+self.member_name+ "_checkpoint.{epoch:02d}-{val_loss:.2f}.h5"
                string = "callbacks.ModelCheckpoint(filepath='" + self.model_checkpoints_filename+"')"
            elif cb == "TensorBoard":
                self.tensorboard_log_dir = self.member_results_folder+"/"+"logs/fit"# +"/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
                utility.check_create_folder(self.member_results_folder+"/" + "logs")
                utility.check_create_folder(self.member_results_folder+"/" +"logs/fit")
                #utility.check_create_folder(self.tensorboard_log_dir)
                string = "callbacks.TensorBoard(log_dir='" + \
                    self.tensorboard_log_dir+"')"
            else:
                string = "callbacks."+cb+"()"
            callbacks_strings.append(string)
            print("Added callback:", string)
        for cb in callbacks_dict:
            key1 = list(cb.keys())[0]
            value1 = list(cb.values())[0]
            string = ""
            if key1 == "PlotLossesKeras":
                self.model_plot_losses_keras_filename = self.member_results_folder+"/"+self.member_name +"_figure_plot_losses_keras.pdf"
                string = "fig_path = '"+self.model_plot_losses_keras_filename + "', "
            elif key1 == "ModelCheckpoint":
                self.model_checkpoints_filename = self.member_results_folder+"/"+self.member_name+ "_checkpoint.{epoch:02d}-{val_loss:.2f}.h5"
                string = "filepath = '"+self.model_checkpoints_filename+"', "
                key1 = "callbacks."+key1
            elif key1 == "TensorBoard":
                self.tensorboard_log_dir = self.member_results_folder+"/"+"logs/fit"# +"/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                utility.check_create_folder(self.member_results_folder+"/" +"logs")
                utility.check_create_folder(self.member_results_folder+"/" +"logs/fit")
                #utility.check_create_folder(self.tensorboard_log_dir)
                string = "log_dir = '"+self.tensorboard_log_dir+"', "
                key1 = "callbacks."+key1
            else:
                key1 = "callbacks."+key1
            for key2, value2 in value1.items():
                if key2 == "monitor" and type(value2) is str:
                    if "val_" in value2:
                        value2 = value2.split("val_")[1]
                    if value2 == 'loss':
                        value2 = 'val_loss'
                    else:
                        value2 = "val_" + utility.metric_name_unabbreviate(value2)
                if type(value2) is str:
                    value2 = "'"+value2+"'"
                if not "path" in key2:
                    string = string+str(key2)+"="+str(value2)+", "
            string = str(key1+"("+string+")").replace(", )", ")")
            callbacks_strings.append(string)
            #callbacks.append(eval(string))
            print("Added callback:", string)
        if not "TerminateOnNaN" in str(callbacks_strings):
            callbacks_strings.append("callbacks.TerminateOnNaN()")
        self.callbacks_strings = callbacks_strings
        self.callbacks = [eval(callback) for callback in callbacks_strings]

    def model_build(self, gpu="auto", verbose=True):
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_tf = 0
        else:
            verbose_tf = verbose
        if self.gpu_mode:
            if gpu is "auto":
                gpu = 0
            elif gpu > len(self.available_gpus):
                print("gpu", gpu, "does not exist. Continuing on first gpu.")
                gpu = 0
            self.training_device = self.available_gpus[gpu]
            device_id = self.training_device[0]
        else:
            if gpu is not "auto":
                print(
                    "GPU mode selected without any active GPU. Proceeding with CPU support.")
            self.training_device = self.available_cpu
            device_id = self.training_device[0]
        strategy = tf.distribute.OneDeviceStrategy(device=device_id)
        print("Building tf model for member", self.member_number,
              "on device", self.training_device)
        with strategy.scope():
            #try:
            #    self.model
            #except:
            self.model_define(verbose=verbose_tf)
            #if not self.model._is_compiled:
            self.model_compile(verbose=verbose_tf)

    def model_train(self,verbose=2):
        """
        Train the Keras Model 'self.model'. Uses parameters from the dictionary 'self.__model_train_member_kwargs'.
        """
        global ShowPrints
        ShowPrints = verbose
        if verbose<0:
            verbose_tf=0
        else:
            verbose_tf=verbose
        # Scale data
        start = timer()
        #print("Checking data")
        try:
            self.X_train
        except:
            #print("Generate test data")
            self.generate_train_data()
        print("Scaling training data.")
        X_train = self.scalerX.transform(self.X_train)
        X_val = self.scalerX.transform(self.X_val)
        Y_train = self.scalerY.transform(self.Y_train.reshape(-1, 1)).reshape(len(self.Y_train))
        Y_val = self.scalerY.transform(self.Y_val.reshape(-1, 1)).reshape(len(self.Y_val))
        print([type(X_train),type(X_val),type(Y_train),type(Y_train)])
        # If PlotLossesKeras is in callbacks set plot style
        if "PlotLossesKeras" in str(self.callbacks_strings):
            try:
                jtplot.reset()
            except:
                pass
            try:
                plt.style.use(mplstyle_path)
            except:
                pass
        # Train model
        print("Start training of model for member",self.member_number, ".")
        if self.weighted:
            # Compute weights
            try:
                self.W_train
            except:
                print(
                    "In order to compute sample weights with the desired parameters please run the function\
                    self.compute_sample_weights(bins=100, power=1) before training.\n Proceding with sample weights\
                    computed with default parameters (bins=100 and power=1).")
                self.compute_sample_weights()
            # Train
            history = self.model.fit(X_train, Y_train, sample_weight=self.W_train, epochs=self.required_epochs, batch_size=self.batch_size, verbose=verbose_tf,
                    validation_data=(X_val, Y_val), callbacks=self.callbacks)
        else:
            history = self.model.fit(X_train, Y_train, epochs=self.required_epochs, batch_size=self.batch_size, verbose=verbose_tf,
                    validation_data=(X_val, Y_val), callbacks=self.callbacks)
        end = timer()
        self.training_time = end - start
        history = history.history
        for k, v in history.items():
            history[k] = list(np.array(v, dtype=self.dtype))
        if self.history == {}:
            print("no existing history")
            self.history = history
        else:
            print("existing history")
            for k, v in self.history.items():
                self.history[k] = v + history[k]
        self.final_epochs = len(self.history['loss'])
        if "PlotLossesKeras" in str(self.callbacks_strings):
            plt.close()
        ShowPrints = verbose
        print("Model for member",self.member_number,"successfully trained for",self.required_epochs, "epochs in", self.training_time,"s.")
        #print("Generating summary log")
        #self.generate_summary_log()

    def model_predict(self, X, batch_size=None, steps=None, verbose=False):
        """
        Predict with the Keras Model 'self.model'.
        """
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_tf = 0
        else:
            verbose_tf = verbose
        # Scale data
        if batch_size is None:
            batch_size = self.batch_size
        start = timer()
        print("Scaling data.")
        X = self.scalerX.transform(X)
        pred = self.scalerY.inverse_transform(self.model.predict(X, batch_size=batch_size, steps=steps, verbose=verbose_tf)).reshape(len(X))
        end = timer()
        prediction_time = end - start
        return [pred, prediction_time]

    def model_predict_scalar(self, x, steps=None, gpu="auto", verbose=False):
        """
        Predict with the Keras Model 'self.model'.
        """
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_2 = 0
        else:
            verbose_2 = verbose
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        pred = self.model_predict(x, batch_size=1, steps=None, verbose=verbose_2)[0][0]
        return pred

    def model_evaluate(self, X, Y, batch_size=1, steps=None, verbose=False):
        """
        Predict with the Keras Model 'self.model'.
        """
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_tf = 0
        else:
            verbose_tf = verbose
        # Scale data
        start = timer()
        print("Scaling data.")
        X = self.scalerX.transform(X)
        Y = self.scalerY.transform(Y.reshape(-1, 1)).reshape(len(Y))
        pred = self.model.evaluate(X, Y, batch_size=batch_size, verbose=verbose_tf)
        end = timer()
        prediction_time = end - start
        return [pred, prediction_time]

    def model_compute_predictions(self, 
                                  CI=inference.CI_from_sigma([inference.sigma_from_CI(0.5), 1, 2, 3]), 
                                  pars_list=None, 
                                  batch_size=None,
                                  verbose=-1,
                                  **HPDI_kwargs):
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose2 = 0
        else:
            verbose2 = verbose
        start_global = timer()
        start = timer()
        if pars_list is None:
            pars_list = self.data_sample.pars_pos_poi.tolist()
        else:
            pars_list = pars_list
        pars_labels = np.array(self.data_sample.pars_labels)[pars_list].tolist()
        if batch_size is None:
            batch_size = self.batch_size
        print('Compute predictions')
        try:
            self.X_train
        except:
            #print("Loading train data")
            self.generate_train_data()
        try:
            self.X_test
        except:
            #print("Generate test data")
            self.generate_test_data()
            self.save_test_data_indices()
        ShowPrints = verbose
        print('Evaluate all metrics on (scaled) train/val/test using best models')
        metrics_names = self.model.metrics_names
        metrics_names_train = [i+"_best" for i in self.model.metrics_names]
        metrics_names_val = ["val_"+i+"_best" for i in self.model.metrics_names]
        metrics_names_test = ["test_"+i+"_best" for i in self.model.metrics_names]
        metrics_train = self.model_evaluate(self.X_train, self.Y_train, batch_size=self.batch_size,verbose=verbose2)[0][0:len(metrics_names)]
        metrics_val = self.model_evaluate(self.X_val, self.Y_val, batch_size=self.batch_size,verbose=verbose2)[0][0:len(metrics_names)]
        metrics_test = self.model_evaluate(self.X_test, self.Y_test, batch_size=self.batch_size,verbose=verbose2)[0][0:len(metrics_names)]
        metrics_true = {**dict(zip(metrics_names_train, metrics_train)), **dict(zip(metrics_names_val, metrics_val)), **dict(zip(metrics_names_test, metrics_test))}
        self.predictions = {**self.predictions, **{"Metrics on scaled data": metrics_true}}
        ShowPrints = verbose
        print("Predict Y for train/val/test samples")
        Y_pred_train, _ = self.model_predict(self.X_train, batch_size=self.batch_size,verbose=verbose2)
        Y_pred_val, _ = self.model_predict(self.X_val, batch_size=self.batch_size,verbose=verbose2)
        Y_pred_test, prediction_time = self.model_predict(self.X_test, batch_size=self.batch_size,verbose=verbose2)
        self.predictions = {**self.predictions, **{"Prediction time": prediction_time}}
        ShowPrints = verbose
        print('Evaluate all metrics on (un-scaled) train/val/test using best models')
        metrics_names_train = [i+"_best_unscaled" for i in self.model.metrics_names]
        metrics_names_val = ["val_"+i+"_best_unscaled" for i in self.model.metrics_names]
        metrics_names_test = ["test_"+i +"_best_unscaled" for i in self.model.metrics_names]
        metrics_train_unscaled = [metric(self.Y_train,Y_pred_train).numpy() for metric in [self.loss]+self.metrics]
        metrics_val_unscaled = [metric(self.Y_val,Y_pred_val).numpy() for metric in [self.loss]+self.metrics]
        metrics_test_unscaled = [metric(self.Y_test,Y_pred_test).numpy() for metric in [self.loss]+self.metrics]
        metrics_unscaled = {**dict(zip(metrics_names_train, metrics_train_unscaled)), **dict(zip(metrics_names_val, metrics_val_unscaled)), **dict(zip(metrics_names_test, metrics_test_unscaled))}
        self.predictions = {**self.predictions, **{"Metrics on unscaled data": metrics_unscaled}}
        print("Compute exp(Y_true) and exp(Y_pred) for train/val/test samples")
        [Y_train_exp, Y_val_exp, Y_test_exp, Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [np.exp(Y) for Y in [self.Y_train, self.Y_val, self.Y_test, Y_pred_train, Y_pred_val, Y_pred_test]]
        end = timer()
        print("Prediction on ("+str(self.npoints_train)+","+str(self.npoints_val)+","+str(self.npoints_test)+")", "(train,val,test) points done in", end-start, "s.")
        print("Compute Bayesian inference benchmarks")
        start = timer()
        print("Computing weights (pred vs true) for reweighting of distributions")
        [W_train, W_val, W_test] = [utility.normalize_weights(W) for W in [Y_pred_train_exp/Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]]
        print("Computing HPDI (pred vs true) using reweighted distributions")
        #(data, intervals=0.68, weights=None, nbins=25, print_hist=False, optimize_binning=True)
        HPDI_result = {}
        for par in pars_list:
            [HPDI_train, HPDI_val, HPDI_test] = [inference.HPDI(X, CI, **HPDI_kwargs) for X in [self.X_train[:, par], self.X_val[:, par], self.X_test[:, par]]]
            [HPDI_pred_train, HPDI_pred_val, HPDI_pred_test] = [inference.HPDI(self.X_train[:, par], CI, W_train, **HPDI_kwargs), inference.HPDI(self.X_val[:, par], CI, W_val, **HPDI_kwargs), inference.HPDI(self.X_test[:, par], CI, W_test, **HPDI_kwargs)]
            HPDI_result[str(par)] = {"true": {"train": HPDI_train, "val": HPDI_val, "test": HPDI_test}, "pred":{"train": HPDI_pred_train, "val": HPDI_pred_val, "test": HPDI_pred_test}}
        HDPI_error = inference.HPDI_error(HPDI_result)
        if "HPDI" not in self.predictions:
            self.predictions['HPDI'] = {}
        if "HPDI_error" not in self.predictions:
            self.predictions['HPDI_error'] = {}
        self.predictions["HPDI"] = {**self.predictions["HPDI"], **HPDI_result}
        self.predictions["HPDI_error"] = {**self.predictions["HPDI_error"], **HDPI_error}
        #self.predictions = {**self.predictions, **{"HPDI": HPDI_result},**{"HPDI_error": HDPI_error}}
        print("Computing KS test between one-dimensional distributions (pred vs true) using reweighted distributions")
        KS_test_pred_train = [[inference.ks_w(self.X_test[:, q], self.X_train[:, q], np.ones(len(self.X_test)), W_train)] for q in range(len(self.X_train[0]))]
        KS_test_pred_val = [[inference.ks_w(self.X_test[:, q], self.X_val[:, q], np.ones(len(self.X_test)), W_val)] for q in range(len(self.X_train[0]))]
        KS_val_pred_test = [[inference.ks_w(self.X_val[:, q], self.X_test[:, q], np.ones(len(self.X_val)), W_test)] for q in range(len(self.X_train[0]))]
        KS_train_test = [[inference.ks_w(self.X_train[:, q], self.X_test[:, q], np.ones(len(self.X_train)), np.ones(len(self.X_train)))] for q in range(len(self.X_train[0]))]
        KS_test_pred_train_median = np.median(np.array(KS_test_pred_train)[:, 0][:, 1]).tolist()
        KS_test_pred_val_median = np.median(np.array(KS_test_pred_val)[:, 0][:, 1]).tolist()
        KS_val_pred_test_median = np.median(np.array(KS_val_pred_test)[:, 0][:, 1]).tolist()
        KS_train_test_median = np.median(np.array(KS_train_test)[:, 0][:, 1]).tolist()
        self.predictions = {**self.predictions, **{"KS": {"Test vs pred on train": KS_test_pred_train,
                                                          "Test vs pred on val": KS_test_pred_val,
                                                          "Val vs pred on test": KS_val_pred_test,
                                                          "Train vs pred on train": KS_train_test}},
                                                **{"KS medians": {"Test vs pred on train": KS_test_pred_train_median,
                                                                  "Test vs pred on val": KS_test_pred_val_median,
                                                                  "Val vs pred on test": KS_val_pred_test_median,
                                                                  "Train vs pred on train": KS_train_test_median}}}
        self.predictions = utility.convert_types_dict(self.predictions)
        # Sort nested dictionary by keys
        self.predictions = utility.sort_dict(self.predictions)
        end = timer()
        print('Bayesian inference benchmarks computed in', end-start, 's.')
        self.save_predictions_json(verbose=True)
        end_global = timer()
        print("All predictions done in",end_global-start_global,"s.")
        #[tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02,
        #    tmu_err_mean] = ["None", "None", "None", "None", "None", "None", "None"]
        #if FREQUENTISTS_RESULTS:
        #     print('Estimating frequentist inference')
        #     start_tmu = timer()
        #     blst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #     tmuexact = np.array(list(map(tmu, blst)))
        #     tmuDNN = np.array(list(map(lambda x: tmu_DNN(x, model, scalerX, scalerY), blst)))
        #     [tmusample001, tmusample005, tmusample01, tmusample02] = [np.array(list(map(lambda x: tmu_sample(x, X_train, Y_train, binsize), blst))) for binsize in [0.01, 0.05, 0.1, 0.2]]
        #     tmu_err_mean = np.mean(np.abs(tmuexact[:, -1]-tmuDNN[:, -1]))
        #     end_tmu = timer()
        #     print('Frequentist inference done in', start_tmu-end_tmu, 's.')
        #end_global = timer()
        #print('Total time for predictions:', end_global-start_global, 's')
        #return [metrics_true, metrics_scaled,
        #        #mean_error_train, mean_error_val, mean_error_test, min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
        #        #quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
        #        #one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
        #        #central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test,
        #        HPDI_train, HPDI_val, HPDI_test, HPDI_pred_train, HPDI_pred_val, HPDI_pred_test, one_sigma_HPDI_rel_err_train, one_sigma_HPDI_rel_err_val, one_sigma_HPDI_rel_err_test, one_sigma_HPDI_rel_err_train_test,
        #        KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median,
        #        tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time]

    def save_train_data_indices(self, verbose=True):
        """ Save indices to member_n_idx.h5 as h5 file
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving data indices to file",self.idx_filename)
        start = timer()
        if os.path.exists(self.idx_filename):
            os.remove(self.idx_filename)
        h5_out = h5py.File(self.idx_filename)
        h5_out.create_group(self.member_name)
        data = h5_out.create_group("idx")
        data["idx_train"] = self.idx_train
        data["idx_val"] = self.idx_val
        h5_out.close()
        end = timer()
        print(self.idx_filename, "created and saved in", end-start, "s.")

    def save_test_data_indices(self, verbose=True):
        """ Save indices to member_n_idx.h5 as h5 file
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving data indices to file", self.idx_filename)
        start = timer()
        h5_out = h5py.File(self.idx_filename)
        data = h5_out.require_group("idx")
        try:
            data["idx_test"] = self.idx_test
        except:
            pass
        h5_out.close()
        end = timer()
        print(self.idx_filename, "modified and saved in", end-start, "s.")

    def save_model_json(self, verbose=True):
        """ Save model to json
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving model to json file", self.model_json_filename)
        start = timer()
        model_json = self.model.to_json()
        with open(self.model_json_filename, "w") as json_file:
            json_file.write(model_json)
        end = timer()
        print(self.model_json_filename, "created and saved.", end-start, "s.")

    def save_model_h5(self, verbose=True):
        """ Save model to h5
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving model to h5 file", self.model_h5_filename)
        start = timer()
        self.model.save(self.model_h5_filename)
        end = timer()
        print(self.model_h5_filename,"created and saved.", end-start, "s.")

    def save_model_onnx(self, verbose=True):
        """ Save model to onnx
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving model to onnx file", self.model_onnx_filename)
        start = timer()
        onnx_model = keras2onnx.convert_keras(self.model, self.member_name)
        onnx.save_model(onnx_model, self.model_onnx_filename)
        end = timer()
        print(self.model_onnx_filename,"created and saved.", end-start, "s.")

    def save_history_json(self,verbose=True):
        """ Save summary log (history plus model specifications) to json
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving history to json file", self.history_json_filename)
        start = timer()
        history = self.history
        #for key in list(history.keys()):
        #    self.history[utility.metric_name_abbreviate(key)] = self.history.pop(key)
        new_hist = utility.convert_types_dict(history)
        with codecs.open(self.history_json_filename, 'w', encoding='utf-8') as f:
            json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)
        end = timer()
        print(self.history_json_filename, "created and saved.", end-start, "s.")

    def save_summary_log_json(self,verbose=True):
        """ Save summary log (history plus model specifications) to json
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving summary_log to json file", self.summary_log_json_filename)
        start = timer()
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        history = {**{'Date time': str(now)}, **utility.dic_minus_keys(self.__dict__,['data_sample', 'optimizer', 'loss',
                                                                                      'metrics', 'callbacks',
                                                                                      'history',  'predictions', 'model',
                                                                                      'idx_train', 'idx_val', 'idx_test',
                                                                                      'X_train', 'X_val', 'X_test',
                                                                                      'Y_train', 'Y_val', 'Y_test',
                                                                                      'scalerX', 'scalerY',
                                                                                      'DNNLik_input_folder']
                                                                        )}
        new_hist = utility.convert_types_dict(history)
        with codecs.open(self.summary_log_json_filename, 'w', encoding='utf-8') as f:
            json.dump(new_hist, f, separators=(',', ':'), indent=4)
        end = timer()
        print(self.summary_log_json_filename, "created and saved.", end-start, "s.")

    def save_predictions_json(self,verbose=True):
        """ Save summary log (history plus model specifications) to json
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving predictions to json file", self.predictions_json_filename)
        start = timer()
        #history = self.predictions
        #new_hist = utility.convert_types_dict(history)
        with codecs.open(self.predictions_json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, separators=(
                ',', ':'), sort_keys=True, indent=4)
        end = timer()
        print(self.predictions_json_filename, "created and saved.", end-start, "s.")

    def save_performance_log_json(self, verbose=True):
        """ Save performance log (metrics and figures of merit) to json
        """
        pass

    def save_scalers_jlib(self, verbose=True):
        """ Save scalers to jlib
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving scalers to jlib file", self.scalerX_jlib_filename)
        start = timer()
        joblib.dump(self.scalerX, self.scalerX_jlib_filename)
        joblib.dump(self.scalerY, self.scalerY_jlib_filename)
        end = timer()
        print(self.scalerX_jlib_filename, " createdverbose_tf and saved in", end-start, "s.")
        print(self.scalerY_jlib_filename, " created and saved in", end-start, "s.")

    def save_model_graph_pdf(self, verbose=True):
        """ Save model graph to pdf
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving model graph to pdf file", self.model_graph_pdf_filename)
        start = timer()
        png_file = os.path.splitext(self.model_graph_pdf_filename)[0]+".png"
        ShowPrints = False
        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=png_file)
        ShowPrints = verbose
        utility.make_pdf_from_img(png_file)
        try:
            os.remove(png_file)
        except:
            try:
                time.sleep(1)
                os.remove(png_file)
            except:
                print('Cannot remove png file',png_file,'.')
        end = timer()
        print(self.model_graph_pdf_filename," created and saved in", end-start, "s.")

    def model_store(self, verbose=True):
        """ Save all model information
        - data indices as hdf5 dataset
        - model in json format
        - model in h5 format (with weights)
        - model in onnx format
        - history, including summary log as json
        - scalers to jlib file
        - model graph to pdf
        """
        global ShowPrints
        ShowPrints = verbose
        self.save_train_data_indices(verbose=verbose)
        try:
            self.idx_test
            self.save_test_data_indices(verbose=verbose)
        except:
            pass
        self.save_model_json(verbose=verbose)
        self.save_model_h5(verbose=verbose)
        self.save_model_onnx(verbose=verbose)
        self.save_history_json(verbose=verbose)
        self.save_summary_log_json(verbose=verbose)
        self.save_scalers_jlib(verbose=verbose)
        self.save_model_graph_pdf(verbose=verbose)

    def R2_metric(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MSE_model =  K.sum(K.square( y_true-y_pred )) 
        MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return (1 - MSE_model/(MSE_baseline + K.epsilon()))

    def Rt_metric(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MAPE_model =  K.sum(K.abs( 1-y_pred/(y_true + K.epsilon()))) 
        MAPE_baseline = K.sum(K.abs( 1-K.mean(y_true)/(y_true+ K.epsilon()) ) ) 
        return ( 1 - MAPE_model/(MAPE_baseline + K.epsilon()))

    #def model_save_fig(self, folder,history,title,summary_text,metrics=['loss'], yscale='log',verbose=True):
    #    folder = folder.rstrip('/')
    #    modname = title.replace(": ", "_")
    #    metrics = np.unique(metrics)
    #    for metric in metrics:
    #        metric = metric_name_unabbreviate(metric)
    #        val_metric = 'val_'+ metric
    #        figname = modname + "_figure_training_"+ metric+".pdf"
    #        jtplot.reset()
    #        try:
    #            plt.style.use('matplotlib.mplstyle')
    #        except:
    #            plt.style.use(r"%s" % ('/'.join(folder.split('/')
    #                                        [:-1])+'/matplotlib.mplstyle'))
    #        #fig = plt.figure(1,figsize=(9.5,7))
    #        #ax = fig.add_subplot(111)
    #        #plt.figure()
    #        #ax.tick_params(axis='both', which='major', labelsize=10)
    #        #ax.tick_params(axis='both', which='minor', labelsize=8)
    #        if type(history) is dict:
    #            plt.plot(history[metric])
    #            plt.plot(history[val_metric])
    #        else:
    #            plt.plot(history.history[metric])
    #            plt.plot(history.history[val_metric])
    #        plt.yscale(yscale)
    #        plt.grid(linestyle="--", dashes=(5, 5))
    #        plt.title(r"%s" % title, fontsize=10)
    #        plt.xlabel(r"epoch")
    #        ylable = (metric.replace("_", "-"))
    #        plt.ylabel(r"%s" % ylable)
    #        plt.legend([r"training", r"validation"])
    #        plt.tight_layout()
    #        ax = plt.axes()
    #        x1, x2, y1, y2 = plt.axis()
    #        plt.text(0.965, 0.06, r"%s" % summary_text, fontsize=7, bbox=dict(facecolor="green", alpha=0.15,
    #                                                                          edgecolor='black', boxstyle='round,pad=0.5'), ha='right', ma='left', transform=ax.transAxes)
    #        plt.savefig(r"%s" % (folder + "/" + figname))
    #        if verbose:
    #            #plt.show()
    #            print(r"%s" % (folder + "/" + figname +
    #                           " created and saved."))
    #        plt.close()
