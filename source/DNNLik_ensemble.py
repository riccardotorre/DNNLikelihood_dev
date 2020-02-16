__all__ = ["DNNLik_ensemble"]

import numpy as np
import os
import shutil
from timeit import default_timer as timer
import time
import builtins
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                                     InputLayer, Concatenate, concatenate)
from multiprocessing import cpu_count
from joblib import parallel_backend, Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from . import utility
from .data_sample import Data_sample
from .DNNLik import DNNLik
from . import set_resources


ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

class DNNLik_ensemble(object):
    def __init__(self,
                 ensemble_name=None,
                 data_sample=None,
                 data_sample_input_filename=None,
                 ensemble_results_folder=None,
                 load_on_RAM=None,
                 seed=1,
                 same_data=True,
                 model_data_ensemble_kwargs=None,
                 model_define_ensemble_kwargs=None,
                 model_optimizers_ensemble_kwargs=None,
                 model_compile_ensemble_kwargs=None,
                 model_callbacks_ensemble_kwargs=None,
                 model_train_ensemble_kwargs=None
                 ):
        ############ Initialize input parameters
        #### Set main inputs and DataSample
        self.ensemble_name = ensemble_name
        self.data_sample = data_sample
        self.data_sample_input_filename = data_sample_input_filename
        self.ensemble_results_folder = ensemble_results_folder
        self.load_on_RAM = load_on_RAM
        self.seed = seed
        self.same_data = same_data
        self.__set_data_sample() # This also fixes self.ndim and self.ensemble_name if not given
        self.__set_ensemble_results_folder() 
        self.__model_data_ensemble_kwargs = model_data_ensemble_kwargs
        self.__model_define_ensemble_kwargs = model_define_ensemble_kwargs
        self.__model_optimizers_ensemble_kwargs = model_optimizers_ensemble_kwargs
        self.__model_compile_ensemble_kwargs = model_compile_ensemble_kwargs
        self.__model_callbacks_ensemble_kwargs = model_callbacks_ensemble_kwargs
        self.__model_train_ensemble_kwargs = model_train_ensemble_kwargs
        self.members = {}
        self.stacks = {}
        self.get_available_gpus()

        #### Set model_data_ensemble_kwargs
        # example: model_data_ensemble_kwargs = {'npoints_list': [[1000,300,600],[2000,600,1000],[3000,1000,1500]]}
        self.__check_model_data_ensemble_kwargs()
        self.model_data_ensemble_kwargs_list = list(utility.product_dict(**self.__model_data_ensemble_kwargs))
        self.model_data_ensemble_kwargs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.model_data_ensemble_kwargs_list]

        #### Set model_define_ensemble_kwargs
        # example: model_define_ensemble_kwargs={"hid_layers_list": [[[50, "selu"], [50, "selu"]],[[100, "selu"], [100, "selu"]], ...],
        #                                        "act_func_out_layer_list": ["linear"], 
        #                                        "dropout_rate_list": [0.1], 
        #                                        "batch_norm_list": [True], 
        #                                        "kernel_initializer_list": ['glorot_uniform']}
        self.__check_model_define_ensemble_kwargs()
        self.model_define_ensemble_kwargs_list = list(utility.product_dict(**self.__model_define_ensemble_kwargs))
        self.model_define_ensemble_kwargs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.model_define_ensemble_kwargs_list]

        #### Set model_optimizers_ensemble_kwargs
        # example: model_optimizers_ensemble_kwargs={"optimizers_list": [{"Adam": {"learning_rate": 0.001,
        #                                                                          "beta_1": 0.9,
        #                                                                          "beta_2": 0.999,
        #                                                                          "amsgrad": False}},
        #                                                                {"SGD": {"learning_rate": 0.01,
        #                                                                         "momentum": 0.0, 
        #                                                                         "nesterov": False}}]}
        self.__check_model_optimizers_ensemble_kwargs()
        self.model_optimizers_ensemble_kwargs_list = list(utility.product_dict(**self.__model_optimizers_ensemble_kwargs))
        self.model_optimizers_ensemble_kwargs_list = [{k.replace("optimizers_list", "optimizer"): v for k, v in i.items()} for i in self.model_optimizers_ensemble_kwargs_list]

        #### Set model_compile_ensemble_kwargs
        # example: model_compile_ensemble_kwargs={"losses_list": ["mae","mse",...],
        #                                         "optimizers_list": [optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-10, decay=0.0, amsgrad=False),...],
        #                                         "metrics":["mse","msle",...]}
        self.__check_model_compile_ensemble_kwargs()
        self.model_compile_ensemble_kwargs_list = list(utility.product_dict(**utility.dic_minus_keys("metrics", self.__model_compile_ensemble_kwargs)))
        self.model_compile_ensemble_kwargs_list = [{**{k.replace("_list", "").replace("losses", "loss"): v for k, v in i.items()}, **{
            "metrics": self.__model_compile_ensemble_kwargs["metrics"]}} for i in self.model_compile_ensemble_kwargs_list]

        #### Set model_callbacks_ensemble_kwargs
        # example: model_callbacks_ensemble_kwargs={"callbacks_list": [{"PlotLossesKeras": True,
        #                                                          "EarlyStopping": {"monitor": model_compile_ensemble_kwargs["metrics"],
        #                                                                            "mode": "min",
        #                                                                            ...},
        #                                                          "ReduceLROnPlateau": {"monitor": model_compile_ensemble_kwargs["metrics"],                                                                   "mode": "min",
        #                                                                                "mode": "min",
        #                                                                                ...},
        #                                                          "ModelCheckpoint": {"monitor": model_compile_ensemble_kwargs["metrics"],
        #                                                                              ...},
        #                                                          "TerminateOnNaN": True,
        #                                                          ...}]
        self.__check_model_callbacks_ensemble_kwargs()
        #self.__import_callbacks()
        self.model_callbacks_ensemble_kwargs_list = [
            {"callbacks": i} for i in self.__model_callbacks_ensemble_kwargs["callbacks_list"]]

        #### Set model_train_ensemble_kwargs
        # example: model_train_ensemble_kwargs={"epochs_list": [200,1000],
        #                                       "batch_size_list": [512,1024,2048],
        self.__check_model_train_ensemble_kwargs()
        self.model_train_ensemble_kwargs_list = list(utility.product_dict(**self.__model_train_ensemble_kwargs))
        self.model_train_ensemble_kwargs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.model_train_ensemble_kwargs_list]

    def __set_data_sample(self):
        if self.data_sample is not None and self.data_sample_input_filename is not None:
            print("Either a DataSample object or a dataset input file name should be passed while you passed both.\nPlease input only one and retry.")
            return
        elif self.data_sample is None and self.data_sample_input_filename is None:
            print("Either a DataSample object or a dataset input file name should be passed while you passed none.\nPlease input one and retry.")
            return
        elif self.data_sample is None and self.data_sample_input_filename is not None:
            self.data_sample = Data_sample(data_X=None,
                                           data_Y=None,
                                           name=None,
                                           data_sample_input_filename=self.data_sample_input_filename,
                                           data_sample_output_filename=None,
                                           load_on_RAM=self.load_on_RAM)
        if self.ensemble_name is None:
            self.ensemble_name = "DNNLikEnsemble_"+self.data_sample.name
        self.ndim = self.data_sample.ndim
        self.n_available_points = self.data_sample.npoints

    def __set_ensemble_results_folder(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        if self.ensemble_results_folder is not None:
            self.ensemble_results_folder = utility.check_rename_folder(self.ensemble_results_folder).replace('\\', '/')
        else:
            self.ensemble_results_folder = utility.check_rename_folder(os.getcwd().replace('\\', '/')+"/"+self.ensemble_name)
        os.mkdir(self.ensemble_results_folder)
        print("All results will be saved in the folder", self.ensemble_results_folder)

    def __check_model_data_ensemble_kwargs(self):
        try:
            self.__model_data_ensemble_kwargs["npoints_list"]
        except:
            print(
                "model_data_kwargs dictionary should contain at least a keyword 'npoints_list'. \
                It may also contain the keys 'scaleX_list', 'scaleY_list', and 'weighted_list'. In case these are not present by \
                default they are set to {'scaleX': False}, {'scaleY': False}, and {'weighted_list':[False]}.")
        for npoints in self.__model_data_ensemble_kwargs["npoints_list"]:
            if npoints[1] <= 1:
                npoints[1] = round(npoints[1]*npoints[0])
            if npoints[2] <= 1:
                npoints[2] = round(npoints[2]*npoints[0])
        try:
            self.__model_train_ensemble_kwargs["scaleX_list"]
        except:
            self.__model_train_ensemble_kwargs["scaleX_list"] = [False]
        try:
            self.__model_train_ensemble_kwargs["scaleY_list"]
        except:
            self.__model_train_ensemble_kwargs["scaleY_list"] = [False]
        try:
            self.__model_train_ensemble_kwargs["weighted_list"]
        except:
            self.__model_train_ensemble_kwargs["weighted_list"] = [False]

    def __check_model_define_ensemble_kwargs(self):
        try:
            self.__model_define_ensemble_kwargs["hid_layers_list"]
        except:
            print(
                "model_define_kwargs dictionary should contain at least the keyword 'hid_layers_list'. \
                It may also contain keys 'act_func_out_layer_list', 'dropout_rate_list', 'batch_norm_list', and 'kernel_initializer_list'. \
                In case these are not present by default they are set to {'act_func_out_layer_list': ['linear']}, {'batch_norm_list': [False]},\
                {'dropout_rate_list': [0]}, {'kernel_initializers': ['glorot_uniform']}. In case of selu activation Dropout layers are replaced \
                with AlphaDropout and kernel_initializers is set to ['lecun_normal'].")
        try:
            self.__model_define_ensemble_kwargs["act_func_out_layer_list"]
        except:
            self.__model_define_ensemble_kwargs["act_func_out_layer_list"] = ["linear"] # Batch normalization layers are added between each pair of layers
        try:
            self.__model_define_ensemble_kwargs["batch_norm_list"]
        except:
            # Batch normalization layers are added between each pair of layers
            self.__model_define_ensemble_kwargs["batch_norm_list"] = [False]
        try:
            self.__model_define_ensemble_kwargs["dropout_rate_list"]
        except:
            # The same dropout is performed between each pair of layers
            self.__model_define_ensemble_kwargs["dropout_rate_list"] = [0]
        try:
            self.__model_define_ensemble_kwargs["kernel_initializer_list"]
        except:
            self.__model_define_ensemble_kwargs["kernel_initializer_list"] = [
                'glorot_uniform']  # The same dropout is performed between each pair of layers
        
    def __check_model_optimizers_ensemble_kwargs(self):
        #  example: model_optimizers_ensemble_kwargs = {"optimizers_list": [{"Adam": {"learning_rate": 0.001,
        #                                                                             "beta_1": 0.9,
        #                                                                             "beta_2": 0.999,
        #                                                                             "amsgrad": False}},
        #                                                                   {"SGD": {"learning_rate": 0.01,
        #                                                                            "momentum": 0.0, 
        #                                                                            "nesterov": False}}]}
        try:
            self.__model_optimizers_ensemble_kwargs["optimizers_list"]
        except:
            print(
                "model_optimizers_kwargs dictionary should contain at least one keyword 'optimizers_list'")
            raise
        
    def __check_model_compile_ensemble_kwargs(self):
        try:
            self.__model_compile_ensemble_kwargs["losses_list"]
        except:
            print(
                "model_compile_kwargs dictionary should contain at least the keyword 'losses_list'. \
                It may also contain the key 'metrics'. \
                In case this is not present by default it is set to {'metrics': (value of losses_list)}.")
        try:
            self.__model_compile_ensemble_kwargs["metrics"]
        except:
            self.__model_compile_ensemble_kwargs["metrics"] = self.__model_compile_ensemble_kwargs["losses_list"]
    
    def __check_model_callbacks_ensemble_kwargs(self):
        #  example: model_callbacks_ensemble_kwargs = {"callbacks_list": [{"PlotLossesKeras": True,
        #                                                             "EarlyStopping": {"monitor": model_compile_ensemble_kwargs["metrics"],
        #                                                                               ...},
        #                                                             "ReduceLROnPlateau": {"monitor": model_compile_ensemble_kwargs["metrics"],                                                                   "mode": "min",
        #                                                                                   ...},
        #                                                             "ModelCheckpoint": {"monitor": model_compile_ensemble_kwargs["metrics"],
        #                                                                                 ...},
        #                                                             "TerminateOnNaN": True,
        #                                                             ...}]
        try:
            self.__model_callbacks_ensemble_kwargs["callbacks_list"]
        except:
            self.__model_callbacks_ensemble_kwargs["callbacks_list"] = [["TerminateOnNaN"]]

    def __check_model_train_ensemble_kwargs(self):
        # example: model_fit_kwargs_list={"epochs_list": [200,1000],
        #                                 "batch_size_list": [512,1024,2048],
        try:
            self.__model_train_ensemble_kwargs["epochs_list"]
            self.__model_train_ensemble_kwargs["batch_size_list"]
        except:
            print(
                "model_fit_kwargs dictionary should contain the keywords 'epochs_list' and 'batch_size_list'.")

    def __import_callbacks(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        list_of_callbacks = []
        for i in self.__model_callbacks_ensemble_kwargs["callbacks_list"]:
            list_of_callbacks = list_of_callbacks+list(i.keys())
        list_of_callbacks = list(dict.fromkeys(list_of_callbacks))
        list_of_callbacks_done = []
        for string in list_of_callbacks:
            try:
                exec(string)
                print(string, " correctly imported")
                list_of_callbacks_done.append(string)
            except:
                pass
            try:
                exec("from keras.callbacks import "+string)
                print(string, "correctly imported from 'keras.callbacks'")
                list_of_callbacks_done.append(string)
            except:
                pass
            try:
                exec("from livelossplot import "+string)
                print(string, "correctly imported from 'livelossplot'")
                list_of_callbacks_done.append(string)
            except:
                pass
        list_of_callbacks_failed = [item for item in list_of_callbacks if item not in list_of_callbacks_done]
        if not list_of_callbacks_failed == []:
            print("Import of modules", str(list_of_callbacks), "failed.")

    def get_available_gpus(self):
        self.available_gpus = set_resources.get_available_gpus()

    def get_available_cpus(self):
        self.available_cpu_cores = set_resources.get_available_cpus()

    def setGPUs(self, n, multi_gpu=False):
        self.available_gpus = set_resources.setGPUs(n, multi_gpu=multi_gpu)

    def generate_member(self, n, seed, model_data_member_kwargs, model_define_member_kwargs, model_optimizer_member_kwargs, model_compile_member_kwargs, model_callbacks_member_kwargs, model_train_member_kwargs, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        self.members[n] = DNNLik(ensemble_name=self.ensemble_name,
                                 member_number=n,
                                 data_sample=self.data_sample,
                                 ensemble_results_folder=self.ensemble_results_folder,
                                 seed=seed,
                                 same_data=self.same_data,
                                 model_data_member_kwargs=model_data_member_kwargs,
                                 model_define_member_kwargs=model_define_member_kwargs,
                                 model_optimizer_member_kwargs=model_optimizer_member_kwargs,
                                 model_compile_member_kwargs=model_compile_member_kwargs,
                                 model_callbacks_member_kwargs=model_callbacks_member_kwargs,
                                 model_train_member_kwargs=model_train_member_kwargs
                                 )
        end = timer()
        ShowPrints = verbose
        print("DNN Likelihood member",str(n),"created in",end-start,"s.")

    def generate_members(self,verbose=-1):
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_2 = 0
        else:
            verbose_2 = verbose
        start = timer()
        all_kwargs = [[A, B, C, D, E, F] for A in self.model_data_ensemble_kwargs_list for B in self.model_define_ensemble_kwargs_list for C in self.model_optimizers_ensemble_kwargs_list for D in self.model_compile_ensemble_kwargs_list for E in self.model_callbacks_ensemble_kwargs_list for F in self.model_train_ensemble_kwargs_list]
        self.n_members = len(all_kwargs)
        if self.same_data:
            self.seeds = np.full(self.n_members,self.seed)
        else:
            np.random.seed(self.seed)
            self.seeds = np.random.choice(np.arange(100*self.n_members),self.n_members,replace=False)
        for i in range(self.n_members):
            model_data_member_kwargs, model_define_member_kwargs, model_optimizer_member_kwargs, model_compile_member_kwargs, model_callbacks_member_kwargs, model_train_member_kwargs = all_kwargs[i]
            self.generate_member(i, 
                                 self.seeds[i], 
                                 model_data_member_kwargs, 
                                 model_define_member_kwargs,
                                 model_optimizer_member_kwargs,
                                 model_compile_member_kwargs, 
                                 model_callbacks_member_kwargs,
                                 model_train_member_kwargs, 
                                 verbose=verbose_2)
        end = timer()
        ShowPrints = verbose
        print(self.n_members,"members (DNNLikelihoods) generated in", end-start, "s.")
        print("Results for member 'n' will be saved in the folders",self.ensemble_name,"_member_n.")

    def generate_data_members(self, members_list="all",force=False,verbose=False):
        global ShowPrints
        ShowPrints = verbose
        if verbose < 0:
            verbose_2 = 0
        else:
            verbose_2 = verbose
        start = timer()
        if members_list is "all":
            members_list = list(range(self.n_members))
        if self.same_data:
            proceed = True
        else:
            if not force:
                ShowPrints = True
                print("You are attempting to generate different data for", len(members_list),
                      "models. Depending on your parameters this may be time/memory consuming.\
                      Please consider generating data on the fly when training. If you wish to \
                      force this operation add the flag 'force=True'.")
                ShowPrints = verbose
                proceed = False
            else:
                proceed = True
        if proceed:
            print("Generating data may require some time, depending on the required number of samples.")
            for i in members_list:
                self.members[i].generate_data(verbose=verbose_2)
            end = timer()
            ShowPrints = verbose
            print("Data for", len(members_list), "models generated in",end-start,"s.")

    def train_member_on_device(self, member, gpu=None, verbose=2):
        global ShowPrints
        ShowPrints = verbose
        if gpu is None:
            gpu = 0
        elif gpu > len(self.available_gpus):
            print("gpu", gpu, "does not exist. Continuing on first gpu.")
            gpu = 0
        device = self.available_gpus[gpu][0]
        strategy = tf.distribute.OneDeviceStrategy(device=device)
        time.sleep(0.1)
        with strategy.scope():
            self.members[member].model_define(verbose=verbose)
            self.members[member].model_compile(verbose=verbose)
            self.members[member].model_train(verbose=verbose)
            self.members[member].model_store(verbose=verbose)
            #tf.compat.v1.reset_default_graph()
        ############ Should save history somewhere and close the tf.session to free up GPU memory

    def train_members_in_parallel_joblib(self, members_list, gpus_list="all", verbose=2):
        """At the moment this does not work. Use train_members_in_parallel_concurrent instead."""
        global ShowPrints
        ShowPrints = verbose
        if gpus_list is "all":
            gpus_list = list(range(len(self.available_gpus)))
        if len(members_list) < len(gpus_list):
            gpus_list = gpus_list[:len(members_list)]
        members_chunks = utility.chunks(members_list, len(gpus_list))
        for chunk in members_chunks:
            if len(chunk) < len(gpus_list):
                gpus_list = gpus_list[:len(chunk)]
            gpu_member_map_dictionary = dict(zip(np.array(gpus_list), np.array(chunk)))
            with parallel_backend('threading', n_jobs=len(gpus_list)):
                Parallel()(delayed(self.train_member_on_device)
                        (member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose) for gpu in gpus_list)

    def train_members_in_parallel_concurrent(self, members_list, gpus_list="all", verbose=2):
        """Function that trains and stores members in parallel."""
        global ShowPrints
        ShowPrints = verbose
        if gpus_list is "all":
            gpus_list = list(range(len(self.available_gpus)))
        if len(members_list) < len(gpus_list):
            gpus_list = gpus_list[:len(members_list)]
        members_chunks = utility.chunks(members_list, len(gpus_list))
        for chunk in members_chunks:
            if len(chunk) < len(gpus_list):
                gpus_list = gpus_list[:len(chunk)]
            gpu_member_map_dictionary = dict(zip(np.array(gpus_list), np.array(chunk)))
            with ThreadPoolExecutor(len(gpus_list)) as executor:
                executor.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_list)

    ###### In the future define the DNNStack object to contain stacked models (plus all related methods)
    def model_define_stacked(self, members_list):
        members = [self.members[i] for i in members_list]
        for i in range(len(members)):
            members.append(self.members[i])
            try:
                model = members[i].model
            except:
                print("Model for member",i,"is not defined. Please define and train members before stacking them.")
            for layer in model.layers:
                layer.trainable = False
                layer._name = "stack_" + str(i+1) + '_' + layer.name
        ensemble_visible = [members[i].model.input for i in range(len(members))]
        ensemble_outputs = [members[i].model.output for i in range(len(members))]
        merge = concatenate(ensemble_outputs)
        hidden = Dense(8, activation='selu')(merge)
        output = Dense(1, activation='linear')(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)
        self.stacks[str(members_list)] = model
