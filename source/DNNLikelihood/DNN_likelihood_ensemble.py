__all__ = ["DnnLikEnsemble"]

import json
#import ndjson as json
import codecs
import numpy as np
import os
import shutil
from timeit import default_timer as timer
import time
from datetime import datetime
import re
import builtins
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                                     InputLayer, Concatenate, concatenate)
from multiprocessing import cpu_count
from joblib import parallel_backend, Parallel, delayed
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from . import utils
from .data import Data
from .dnn_likelihood import DnnLik
from .show_prints import print
from .resources import Resources

class DnnLikEnsemble(Resources): #show_prints.Verbosity inherited from resources.Resources
    def __init__(self,
                 DNNLik_ensemble_input_folder=None,
                 ensemble_name=None,
                 data=None,
                 input_data_file=None,
                 ensemble_folder=None,
                 load_on_RAM=False,
                 seed=1,
                 dtype = None,
                 same_data=True,
                 model_data_ensemble_inputs=None,
                 model_define_ensemble_inputs=None,
                 model_optimizers_ensemble_inputs=None,
                 model_compile_ensemble_inputs=None,
                 model_callbacks_ensemble_kwargs=None,
                 model_train_ensemble_kwargs=None,
                 gpus_id_list='all',
                 verbose=True
                 ):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #### Set model date time
        self.ensemble_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        #### Set resources
        self.get_available_gpus(verbose=True)
        self.get_available_cpu(verbose=True)
        self.set_gpus(gpus_id_list,verbose=True)
        ############ Check wheather to create a new DNNLik_ensemble object from inputs or from files
        self.DNNLik_ensemble_input_folder = DNNLik_ensemble_input_folder
        if self.DNNLik_ensemble_input_folder is None:
            ############ Initialize input parameters from arguments
            #### Set main inputs and DataSample
            self.ensemble_name = ensemble_name
            self.data = data
            self.input_data_file = os.path.abspath(input_data_file)
            self.ensemble_folder = ensemble_folder
            self.load_on_RAM = load_on_RAM
            self.seed = seed
            if dtype is None:
                self.dtype = "float64"
            else:
                self.dtype = dtype
            self.same_data = same_data
            self.__set_seed()
            self.__set_dtype()
            self.__set_data()
            self.__set_ensemble_name()
            self.__set_ensemble_folder()
            self.__set_ensemble_results_folder()
            self.__model_data_ensemble_inputs = model_data_ensemble_inputs
            self.__model_define_ensemble_inputs = model_define_ensemble_inputs
            self.__model_optimizers_ensemble_inputs = model_optimizers_ensemble_inputs
            self.__model_compile_ensemble_inputs = model_compile_ensemble_inputs
            self.__model_callbacks_ensemble_inputs = model_callbacks_ensemble_kwargs
            self.__model_train_ensemble_inputs = model_train_ensemble_kwargs
        else:
            ############ Initialize input parameters from file
            #### Load summary_log dictionary
            print("When providing DNNLik input folder all arguments but load_on_RAM and dtype are ignored and the object is constructed from saved data")
            summary_log = self.__load_summary_log()

            #### Set main inputs and DataSample
            self.ensemble_name = summary_log['ensemble_name']
            self.data = None
            self.input_data_file = summary_log['input_data_file']
            self.ensemble_folder = summary_log['ensemble_folder']
            self.load_on_RAM = load_on_RAM
            self.seed = summary_log['seed']
            self.dtype = dtype
            if self.dtype is None:
                self.dtype = summary_log['dtype']
            self.same_data = summary_log['same_data']
            self.__set_seed()
            self.__set_dtype()
            self.__set_data() # This also fixes self.ndims and self.ensemble_name if not given
            self.ensemble_folder = summary_log['ensemble_folder']
            self.ensemble_results_folder = summary_log['ensemble_results_folder']
            self.__model_data_ensemble_inputs = summary_log['_DNNLik_ensemble__model_data_ensemble_inputs']
            self.__model_define_ensemble_inputs = summary_log['_DNNLik_ensemble__model_define_ensemble_inputs']
            self.__model_optimizers_ensemble_inputs = summary_log['_DNNLik_ensemble__model_optimizers_ensemble_inputs']
            self.__model_compile_ensemble_inputs = summary_log['_DNNLik_ensemble__model_compile_ensemble_inputs']
            self.__model_callbacks_ensemble_inputs = summary_log['_DNNLik_ensemble__model_callbacks_ensemble_inputs']
            self.__model_train_ensemble_inputs = summary_log['_DNNLik_ensemble__model_train_ensemble_inputs']
            self.n_members = summary_log['n_members']
            self.seeds = np.array(summary_log['seeds'])

        #### Set other attributes
        self.members = {}
        self.stacks = {}
        self.summary_log_json_filename = self.ensemble_results_folder + \
            "/"+self.ensemble_name+"_summary_log.json"

        #### Set model_data_ensemble_inputs
        # example: model_data_ensemble_inputs = {'npoints_list': [[1000,300],[2000,600],[3000,1000]]}
        self.__check_model_data_ensemble_inputs()
        self.__model_data_ensemble_inputs_list = list(utils.product_dict(**self.__model_data_ensemble_inputs))
        self.__model_data_ensemble_inputs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.__model_data_ensemble_inputs_list]

        #### Set model_define_ensemble_inputs
        # example: model_define_ensemble_inputs={"hidden_layers_list": [[[50, "selu"], [50, "selu"]],[[100, "selu"], [100, "selu"]], ...],
        #                                        "act_func_out_layer_list": ["linear"], 
        #                                        "dropout_rate_list": [0.1], 
        #                                        "batch_norm_list": [True], 
        #                                        "kernel_initializer_list": ['glorot_uniform']}
        self.__check_model_define_ensemble_inputs()
        self.__model_define_ensemble_inputs_list = list(utils.product_dict(**self.__model_define_ensemble_inputs))
        self.__model_define_ensemble_inputs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.__model_define_ensemble_inputs_list]

        #### Set model_optimizers_ensemble_inputs
        # example: model_optimizers_ensemble_inputs={"optimizers_list": [{"Adam": {"learning_rate": 0.001,
        #                                                                          "beta_1": 0.9,
        #                                                                          "beta_2": 0.999,
        #                                                                          "amsgrad": False}},
        #                                                                {"SGD": {"learning_rate": 0.01,
        #                                                                         "momentum": 0.0, 
        #                                                                         "nesterov": False}}]}
        #### TO BE REWRITTEN
        #self.__check_model_optimizers_ensemble_inputs()
        self.__model_optimizers_ensemble_inputs_list = self.__model_optimizers_ensemble_inputs
        #list(utils.product_dict(**self.__model_optimizers_ensemble_inputs))
        #self.__model_optimizers_ensemble_inputs_list = [{k.replace("optimizers_list", "optimizer"): v for k, v in i.items()} for i in self.__model_optimizers_ensemble_inputs_list]

        #### Set model_compile_ensemble_inputs
        # example: model_compile_ensemble_inputs={"losses_list": ["mae","mse",...],
        #                                         "optimizers_list": [optimizers.Adam(lr=0.001, beta_1=0.95, beta_2=0.999, epsilon=1e-10, decay=0.0, amsgrad=False),...],
        #                                         "metrics":["mse","msle",...]}
        self.__check_model_compile_ensemble_inputs()
        self.__model_compile_ensemble_inputs_list = list(utils.product_dict(**utils.dic_minus_keys(self.__model_compile_ensemble_inputs,"metrics")))
        self.__model_compile_ensemble_inputs_list = [{**{k.replace("_list", "").replace("losses", "loss"): v for k, v in i.items()}, **{
            "metrics": self.__model_compile_ensemble_inputs["metrics"]}} for i in self.__model_compile_ensemble_inputs_list]

        #### Set model_callbacks_ensemble_kwargs
        # example: model_callbacks_ensemble_kwargs={"callbacks_list": [{"PlotLossesKeras": True,
        #                                                          "EarlyStopping": {"monitor": model_compile_ensemble_inputs["metrics"],
        #                                                                            "mode": "min",
        #                                                                            ...},
        #                                                          "ReduceLROnPlateau": {"monitor": model_compile_ensemble_inputs["metrics"],                                                                   "mode": "min",
        #                                                                                "mode": "min",
        #                                                                                ...},
        #                                                          "ModelCheckpoint": {"monitor": model_compile_ensemble_inputs["metrics"],
        #                                                                              ...},
        #                                                          "TerminateOnNaN": True,
        #                                                          ...}]
        #### TO BE REWRITTEN
        #self.__check_model_callbacks_ensemble_kwargs()
        self.__model_callbacks_ensemble_inputs_list = self.__model_callbacks_ensemble_inputs
        #[{"callbacks": i} for i in self.__model_callbacks_ensemble_inputs["callbacks_list"]]

        #### Set model_train_ensemble_kwargs
        # example: model_train_ensemble_kwargs={"epochs_list": [200,1000],
        #                                       "batch_size_list": [512,1024,2048],
        self.__check_model_train_ensemble_kwargs()
        self.__model_train_ensemble_inputs_list = list(utils.product_dict(**self.__model_train_ensemble_inputs))
        self.__model_train_ensemble_inputs_list = [{k.replace("_list", ""): v for k, v in i.items()} for i in self.__model_train_ensemble_inputs_list]

        #### Generate or import members (and save summary_log)
        #
        if self.DNNLik_ensemble_input_folder is None:
            self.generate_members(verbose=-1)
        else:
            self.__import_members()
        self.save_summary_log_json()

    def __set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __set_dtype(self):
        K.set_floatx(self.dtype)
        print("Working with",self.dtype,"precision.")
        #tf.DType = ""

    def __set_data(self):
        if self.data is not None and self.input_data_file is not None:
            print("Input file is ignored when a data object is provided")
        elif self.data is None and self.input_data_file is None:
            raise Exception("Either a DataSample object or a dataset input file name should be passed while you passed none.\nPlease input one and retry.")
        elif self.data is None and self.input_data_file is not None:
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
                             verbose=True
                             )
        self.ndims = self.data.ndims
        self.__set_pars_info()

    def __set_pars_info(self):
        self.pars_pos_poi = self.data.pars_pos_poi.tolist()
        self.pars_pos_nuis = self.data.pars_pos_nuis.tolist()
        self.pars_labels = self.data.pars_labels
        
    def __set_ensemble_name(self):
        if self.ensemble_name is None:
            string = self.data.name
            try:
                match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', string).group()
            except:
                match = ""
            self.ensemble_name = "DNNLikEnsemble_"+string.replace(match,"")+self.ensemble_date_time
        else:
            self.ensemble_name = self.ensemble_name+"_"+self.ensemble_date_time

    def __check_npoints(self):
        available_points_tot = self.data.npoints
        available_points_train = (1-self.data.test_fraction)*available_points_tot
        available_points_test = self.data.test_fraction*available_points_tot
        max_required_points_train = np.max(np.array(self.__model_data_ensemble_inputs['npoints_list'])[:,0]+np.array(self.__model_data_ensemble_inputs['npoints_list'])[:,1])
        if max_required_points_train > available_points_train:
            print("Some models require more training points than available in data. Please reduce npoints_train+npoints_val or change test_fraction in the Data object.")
        max_required_points_test=np.max(np.array(self.__model_data_ensemble_inputs['npoints_list'])[:,2])
        if max_required_points_test > available_points_test:
            print("Some models requiring more test points than available in data. Please reduce npoints_test or change test_fraction in the Data object.")
        
    def __set_ensemble_folder(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        if self.ensemble_folder is not None:
            utils.check_rename_folder(self.ensemble_folder)
        else:
            utils.check_rename_folder(os.getcwd().replace('\\', '/')+"/"+self.ensemble_name)
        os.mkdir(self.ensemble_folder)
        print("Ensemble folder", self.ensemble_folder, "created.")

    def __set_ensemble_results_folder(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        self.ensemble_results_folder = self.ensemble_folder+"/ensemble"
        utils.check_create_folder(self.ensemble_results_folder)
        print("Ensemble results will be saved in the folder",self.ensemble_results_folder, ".")

    def __load_summary_log(self):
        summary_log_files = []
        for _, _, f in os.walk(self.DNNLik_ensemble_input_folder+"/ensemble"):
            for file in f:
                if "summary_log.json" in file:
                    summary_log_files.append(file)
        summary_log_file = os.path.join(
            self.DNNLik_ensemble_input_folder+"/ensemble", summary_log_files[-1])
        with open(summary_log_file) as json_file:
            summary_log = json.load(json_file)
        return summary_log

    def __check_model_data_ensemble_inputs(self):
        try:
            self.__model_data_ensemble_inputs["npoints_list"]
        except:
            print(
                "model_data_kwargs dictionary should contain at least a keyword 'npoints_list'. \
                It may also contain the keys 'scaleX_list', 'scaleY_list', and 'weighted_list'. In case these are not present by \
                default they are set to {'scaleX': False}, {'scaleY': False}, and {'weighted_list':[False]}.")
        for npoints in self.__model_data_ensemble_inputs["npoints_list"]:
            if npoints[1] <= 1:
                npoints[1] = round(npoints[1]*npoints[0])
        max_npoints_train = int(np.max(np.array(self.__model_data_ensemble_inputs["npoints_list"])[:, 0]))
        for npoints in self.__model_data_ensemble_inputs["npoints_list"]:
            if len(npoints)==2:
                npoints.append(max_npoints_train)
        self.__check_npoints()
        try:
            self.__model_train_ensemble_inputs["scaleX_list"]
        except:
            self.__model_train_ensemble_inputs["scaleX_list"] = [False]
        try:
            self.__model_train_ensemble_inputs["scaleY_list"]
        except:
            self.__model_train_ensemble_inputs["scaleY_list"] = [False]
        try:
            self.__model_train_ensemble_inputs["weighted_list"]
        except:
            self.__model_train_ensemble_inputs["weighted_list"] = [False]

    def __check_model_define_ensemble_inputs(self):
        try:
            self.__model_define_ensemble_inputs["hidden_layers_list"]
        except:
            print(
                "model_define_kwargs dictionary should contain at least the keyword 'hidden_layers_list'. \
                It may also contain keys 'act_func_out_layer_list', 'dropout_rate_list', 'batch_norm_list', and 'kernel_initializer_list'. \
                In case these are not present by default they are set to {'act_func_out_layer_list': ['linear']}, {'batch_norm_list': [False]},\
                {'dropout_rate_list': [0]}, {'kernel_initializers': ['glorot_uniform']}. In case of selu activation Dropout layers are replaced \
                with AlphaDropout and kernel_initializers is set to ['lecun_normal'].")
        try:
            self.__model_define_ensemble_inputs["act_func_out_layer_list"]
        except:
            self.__model_define_ensemble_inputs["act_func_out_layer_list"] = ["linear"] # Batch normalization layers are added between each pair of layers
        try:
            self.__model_define_ensemble_inputs["batch_norm_list"]
        except:
            # Batch normalization layers are added between each pair of layers
            self.__model_define_ensemble_inputs["batch_norm_list"] = [False]
        try:
            self.__model_define_ensemble_inputs["dropout_rate_list"]
        except:
            # The same dropout is performed between each pair of layers
            self.__model_define_ensemble_inputs["dropout_rate_list"] = [0]
        try:
            self.__model_define_ensemble_inputs["kernel_initializer_list"]
        except:
            self.__model_define_ensemble_inputs["kernel_initializer_list"] = [
                'glorot_uniform']  # The same dropout is performed between each pair of layers
        
    #def __check_model_optimizers_ensemble_inputs(self):
    #    #  example: model_optimizers_ensemble_inputs = {"optimizers_list": [{"Adam": {"learning_rate": 0.001,
    #    #                                                                             "beta_1": 0.9,
    #    #                                                                             "beta_2": 0.999,
    #    #                                                                             "amsgrad": False}},
    #    #                                                                   {"SGD": {"learning_rate": 0.01,
    #    #                                                                            "momentum": 0.0, 
    #    #                                                                            "nesterov": False}}]}
    #    try:
    #        self.__model_optimizers_ensemble_inputs["optimizers_list"]
    #    except:
    #        print(
    #            "model_optimizers_kwargs dictionary should contain at least one keyword 'optimizers_list'")
    #        raise
        
    def __check_model_compile_ensemble_inputs(self):
        try:
            self.__model_compile_ensemble_inputs["losses_list"]
        except:
            print(
                "model_compile_kwargs dictionary should contain at least the keyword 'losses_list'. \
                It may also contain the key 'metrics'. \
                In case this is not present by default it is set to {'metrics': (value of losses_list)}.")
        try:
            self.__model_compile_ensemble_inputs["metrics"]
        except:
            self.__model_compile_ensemble_inputs["metrics"] = self.__model_compile_ensemble_inputs["losses_list"]
    
    #def __check_model_callbacks_ensemble_kwargs(self):
    #    #  example: model_callbacks_ensemble_kwargs = {"callbacks_list": [{"PlotLossesKeras": True,
    #    #                                                             "EarlyStopping": {"monitor": model_compile_ensemble_inputs["metrics"],
    #    #                                                                               ...},
    #    #                                                             "ReduceLROnPlateau": {"monitor": model_compile_ensemble_inputs["metrics"],                                                                   "mode": "min",
    #    #                                                                                   ...},
    #    #                                                             "ModelCheckpoint": {"monitor": model_compile_ensemble_inputs["metrics"],
    #    #                                                                                 ...},
    #    #                                                             "TerminateOnNaN": True,
    #    #                                                             ...}]
    #    try:
    #        self.__model_callbacks_ensemble_inputs["callbacks_list"]
    #    except:
    #        self.__model_callbacks_ensemble_inputs["callbacks_list"] = [["TerminateOnNaN"]]

    def __check_model_train_ensemble_kwargs(self):
        # example: model_train_kwargs_list={"epochs_list": [200,1000],
        #                                 "batch_size_list": [512,1024,2048],
        try:
            self.__model_train_ensemble_inputs["epochs_list"]
            self.__model_train_ensemble_inputs["batch_size_list"]
        except:
            print(
                "model_train_kwargs dictionary should contain the keywords 'epochs_list' and 'batch_size_list'.")

    def __check_member_existence(self, DNNLik_input_folder):
        summary_log_files = []
        for _, _, f in os.walk(DNNLik_input_folder):
            for file in f:
                if "summary_log.json" in file:
                    summary_log_files.append(file)
        if len(summary_log_files) > 0:
            return True
        else:
            return False

    def __import_members(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        n_with_results = []
        n_without_results = []
        #gpus_id_list = [eval(s.split(":")[-1]) for s in np.array(self.active_gpus)[:, 0]]
        for n in range(self.n_members):
            DNNLik_input_folder = self.ensemble_folder+"/member_"+str(n)
            if self.__check_member_existence(DNNLik_input_folder):
                self.members[n] = DnnLik(DNNLik_input_folder=DNNLik_input_folder,
                                         data=self.data,
                                         resources_inputs=self.get_resources_inputs(),
                                         verbose=False
                                         )
                n_with_results.append(n)
            else:
                self.generate_members(n=n, verbose=False)
                n_without_results.append(n)
        end = timer()
        ShowPrints=verbose
        print("Results available for members",n_with_results,".")
        print("Results not available for members",n_without_results,".")
        print(self.n_members,"members imported in", str(end-start), "s.")

    def get_resources_inputs(self):
        return {"available_cpu": self.available_cpu, 
                "available_gpus": self.available_gpus, 
                "active_gpus": self.active_gpus,
                "gpu_mode": self.gpu_mode}

    def get_available_gpus(self, verbose=0):
        self.available_gpus = set_resources.get_available_gpus(verbose=verbose)

    def get_available_cpu(self, verbose=0):
        self.available_cpu = set_resources.get_available_cpu(verbose=verbose)

    def set_gpus(self, gpus_id_list, verbose=0):
        self.active_gpus = set_resources.set_gpus(gpus_id_list, verbose=verbose)
        if self.active_gpus != []:
            self.gpu_mode = True
        else:
            self.gpu_mode = False

    def generate_member(self, n, seed, model_data_inputs, model_define_inputs, model_optimizer_inputs, model_compile_inputs, model_callbacks_inputs, model_train_inputs, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        #gpus_id_list = [eval(s.split(":")[-1]) for s in np.array(self.active_gpus)[:, 0]]
        self.members[n] = DnnLik(DNNLik_input_folder=None,
                                 ensemble_name=self.ensemble_name,
                                 member_number=n,
                                 data=self.data,
                                 input_data_file=self.input_data_file,
                                 ensemble_folder=self.ensemble_folder,
                                 load_on_RAM=False,
                                 seed=seed,
                                 dtype=self.dtype,
                                 same_data=self.same_data,
                                 model_data_inputs=model_data_inputs,
                                 model_define_inputs=model_define_inputs,
                                 model_optimizer_inputs=model_optimizer_inputs,
                                 model_compile_inputs=model_compile_inputs,
                                 model_callbacks_inputs=model_callbacks_inputs,
                                 model_train_inputs=model_train_inputs,
                                 resources_inputs=self.get_resources_inputs(),
                                 verbose=False
                                 )
        end = timer()
        ShowPrints = verbose
        print("DNN Lik member",str(n),"created in",str(end-start),"s.")

    def generate_members(self,n="all",verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        all_kwargs = [[A, B, C, D, E, F] for A in self.__model_data_ensemble_inputs_list for B in self.__model_define_ensemble_inputs_list for C in self.__model_optimizers_ensemble_inputs_list for D in self.__model_compile_ensemble_inputs_list for E in self.__model_callbacks_ensemble_inputs_list for F in self.__model_train_ensemble_inputs_list]
        self.n_members = len(all_kwargs)
        if self.same_data:
            self.seeds = np.full(self.n_members,self.seed)
        else:
            np.random.seed(self.seed)
            self.seeds = np.random.choice(np.arange(100*self.n_members),self.n_members,replace=False)
        if n is "all":
            members_to_generate = range(self.n_members)
        else:
            members_to_generate = np.array([n]).flatten()
        for i in members_to_generate:
            model_data_inputs, model_define_inputs, model_optimizer_inputs, model_compile_inputs, model_callbacks_inputs, model_train_inputs = all_kwargs[i]
            self.generate_member(i, 
                                 self.seeds[i], 
                                 model_data_inputs, 
                                 model_define_inputs,
                                 model_optimizer_inputs,
                                 model_compile_inputs, 
                                 model_callbacks_inputs,
                                 model_train_inputs, 
                                 verbose=verbose_sub)
        #self.save_summary_log_json(verbose=False)
        end = timer()
        ShowPrints = verbose
        print(self.n_members,"members (DNNLikelihoods) generated in", str(end-start), "s.",show=verbose)
        print("Results for member 'n' will be saved in the folders",self.ensemble_name,"_member_n.",show=verbose)

    def generate_data_members(self, members_list="all",force=False,test=False,verbose=False):
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
                self.members[i].generate_train_data(verbose=verbose_2)
                if test:
                    self.members[i].generate_test_data(verbose=verbose_2)
            end = timer()
            ShowPrints = verbose
            print("Data for", len(members_list), "models generated in",str(end-start),"s.")

    def train_member_on_device(self, member, gpu="auto", verbose=2):
        global ShowPrints
        ShowPrints = verbose
        if gpu is "auto":
            gpu = 0
        elif gpu > len(self.available_gpus):
            print("GPU", gpu, "does not exist. Continuing on first active GPU.")
            gpu = 0
        time.sleep(1)
        self.members[member].model_build(gpu=gpu, verbose=verbose)
        self.members[member].model_train(verbose=verbose)
        self.members[member].model_store(verbose=verbose)

    def train_members_in_parallel(self, members_list, gpus_id_list="all", method="auto", verbose=2):
        """At the moment this only works with the concurrent method and even with this method it does not work optimally."""
        global ShowPrints
        ShowPrints = verbose
        if len(self.active_gpus) <= 1:
            if not self.gpu_mode:
                print("No active GPU. Proceeding with serial training on the CPU.")
            else:
                print("Only one active GPU. Proceeding with serial training on the GPU.")
            for member in members_list:
                self.train_member_on_device(member, gpu="auto", verbose=2)
        else:
            if gpus_id_list is "all":
                gpus_id_list = list(range(len(self.active_gpus)))
            if len(members_list) < len(gpus_id_list):
                gpus_id_list = gpus_id_list[:len(members_list)]
            members_chunks = utils.chunks(members_list, len(gpus_id_list))
            if method == "auto":
                try:
                    print("Training with joblib parallel method.")
                    for chunk in members_chunks:
                        if len(chunk) < len(gpus_id_list):
                            gpus_id_list = gpus_id_list[:len(chunk)]
                        gpu_member_map_dictionary = dict(zip(np.array(gpus_id_list), np.array(chunk)))
                        with parallel_backend('threading', n_jobs=len(gpus_id_list)):
                            Parallel()(delayed(self.train_member_on_device)(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose) for gpu in gpus_id_list)
                except:
                    print("Training with joblib parallel method failed.")
                    try:
                        print("Training with pool parallel method.")
                        for chunk in members_chunks:
                            if len(chunk) < len(gpus_id_list):
                                gpus_id_list = gpus_id_list[:len(chunk)]
                            gpu_member_map_dictionary = dict(zip(np.array(gpus_id_list), np.array(chunk)))
                            with Pool(len(gpus_id_list)) as pool:
                                pool.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
                    except:
                        print("Training with pool parallel method failed.")
                        try:
                            print("Training with concurrent parallel method.")
                            for chunk in members_chunks:
                                if len(chunk) < len(gpus_id_list):
                                    gpus_id_list = gpus_id_list[:len(chunk)]
                                gpu_member_map_dictionary = dict(zip(np.array(gpus_id_list), np.array(chunk)))
                                with ThreadPoolExecutor(len(gpus_id_list)) as executor:
                                    executor.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
                        except:
                            print("Training with concurrent parallel method failed.")
                            raise Exception("None of the parallel methods worked.")
            else:
                for chunk in members_chunks:
                    if len(chunk) < len(gpus_id_list):
                        gpus_id_list=gpus_id_list[:len(chunk)]
                    gpu_member_map_dictionary=dict(zip(np.array(gpus_id_list), np.array(chunk)))
                    if method == "joblib":
                        try:
                            with parallel_backend('threading', n_jobs=len(gpus_id_list)):
                                Parallel()(delayed(self.train_member_on_device)(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose) for gpu in gpus_id_list)
                        except:
                            print("Training with joblib parallel method failed. Trying with auto method.")
                            self.train_members_in_parallel(members_list, gpus_id_list = gpus_id_list, method="auto",verbose = verbose)
                    elif method == "pool":
                        try:
                            with Pool(len(gpus_id_list)) as pool:
                                pool.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
                        except:
                            print("Training with pool parallel method failed. Trying with auto method.")
                            self.train_members_in_parallel(members_list, gpus_id_list = gpus_id_list, method="auto",verbose = verbose)
                    elif method == "concurrent":
                        try:
                            with ThreadPoolExecutor(len(gpus_id_list)) as executor:
                                executor.map(lambda gpu: self.train_member_on_device(
                                    member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
                        except:
                            print("Training with concurrent parallel method failed. Trying with auto method.")
                            self.train_members_in_parallel(members_list, gpus_id_list=gpus_id_list, method="auto", verbose=verbose)
                        #try:
                        #    with ThreadPoolExecutor(len(gpus_id_list)) as executor:
                        #        executor.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
                        #except:
                        #    self.train_members_in_parallel(members_list, gpus_id_list = gpus_id_list, method="auto",verbose = verbose)
            failed_members_list = []
            for member in members_list:
                try:
                    self.members[member].model
                except:
                    print("Training of member", member,
                          "failed. Trying again.")
                    failed_members_list.append(member)
            #if len(failed_members_list) != 0:
            #    self.train_members_in_parallel(failed_members_list, gpus_id_list=gpus_id_list, method=method, verbose=verbose)

    def train_members_in_parallel_concurrent(self, members_list, gpus_id_list="all", verbose=2):
        """Function that trains and stores members in parallel."""
        global ShowPrints
        ShowPrints = verbose
        if len(self.active_gpus) <= 1:
            if not self.gpu_mode:
                print("No active GPU. Proceeding with serial training on the CPU.")
            else:
                print("Only one active GPU. Proceeding with serial training on the GPU.")
            for member in members_list:
                self.train_member_on_device(member, gpu="auto", verbose=2)
        else:
            if gpus_id_list is "all":
                gpus_id_list = list(range(len(self.active_gpus)))
            if len(members_list) < len(gpus_id_list):
                gpus_id_list = gpus_id_list[:len(members_list)]
            members_chunks = utils.chunks(members_list, len(gpus_id_list))
            for chunk in members_chunks:
                if len(chunk) < len(gpus_id_list):
                    gpus_id_list = gpus_id_list[:len(chunk)]
                gpu_member_map_dictionary = dict(zip(np.array(gpus_id_list), np.array(chunk)))
                with ThreadPoolExecutor(len(gpus_id_list)) as executor:
                    executor.map(lambda gpu: self.train_member_on_device(member=gpu_member_map_dictionary[gpu], gpu=gpu, verbose=verbose), gpus_id_list)
            failed_members_list = []
            for member in members_list:
                try:
                    self.members[member].model
                except:
                    print("Training of member", member,"failed. Trying again.")
                    failed_members_list.append(member)
            if len(failed_members_list) != 0:
                self.train_members_in_parallel_concurrent(failed_members_list, gpus_id_list=gpus_id_list, verbose=verbose)


    def get_files_list_in_ensemble(self, string=""):
        folders = [self.members[i].member_results_folder for i in range(self.n_members)]
        mylist = []
        i = 0
        for thisdir in folders:
            # r=root, d=directories, f = files
            for r, _, f in os.walk(thisdir):
                for file in f:
                    if string in file:
                        current_file = os.path.join(r, file)
                        mylist.append(current_file.replace('\\', '/'))
                        i = i + 1
        print(str(i)+' files')
        return mylist

    def save_summary_log_json(self, verbose=True):
        """ Save summary log (history plus model specifications) to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print("Saving summary_log to json file",
              self.summary_log_json_filename)
        start = timer()
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        history = {**{'Date time': str(now)}, **utils.dic_minus_keys(self.__dict__, ['data',
                                                                                       'members', 'stacks',
                                                                                       '_DNNLik_ensemble__model_data_ensemble_inputs_list',
                                                                                       '_DNNLik_ensemble__model_define_ensemble_inputs_list',
                                                                                       '_DNNLik_ensemble__model_optimizers_ensemble_inputs_list',
                                                                                       '_DNNLik_ensemble__model_compile_ensemble_inputs_list',
                                                                                       '_DNNLik_ensemble__model_callbacks_ensemble_inputs_list',
                                                                                       '_DNNLik_ensemble__model_train_ensemble_inputs_list'])}
        new_hist = utils.convert_types_dict(history)
        utils.check_rename_file(self.summary_log_json_filename, verbose=verbose_sub)
        with codecs.open(self.summary_log_json_filename, 'w', encoding='utf-8') as f:
            json.dump(new_hist, f, separators=(
                ',', ':'), indent=4)
        end = timer()
        print(self.summary_log_json_filename, "created and saved.", str(end-start), "s.")

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
