__all__ = ["DNNLik"]

import json
import codecs
import h5py
from timeit import default_timer as timer
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
    def __init__(self,
                 ensemble_name=None,
                 member_number=0,
                 data_sample=None,
                 ensemble_results_folder=None,
                 seed=1,
                 same_data=True,
                 model_data_member_kwargs=None,
                 model_define_member_kwargs=None,
                 model_optimizer_member_kwargs=None,
                 model_compile_member_kwargs=None,
                 model_callbacks_member_kwargs=None,
                 model_train_member_kwargs=None
                 ):
        ############ Initialize input parameters
        #### Set main inputs and DataSample
        self.data_sample = data_sample
        self.ndim = self.data_sample.data_X.shape[1]
        self.seed = seed
        self.same_data = same_data
        self.__model_data_member_kwargs = model_data_member_kwargs
        self.__model_define_member_kwargs = model_define_member_kwargs
        self.__model_optimizer_member_kwargs = model_optimizer_member_kwargs
        self.__model_compile_member_kwargs = model_compile_member_kwargs
        self.__model_callbacks_member_kwargs = model_callbacks_member_kwargs
        self.__model_train_member_kwargs = model_train_member_kwargs
        self.npoints_train, self.npoints_val, self.npoints_test = self.__model_data_member_kwargs["npoints"]
        
        ### Set name, folders and files names
        self.ensemble_name = ensemble_name
        self.member_number = member_number
        self.member_name = self.ensemble_name +"_member_"+str(self.member_number)
        self.ensemble_results_folder = ensemble_results_folder
        self.member_results_folder = self.ensemble_results_folder + "/member_"+str(self.member_number)
        self.__check_ensemble_results_folder()
        self.__set_member_results_folder()
        self.history_json_filename = self.member_results_folder+"/"+self.member_name+"_history.json"
        self.idx_filename = self.member_results_folder+"/"+self.member_name+"_idx.h5"
        self.model_json_filename = self.member_results_folder+"/"+self.member_name+"_model.json"
        self.model_h5_filename = self.member_results_folder+"/"+self.member_name+"_model.h5"
        self.model_onnx_filename = self.member_results_folder+"/"+self.member_name+"_model.onnx"
        self.scalerX_jlib_filename = self.member_results_folder+"/"+self.member_name+"_scalerX.jlib"
        self.scalerY_jlib_filename = self.member_results_folder+"/"+self.member_name+"_scalerY.jlib"
        self.model_graph_pdf_filename = self.member_results_folder+"/"+self.member_name+"_model_graph.pdf"

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

    def __check_ensemble_results_folder(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        if not os.path.exists(self.ensemble_results_folder):
            os.mkdir(self.ensemble_results_folder)
            print("Ensemble folder", self.ensemble_results_folder,"did not exist and has been created.")

    def __set_member_results_folder(self):
        self.member_results_folder = utility.check_rename_folder(self.member_results_folder).replace('\\', '/')
        os.mkdir(self.member_results_folder)

    def compute_sample_weights(self, bins=100, power=1):
        # Generate weights
        self.W_train = self.data_sample.compute_sample_weights(self.Y_train, bins=bins, power=power)
        self.W_val = self.data_sample.compute_sample_weights(self.Y_val, bins=bins, power=power)
        self.W_test = self.data_sample.compute_sample_weights(self.Y_test, bins=bins, power=power)
        #self.W_train = np.full(len(self.Y_train), 1)
        #self.W_val = np.full(len(self.Y_val), 1)
        #self.W_test = np.full(len(self.Y_test), 1)

    def define_scalers(self, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.scalerX, self.scalerY = self.data_sample.define_scalers(self.X_train, self.Y_train, self.scalerX_bool, self.scalerY_bool, verbose=verbose)

    def generate_data(self, bins=100, power=1, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        # Generate data
        if self.same_data:
            self.data_sample.update_data(
                self.npoints_train, self.npoints_val, self.npoints_test, self.seed, verbose=verbose)
        else:
            self.data_sample.generate_data(
                self.npoints_train, self.npoints_val, self.npoints_test, self.seed, verbose=verbose)
        self.X_train = self.data_sample.data_dictionary["X_train"][:self.npoints_train]
        self.Y_train = self.data_sample.data_dictionary["Y_train"][:self.npoints_train]
        self.X_val = self.data_sample.data_dictionary["X_val"][:self.npoints_val]
        self.Y_val = self.data_sample.data_dictionary["Y_val"][:self.npoints_val]
        self.X_test = self.data_sample.data_dictionary["X_test"][:self.npoints_test]
        self.Y_test = self.data_sample.data_dictionary["Y_test"][:self.npoints_test]
        self.idx_train = self.data_sample.data_dictionary["idx_train"][:self.npoints_train]
        self.idx_test = self.data_sample.data_dictionary["idx_test"][:self.npoints_train]
        self.idx_val = self.data_sample.data_dictionary["idx_val"][:self.npoints_train]
        # Define scalers
        self.define_scalers()

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
        print("Model defined in", end-start, "s.")
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

    def metric_name_abbreviate(self, name):
        name_dict = {"accuracy": "acc", "mean_error": "me", "mean_percentage_error": "mpe", "mean_squared_error": "mse",
                     "mean_absolute_error": "mae", "mean_absolute_percentage_error": "mape", "mean_squared_logarithmic_error": "msle"}
        for key in name_dict:
            name = name.replace(key, name_dict[key])
        return name

    def metric_name_unabbreviate(self, name):
        name_dict = {"acc": "accuracy", "me": "mean_error", "mpe": "mean_percentage_error", "mse": "mean_squared_error",
                     "mae": "mean_absolute_error", "mape": "mean_absolute_percentage_error", "msle": "mean_squared_logarithmic_error"}
        for key in name_dict:
            name = name.replace(key, name_dict[key])
        return name

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
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        end = timer()
        print("Model compiled in",end-start,"s.")

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
            else:
                key1 = "callbacks."+key1
            for key2, value2 in value1.items():
                if key2 == "monitor" and type(value2) is str:
                    if "val_" in value2:
                        value2 = value2.split("val_")[1]
                    if value2 == 'loss':
                        value2 = 'val_loss'
                    else:
                        value2 = "val_" + self.metric_name_unabbreviate(value2)
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
        print("Scaling training data.")
        X_train = self.scalerX.transform(self.X_train)
        X_val = self.scalerX.transform(self.X_val)
        Y_train = self.scalerY.transform(self.Y_train.reshape(-1, 1)).reshape(len(self.Y_train))
        Y_val = self.scalerY.transform(self.Y_val.reshape(-1, 1)).reshape(len(self.Y_val))
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
        print("Start training of member",self.member_number, ".")
        if self.weighted:
            # Compute weights
            self.compute_sample_weights()
            # Train
            history = self.model.fit(X_train, Y_train, sample_weight=self.W_train, epochs=self.required_epochs, batch_size=self.batch_size, verbose=verbose_tf,
                    validation_data=(X_val, Y_val), callbacks=self.callbacks)
        else:
            history = self.model.fit(X_train, Y_train, epochs=self.required_epochs, batch_size=self.batch_size, verbose=verbose_tf,
                    validation_data=(X_val, Y_val), callbacks=self.callbacks)
        self.history = history.history
        self.exact_epochs = len(self.history['loss'])
        end = timer()
        self.training_time = end - start
        if "PlotLossesKeras" in str(self.callbacks_strings):
            plt.close()
        ShowPrints = verbose
        print("Member",self.member_number,"successfully trained for",self.required_epochs, "epochs in", self.training_time,"s.")
        #print("Generating summary log")
        self.generate_summary_log()

    def save_data_indices(self, verbose=True):
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
        data["idx_test"] = self.idx_test
        h5_out.close()
        end = timer()
        print(self.idx_filename, "created and saved in", end-start, "s.")

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

    def save_summary_log_json(self,verbose=True):
        """ Save summary log (history plus model specifications) to json
        """
        global ShowPrints
        ShowPrints = verbose
        print("Saving history and summary_log to json file", self.history_json_filename)
        start = timer()
        history = self.summary_log
        new_hist = {}
        for key in list(history.keys()):
            if type(history[key]) == np.ndarray:
                new_hist[key] == history[key].tolist()
            elif type(history[key]) == list:
                if type(history[key][0]) == np.float64:
                    new_hist[key] = list(map(float, history[key]))
                elif type(history[key][0]) == np.float32:
                    new_hist[key] = list(map(float, history[key]))
                else:
                    new_hist[key] = history[key]
            else:
                new_hist[key] = history[key]
        with codecs.open(self.history_json_filename, 'w', encoding='utf-8') as f:
            json.dump(new_hist, f, separators=(
                ',', ':'), sort_keys=True, indent=4)
        end = timer()
        print(self.history_json_filename, "created and saved.", end-start, "s.")

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
        os.remove(png_file)
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
        self.save_data_indices(verbose=verbose)
        self.save_model_json(verbose=verbose)
        self.save_model_h5(verbose=verbose)
        self.save_model_onnx(verbose=verbose)
        self.save_summary_log_json(verbose=verbose)
        self.save_scalers_jlib(verbose=verbose)
        self.save_model_graph_pdf(verbose=verbose)

    def generate_summary_log(self,verbose=False):
        #self.summary_log = {**metrics, **metrics_scaled}
        self.summary_log = self.history
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.summary_log['Date time'] = str(now)
        self.summary_log['Ensemble name'] = self.ensemble_name
        self.summary_log['Member number'] = self.member_number
        self.summary_log['Member name'] = self.member_name
        try:
            model_plot_losses_keras_filename = self.model_plot_losses_keras_filename
        except:
            model_plot_losses_keras_filename = ""
        try:
            model_checkpoints_filename = self.model_checkpoints_filename
        except:
            model_checkpoints_filename = ""
        self.summary_log['Files and folders'] = {
                                                 'Data_sample file name': self.data_sample.data_sample_input_filename,
                                                 'Ensemble results folder': self.ensemble_results_folder,
                                                 'Member results folder': self.member_results_folder,
                                                 'History file name': self.history_json_filename,
                                                 'Indices file name': self.idx_filename,
                                                 'Model json file name': self.model_json_filename,
                                                 'Model h5 file name': self.model_h5_filename,
                                                 'Model onnx file name': self.model_onnx_filename,
                                                 'ScalerX file name': self.scalerX_jlib_filename,
                                                 'ScalerY file name': self.scalerY_jlib_filename,
                                                 'Model graph file name': self.model_graph_pdf_filename,
                                                 'Model checkpoint filename': model_checkpoints_filename,
                                                 'Plot losses keras filename': model_plot_losses_keras_filename
                                                }
        self.summary_log['Ndim'] = self.ndim
        self.summary_log['N points train'] = self.npoints_train
        self.summary_log['N points val'] = self.npoints_val
        self.summary_log['N points test'] = self.npoints_test
        self.summary_log['Weighted'] = self.weighted
        self.summary_log['Scaled X'] = self.scalerX_bool
        self.summary_log['Scaled Y'] = self.scalerY_bool
        self.summary_log['Loss'] = self.loss_string
        self.summary_log['Metrics'] = self.metrics_string
        self.summary_log['Hidden layers'] = self.hid_layers
        self.summary_log['Params'] = self.model_params
        self.summary_log['Trainable params'] = self.model_trainable_params
        self.summary_log['Non-trainable params'] = self.model_non_trainable_params
        self.summary_log['Dropout'] = self.dropout_rate
        self.summary_log['Callbacks'] = self.__model_callbacks_member_kwargs["callbacks"]
        self.summary_log['Activation function out layer'] = self.act_func_out_layer
        self.summary_log['Batch norm'] = self.batch_norm
        self.summary_log['Optimizer'] = self.__model_optimizer_member_kwargs["optimizer"]
        self.summary_log['Batch size'] = self.batch_size
        self.summary_log['Required epochs'] = self.required_epochs
        self.summary_log['Exact epochs'] = self.exact_epochs
        #self.summary_log['GPU(s)'] = GPU_names[:N_GPUS]
        self.summary_log['Training time'] = self.training_time
        for key in list(self.summary_log.keys()):
            self.summary_log[self.metric_name_abbreviate(key)] = self.summary_log.pop(key)

    def R2_metric(self, y_true, y_pred):
        MSE_model =  K.sum(K.square( y_true-y_pred )) 
        MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return (1 - MSE_model/(MSE_baseline + K.epsilon()))

    def Rt_metric(self, y_true, y_pred):
        MAPE_model =  K.sum(K.abs( 1-y_pred/(y_true + K.epsilon()))) 
        MAPE_baseline = K.sum(K.abs( 1-K.mean(y_true)/(y_true+ K.epsilon()) ) ) 
        return ( 1 - MAPE_model/(MAPE_baseline + K.epsilon()))