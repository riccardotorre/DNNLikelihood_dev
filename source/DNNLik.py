__all__ = ["DNNLik"]

import multiprocessing
import builtins
ShowPrints = True

def print(*args, **kwargs):
    global ShowPrints
    if ShowPrints:
        return builtins.print(*args, **kwargs)
    else:
        return None

import numpy as np
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import (EarlyStopping, History, ReduceLROnPlateau,
                             TerminateOnNaN)
from keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                          Input)
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.utils import plot_model
import os
import pickle
from sklearn.model_selection import train_test_split

from .data_sample import Data_sample
from . import utility

class DNNLik():
    def __init__(self,
                 name=None,
                 folder=None,
                 n_dim=None,
                 hid_layers=None,
                 dropout_rate=None,
                 act_func_out_layer=None,
                 batch_norm=None,
                 loss=None,
                 optimizer=None, 
                 metrics=None,
                 multi_gpu=False,
                 X_train=None, 
                 Y_train=None, 
                 X_val=None, 
                 Y_val=None, 
                 scalerX=None, 
                 scalerY=None, 
                 epochs=None, 
                 batch_size=None, 
                 sample_weights=None, 
                 early_stopping=False, 
                 reduce_LR_patience=None,
                 history=None,
                 summary_text=None
                 ):
        ############ Initialize input parameters
        self.name = name
        self.folder = folder.rstrip('/')
        self.n_dim = n_dim
        self.hid_layers = hid_layers
        self.dropout_rate = dropout_rate
        self.act_func_out_layer = act_func_out_layer
        self.batch_norm = batch_norm
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.multi_gpu = multi_gpu
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.early_stopping = early_stopping
        self.reduce_LR_patience = reduce_LR_patience
        self.history = history
        self.summary_text = summary_text

        ############ Initialize additional attributes
        self.availableGPUs = K.tensorflow_backend._get_available_gpus()
        self.availableCPUCoresNumber = multiprocessing.cpu_count()
        
        ############ Create model
        self.model_define()
        
    def __set_param__(self, par_name, par_val):
        if par_val is None:
            par_val = eval("self."+par_name)
            print("No parameter"+par_val+"specified. Its value has been set to",
                  par_val, ".")
        else:
            setattr(self, par_name, par_val)

    def model_define(self,n_dim=None,hid_layers=None,dropout_rate=None,act_func_out_layer=None,batch_norm=None,verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("n_dim", n_dim)
        self.__set_param__("hid_layers", hid_layers)
        self.__set_param__("dropout_rate", dropout_rate)
        self.__set_param__("act_func_out_layer", act_func_out_layer)
        self.__set_param__("batch_norm", batch_norm)
        inputLayer = Input(shape=(n_dim,))
        if batch_norm:
            x = BatchNormalization()(inputLayer)
        if hid_layers[0][1] == 'selu':
            x = Dense(hid_layers[0][0], activation=hid_layers[0][1], kernel_initializer='lecun_normal')(inputLayer)
        else:
            x = Dense(hid_layers[0][0], activation=hid_layers[0][1], kernel_initializer='glorot_uniform')(inputLayer)
        if batch_norm:
            x = BatchNormalization()(x)
        if dropout_rate != 0:
            if hid_layers[0][1] == 'selu':
                x = AlphaDropout(dropout_rate)(x)
            else:
                x = Dropout(dropout_rate)(x)
        if len(hid_layers)>1:
            for i in hid_layers[1:]:
                if i[1] == 'selu':
                    x = Dense(i[0], activation=i[1], kernel_initializer='lecun_normal')(x)
                else:
                    x = Dense(i[0], activation=i[1], kernel_initializer='glorot_uniform')(x)
                if batch_norm:
                    x = BatchNormalization()(x)
                if dropout_rate != 0:
                    if i[1] == 'selu':
                        x = AlphaDropout(dropout_rate)(x)
                    else:
                        x = Dropout(dropout_rate)(x)
        outputLayer = Dense(1, activation=act_func_out_layer)(x)
        model = Model(inputs=inputLayer, outputs=outputLayer)
        print(model.summary())
        self.model = model

    def compute_model_params(self,model=None): # Compute number of params in a model (the actual number of floats)
        self.__set_param__("model", model)
        self.model_params = int(model.count_params())
    def compute_model_trainable_params(self,model=None): # Compute number of params in a model (the actual number of floats)
        self.__set_param__("model", model)
        self.model_trainable_params = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    def compute_model_non_trainable_params(self, model=None):
        self.__set_param__("model", model)
        self.model_non_trainable_params = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    def model_compile(self,model=None,loss=None,optimizer=None,metrics=None,multi_gpu=None,verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("model", model)
        self.__set_param__("loss", loss)
        self.__set_param__("optimizer", optimizer)
        self.__set_param__("metrics", metrics)
        self.__set_param__("multi_gpu", multi_gpu)
        if len(self.availableGPUs) > 1:
            print(str(len(self.availableGPUs))+" GPUs available")
            if multi_gpu:
                print("Compiling model on available GPUs")
                # Replicates `model` on availableGPUs.
                parallel_model = multi_gpu_model(model, gpus=len(availableGPUs))
                parallel_model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
                model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            else:
                print("multi_gpu=False. Compiling model on single GPU")
                model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
        elif len(availableGPUs)==1:
            print("One GPU available. Compiling model on single GPU")
            model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
        else:
            print("No GPU available. Compiling model on CPU")
            print(str(self.availableCPUCoresNumber)+" CPU cores available")
        #config = K.tf.ConfigProto(device_count={"CPU": availableCPUCoresNumber})
        #K.set_session(K.tf.Session(config=config))
        ##K.set_session(K.tf.Session(config=K.tf.ConfigProto(device_count={"CPU": availableCPUCoresNumber}, intra_op_parallelism_threads=availableCPUCoresNumber, inter_op_parallelism_threads=availableCPUCoresNumber)))
            model.compile(loss=loss, optimizer=optimizer,metrics=metrics)
        try:
            self.training_model = parallel_model
        except:
            self.training_model = model
        self.model = model

    def model_fit(self,model=None,X_train=None,Y_train=None,X_val=None,Y_val=None,scalerX=None,scalerY=None,epochs=None,batch_size=None,sample_weights=None,early_stopping=False,reduce_LR_patience=None,verbose=False):
        global ShowPrints
        if verbose > 0:
            ShowPrints = True
        else:
            ShowPrints = False
        self.__set_param__("model", model)
        self.__set_param__("X_train", X_train)
        self.__set_param__("Y_train", Y_train)
        self.__set_param__("X_val", X_val)
        self.__set_param__("Y_val", Y_val)
        self.__set_param__("scalerX", scalerX)
        self.__set_param__("scalerY", scalerY)
        self.__set_param__("epochs", epochs)
        self.__set_param__("batch_size", batch_size)
        self.__set_param__("sample_weights", sample_weights)
        self.__set_param__("early_stopping", epochearly_stoppings)
        self.__set_param__("reduce_LR_patience", reduce_LR_patience)
        X_train = scalerX.transform(X_train)
        X_val = scalerX.transform(X_val)
        Y_train = scalerY.transform(Y_train.reshape(-1, 1)).reshape(len(Y_train))
        Y_val = scalerY.transform(Y_val.reshape(-1, 1)).reshape(len(Y_val))
        start = timer()
        #plot_losses = PlotLossesKeras()
        if early_stopping:
            callbacks = [
                #plot_losses,
                EarlyStopping(monitor='val_loss', patience=2*reduce_LR_patience, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=reduce_LR_patience, verbose=verbose),
                TerminateOnNaN()]
        else:
            callbacks = [
                #plot_losses,
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=reduce_LR_patience, verbose=verbose),
                TerminateOnNaN()]
        history = model.fit(X_train, Y_train, sample_weight=sample_weights, epochs=epochs, batch_size=batch_size, verbose = verbose,validation_data=(X_val, Y_val),callbacks = callbacks)
        end = timer()
        training_time = end - start
        self.history = history
        self.training_time = training_time

    def model_save_fig(self,name=None,folder=None,history=None,summary_text=None,metric='loss',yscale='linear',verbose=True):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("history", history)
        self.__set_param__("summary_text", summary_text)
        folder = folder.rstrip('/')
        filename = name.replace(": ", "_")
        val_metric = 'val_' + metric
        jtplot.reset()
        try:
            plt.style.use('matplotlib.mplstyle')
        except FileNotFoundError:
            print("Matplotlib style file 'matplotlib.mplstyle' not found.")
        fig = plt.figure(1,figsize=(9.5,7))
        ax = fig.add_subplot(111)
        if type(history) is dict:
            ax.plot(history[metric])
            ax.plot(history[val_metric])
        else:
            ax.plot(history.history[metric])
            ax.plot(history.history[val_metric])
        plt.yscale(yscale)
        plt.grid(linestyle="--", dashes=(5,5))
        plt.title(r"%s"%title,fontsize = 14)
        plt.xlabel(r"epoch",fontsize = 24)
        ylable = (metric.replace("_","-"))
        plt.ylabel(r"%s"%ylable,fontsize = 24)
        plt.legend([r"training", r"validation"], fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.text(0.96,0.1,r"%s"%summary_text, fontsize=11, bbox=dict(facecolor="gray",alpha=0.25, edgecolor='black', boxstyle='round,pad=0.5'),transform = ax.transAxes, ha='right',ma='left')
        #plt.show()
        fig.canvas.draw()
        plt.savefig(r"%s" % (folder + "/" + filename + "_figure.pdf"))
        plt.close()
        print(r"%s"%(folder + "/" + filename + "_figure.pdf" + " created and saved."))

    def model_save_history(self,name=None,folder=None,history=None,verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("history", history)
        if type(history) is not dict:
            history = history.history
        folder = folder.rstrip('/')
        filename = name.replace(": ", "_") + "_history.json"
        new_hist = {}
        for key in list(history.keys()):
            if type(history[key]) == np.ndarray:
                new_hist[key] == history[key].tolist()
            elif type(history[key]) == list:
                if  type(history[key][0]) == np.float64:
                    new_hist[key] = list(map(float, history[key]))
                elif  type(history[key][0]) == np.float32:
                    new_hist[key] = list(map(float, history[key]))
                else:
                    new_hist[key] = history[key]
            else:
                new_hist[key] = history[key]
        with codecs.open(r"%s" % (folder + "/" + filename), 'w', encoding='utf-8') as f:
            json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)
        print(r"%s"%(folder + "/" + filename + " created and saved."))

    def model_save_data_indices(self,name=None, folder=None, idx_train=None, idx_val=None, idx_test=None, verbose=False):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        timestamp = str(now)
        if idx_train is not False:
            self.__set_param__("idx_train", idx_train)
            filename_train = name.replace(": ", "_")+"_samples_indices_train.pickle"
            idx_train_len = len(idx_train)
            pickle_out = open(folder + "/" + filename_train, 'wb')
            pickle.dump(filename_train, pickle_out, protocol=4)
            pickle.dump(timestamp, pickle_out, protocol=4)
            pickle.dump(idx_train_len, pickle_out, protocol=4)
            pickle.dump(idx_train, pickle_out, protocol=4)
            print(folder + "/" + filename_train + " created and saved."))
        if idx_val is not False:
            self.__set_param__("idx_val", idx_val)
            filename_val = name.replace(": ", "_")+"_samples_indices_val.pickle"
            idx_val_len = len(idx_val)
            pickle_out = open(folder + "/" + filename_val, 'wb')
            pickle.dump(filename_val, pickle_out, protocol=4)
            pickle.dump(timestamp, pickle_out, protocol=4)
            pickle.dump(idx_val_len, pickle_out, protocol=4)
            pickle.dump(idx_val, pickle_out, protocol=4)
            print(folder + "/" + filename_val + " created and saved."))
        if idx_test is not False:
            self.__set_param__("idx_test", idx_test)
            filename_test = name.replace(": ", "_")+"_samples_indices_test.pickle"
            idx_test_len = len(idx_test)
            pickle_out = open(folder + "/" + filename_test, 'wb')
            pickle.dump(filename_test, pickle_out, protocol=4)
            pickle.dump(timestamp, pickle_out, protocol=4)
            pickle.dump(idx_test_len, pickle_out, protocol=4)
            pickle.dump(idx_test, pickle_out, protocol=4)
            print(folder + "/" + filename_test + " created and saved."))

    def model_save_json(self, name=None, folder=None, model=None,verbose=False):
        global ShowPrints
        ShowPrints=verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("model", model)
        model_json=model.to_json()
        folder=folder.rstrip('/')
        filename=name.replace(": ", "_") + "_model.json"
        with open(r"%s"%(folder + "/" + filename), "w") as json_file:
            json_file.write(model_json)
        print(folder + "/" + filename + " created and saved."))

    def model_save_HDF5(self, name=None, folder=None, model=None,verbose=False):
        global ShowPrints
        ShowPrints=verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("model", model)
        folder=folder.rstrip('/')
        filename=name.replace(": ", "_") + "_model.h5"
        model.save(r"%s"%(folder + "/" + filename))
        print(folder + "/" + filename + " created and saved."))

    def model_save_scaler(self, name=None, folder=None, scalerX=None, scalerY=None,verbose=False):
        global ShowPrints
        ShowPrints=verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        if scalerX is not False:
            self.__set_param__("scalerX", scalerX)
            folder=folder.rstrip('/')
            filename_X=name.replace(": ", "_") + "_scalerX.jlib"
            joblib.dump(scalerX, r"%s"%(folder + "/" + filename))
            print(folder + "/" + filename_X + " created and saved."))
        if scalerY is not False:
            self.__set_param__("scalerY", scalerY)
            folder=folder.rstrip('/')
            filename_Y=name.replace(": ", "_") + "_scalerY.jlib"
            joblib.dump(scalerY, r"%s"%(folder + "/" + filename))
            print(folder + "/" + filename_Y + " created and saved."))

    def model_save_model_graph(self, name=None, folder=None, model=None,verbose=False):
        global ShowPrints
        ShowPrints=verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("model", model)
        folder=folder.rstrip('/')
        filename=name.replace(": ", "_") + "_model_graph.pdf"
        plot_model(model, show_shapes=True, show_layer_names=True,to_file=r"%s" % (folder + "/" + filename))
        print(folder + "/" + filename + " created and saved."))

    def model_store(self,name=None, folder=None, idx_train=None, idx_val=None, idx_test=None, model=None, scalerX=None, scalerY=None, history=None, summary_log=None, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        self.__set_param__("name", name)
        self.__set_param__("folder", folder)
        self.__set_param__("idx_train", idx_train)
        self.__set_param__("idx_val", idx_val)
        self.__set_param__("idx_test", idx_test)
        self.__set_param__("model", model)
        self.__set_param__("scalerX", scalerX)
        self.__set_param__("scalerY", scalerY)
        self.__set_param__("history", history)
        self.__set_param__("summary_log", summary_log)
        history={**summary_log, **history}
        self.history=history
        #Save samples indices
        self.model_save_data_indices(verbose=True)
        #Save model as JSON
        self.model_save_json(verbose=True)
        #Save mosel as h5
        self.model_save_HDF5(verbose=True)
        #Save history as json
        self.model_save_history(verbose=True)
        #Save scalers
        self.model_save_scaler(verbose=True)
        #Save model graph
        self.model_save_model_graph(verbose=True)

    def load_data_indices(filename):
        file = file.replace('model.h5', 'samples_indices.pickle')
        pickle_in = open(file, 'rb')
        idx_train = pickle.load(pickle_in)
        idx_val = pickle.load(pickle_in)
        idx_test = pickle.load(pickle_in)
        pickle_in.close()
        return [idx_train, idx_val, idx_test]

    def model_predict(model,scalerX,scalerY,X,batch_size=1,steps=None,verbose=0):
        start = timer()
        X = scalerX.transform(X)
        pred = scalerY.inverse_transform(model.predict(X, batch_size=batch_size, steps=steps, verbose = verbose)).reshape(len(X))
        end = timer()
        prediction_time = end - start
        return [pred, prediction_time]

    def compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, LOSS, NEVENTS_TRAIN, BATCH_SIZE, FREQUENTISTS_RESULTS):
        print('Computing predictions')
        start_global = timer()
        start = timer()
        [idx_train, idx_val, idx_test] = [np.random.choice(np.arange(len(X)), min(
            int(NEVENTS_TRAIN), len(X)), replace=False) for X in [X_train, X_val, X_test]]
        pred_train, prediction_time_train = model_predict(
            model, scalerX, scalerY, X_train[idx_train], batch_size=BATCH_SIZE)
        pred_val, prediction_time_val = model_predict(
            model, scalerX, scalerY, X_val[idx_val], batch_size=BATCH_SIZE)
        pred_test, prediction_time_test = model_predict(
            model, scalerX, scalerY, X_test[idx_test], batch_size=BATCH_SIZE)
        prediction_time = prediction_time_test
        [Y_pred_train, Y_pred_val, Y_pred_test] = [pred_train, pred_val, pred_test]
        [Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [
            np.exp(Y_pred_train), np.exp(Y_pred_val), np.exp(Y_pred_test)]
        [Y_train_exp, Y_val_exp, Y_test_exp] = [
            np.exp(Y_train[idx_train]), np.exp(Y_val[idx_val]), np.exp(Y_test[idx_test])]
        [min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test] = [keras.losses.deserialize(LOSS)(Y_train[idx_train], Y_pred_train).eval(session=tf.Session(
        )), keras.losses.deserialize(LOSS)(Y_val[idx_val], Y_pred_val).eval(session=tf.Session()), keras.losses.deserialize(LOSS)(Y_test[idx_test], Y_pred_test).eval(session=tf.Session())]
        [mape_on_exp_train, mape_on_exp_val, mape_on_exp_test] = [100/len(Y_train_exp)*np.sum(np.abs(Y_pred_train_exp-Y_train_exp)/Y_train_exp), 100/len(
            Y_val_exp)*np.sum(np.abs(Y_pred_val_exp-Y_val_exp)/Y_val_exp), 100/len(Y_test_exp)*np.sum(np.abs(Y_pred_test_exp-Y_test_exp)/Y_test_exp)]
        end = timer()
        print('Prediction on', NEVENTS_TRAIN,
              'points done form training, validation, and test data in', end-start, 's.')
        print('Estimating Bayesian inference')
        start = timer()
        [Weights_train, Weights_val, Weights_test] = [Y_pred_train_exp /
                                                      Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]
        [Weights_train, Weights_val, Weights_test] = [Weights_train/np.sum(Weights_train)*len(
            Weights_train), Weights_val/np.sum(Weights_val)*len(Weights_val), Weights_test/np.sum(Weights_test)*len(Weights_test)]
        quantiles = get_CI_from_sigma(
            [get_sigma_from_CI(0.5), 1, 2, 3, 4, 5])
        [quantiles_train, quantiles_val, quantiles_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles), weighted_quantiles(
            X_val[idx_val, 0], quantiles), weighted_quantiles(X_test[idx_test, 0], quantiles)]
        [quantiles_pred_train, quantiles_pred_val, quantiles_pred_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, Weights_train), weighted_quantiles(
            X_val[idx_val, 0], quantiles, Weights_val), weighted_quantiles(X_test[idx_test, 0], quantiles, Weights_test)]
        [one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, onesided=True), weighted_quantiles(
            X_val[idx_val, 0], quantiles, onesided=True), weighted_quantiles(X_test[idx_test, 0], quantiles, onesided=True)]
        [one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, Weights_train, onesided=True), weighted_quantiles(
            X_val[idx_val, 0], quantiles, Weights_val, onesided=True), weighted_quantiles(X_test[idx_test, 0], quantiles, Weights_test, onesided=True)]
        [central_quantiles_train, central_quantiles_val, central_quantiles_test] = [weighted_central_quantiles(
            X_train[idx_train, 0], quantiles), weighted_central_quantiles(X_val[idx_val, 0], quantiles), weighted_central_quantiles(X_test[idx_test, 0], quantiles)]
        [central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test] = [weighted_central_quantiles(X_train[idx_train, 0], quantiles, Weights_train), weighted_central_quantiles(
            X_val[idx_val, 0], quantiles, Weights_val), weighted_central_quantiles(X_test[idx_test, 0], quantiles, Weights_test)]
        [HPI_train, HPI_val, HPI_test] = [HPD_intervals(X_train[idx_train, 0], quantiles), HPD_intervals(
            X_val[idx_val, 0], quantiles), HPD_intervals(X_test[idx_test, 0], quantiles)]
        [HPI_pred_train, HPI_pred_val, HPI_pred_test] = [HPD_intervals(X_train[idx_train, 0], quantiles, Weights_train), HPD_intervals(
            X_val[idx_val, 0], quantiles, Weights_val), HPD_intervals(X_test[idx_test, 0], quantiles, Weights_test)]
        [one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test] = [((HPI_train[1][1][0][1]-HPI_train[1][1][0][0]) - (HPI_pred_train[1][1][0][1]-HPI_pred_train[1][1][0][0]))/(HPI_train[1][1][0][1]-HPI_train[1][1][0][0]),
                                                                                                ((HPI_val[1][1][0][1]-HPI_val[1][1][0][0]) - (HPI_pred_val[1][1][0][1]-HPI_pred_val[1][1][0][0]))/(HPI_val[1][1][0][1]-HPI_val[1][1][0][0]),
                                                                                                ((HPI_test[1][1][0][1]-HPI_test[1][1][0][0]) - (HPI_pred_test[1][1][0][1]-HPI_pred_test[1][1][0][0]))/(HPI_test[1][1][0][1]-HPI_test[1][1][0][0]), 
                                                                                                ((HPI_train[1][1][0][1]-HPI_train[1][1][0][0]) - (HPI_test[1][1][0][1]-HPI_test[1][1][0][0]))/(HPI_train[1][1][0][1]-HPI_train[1][1][0][0])]
        KS_test_pred_train = [[ks_w(X_test[idx_test,q],X_train[idx_train,q], np.ones(len(idx_test)), Weights_train)] for q in range(len(X_train[0]))]
        KS_test_pred_val = [[ks_w(X_test[idx_test,q],X_val[idx_val,q], np.ones(len(idx_test)), Weights_val)] for q in range(len(X_train[0]))]
        KS_val_pred_test = [[ks_w(X_val[idx_val,q],X_test[idx_test,q], np.ones(len(idx_val)), Weights_test)] for q in range(len(X_train[0]))]
        KS_train_test = [[ks_w(X_train[idx_train,q],X_test[idx_test,q], np.ones(len(idx_train)), np.ones(len(idx_test)))] for q in range(len(X_train[0]))]
        KS_test_pred_train_median = np.median(np.array(KS_test_pred_train)[:,0][:,1])
        KS_test_pred_val_median = np.median(np.array(KS_test_pred_val)[:, 0][:, 1])
        KS_val_pred_test_median = np.median(np.array(KS_val_pred_test)[:,0][:,1])
        KS_train_test_median = np.median(np.array(KS_train_test)[:, 0][:, 1])
        end = timer()
        print('Bayesian inference done in', end-start, 's.')
        [tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean] = ["None", "None", "None", "None", "None", "None", "None"]
        if FREQUENTISTS_RESULTS:
             print('Estimating frequentist inference')
             start_tmu = timer()
             blst = [
                 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
             tmuexact = np.array(
                 list(map(tmu, blst)))
             tmuDNN = np.array(
                 list(map(lambda x: tmu_DNN(x, model, scalerX, scalerY), blst)))
             [tmusample001, tmusample005, tmusample01, tmusample02] = [np.array(list(map(
                 lambda x: tmu_sample(x, X_train, Y_train, binsize), blst))) for binsize in [0.01, 0.05, 0.1, 0.2]]
             tmu_err_mean = np.mean(
                 np.abs(tmuexact[:, -1]-tmuDNN[:, -1]))
             end_tmu = timer()
             print('Frequentist inference done in', end-start, 's.')
        end_global = timer()
        print('Total time for predictions:',end_global-start_global,'s')
        return [min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
                quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
                one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
                central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test, 
                HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time]

    def generate_summary_log(model,now,FILE_SAMPLES,n_dim,NEVENTS_TRAIN,NEVENTS_VAL,WEIGHT_SAMPLES,SCALE_X,SCALE_Y,LOSS,HID_LAYERS,DROPOUT_RATE,EARLY_STOPPING,REDUCE_LR_PATIENCE,ACT_FUNC_OUT_LAYER,BATCH_NORM,
                             LEARNING_RATE,BATCH_SIZE,EXACT_EPOCHS,GPU_names,N_GPUS,training_time,min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
                             quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
                             one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
                             central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test, 
                             HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                             KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, FREQUENTISTS_RESULTS,
                             tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean):
        summary_log = {}
        summary_log['Date time'] = str(now)
        summary_log['Samples file'] = FILE_SAMPLES
        summary_log['n_dim'] = n_dim
        summary_log['Nevt-train'] = NEVENTS_TRAIN
        summary_log['Nevt-val'] = NEVENTS_VAL
        summary_log['Weighted'] = WEIGHT_SAMPLES
        summary_log['Scaled X'] = SCALE_X
        summary_log['Scaled Y'] = SCALE_Y
        if type(LOSS) == str:
            summary_log['Loss'] = LOSS
        else:
            summary_log['Loss'] = str(LOSS).split(' ')[1]
        summary_log['Hidden layers'] = HID_LAYERS
        summary_log['Params'] = model_params(model)
        summary_log['Trainable params'] = model_trainable_params(model)
        summary_log['Non-trainable params'] = model_non_trainable_params(model)
        summary_log['Dropout'] = DROPOUT_RATE
        summary_log['Early stopping'] = EARLY_STOPPING
        summary_log['Reduce LR patience'] = REDUCE_LR_PATIENCE
        summary_log['AF out'] = ACT_FUNC_OUT_LAYER
        summary_log['Batch norm'] = BATCH_NORM
        summary_log['Optimizer'] = 'Adam (LR' + str(LEARNING_RATE)+')'
        summary_log['Batch size'] = BATCH_SIZE
        summary_log['Epochs'] = EXACT_EPOCHS
        summary_log['GPU(s)'] = GPU_names[:N_GPUS]
        summary_log['Training time'] = training_time
        summary_log['Min loss scaled train'] = min_loss_scaled_train
        summary_log['Min loss scaled val'] = min_loss_scaled_val
        summary_log['Min loss scaled test'] = min_loss_scaled_test
        summary_log['Mape on exp train'] = mape_on_exp_train
        summary_log['Mape on exp val'] = mape_on_exp_val
        summary_log['Mape on exp test'] = mape_on_exp_test
        summary_log['Quantiles train'] = quantiles_train
        summary_log['Quantiles val'] = quantiles_val
        summary_log['Quantiles test'] = quantiles_test
        summary_log['Quantiles pred train'] = quantiles_pred_train
        summary_log['Quantiles pred val'] = quantiles_pred_val
        summary_log['Quantiles pred test'] = quantiles_pred_test
        summary_log['One-sided quantiles train'] = one_sised_quantiles_train
        summary_log['One-sided quantiles val'] = one_sised_quantiles_val
        summary_log['One-sided quantiles test'] = one_sised_quantiles_test
        summary_log['One-sided quantiles pred train'] = one_sised_quantiles_pred_train
        summary_log['One-sided quantiles pred val'] = one_sised_quantiles_pred_val
        summary_log['One-sided quantiles pred test'] = one_sised_quantiles_pred_test
        summary_log['Central quantiles train'] = central_quantiles_train
        summary_log['Central quantiles val'] = central_quantiles_val
        summary_log['Central quantiles test'] = central_quantiles_test
        summary_log['Central quantiles pred train'] = central_quantiles_pred_train
        summary_log['Central quantiles pred val'] = central_quantiles_pred_val
        summary_log['Central quantiles pred test'] = central_quantiles_pred_test
        summary_log['HPI train'] = HPI_train
        summary_log['HPI val'] = HPI_val
        summary_log['HPI test'] = HPI_test
        summary_log['HPI pred train'] = HPI_pred_train
        summary_log['HPI pred val'] = HPI_pred_val
        summary_log['HPI pred test'] = HPI_pred_test
        summary_log['1$\sigma$ HPI rel err train'] = one_sigma_HPI_rel_err_train
        summary_log['1$\sigma$ HPI rel err val'] = one_sigma_HPI_rel_err_val
        summary_log['1$\sigma$ HPI rel err test'] = one_sigma_HPI_rel_err_test
        summary_log['1$\sigma$ HPI rel err train-test'] = one_sigma_HPI_rel_err_train_test
        summary_log['KS test-pred_train'] = KS_test_pred_train
        summary_log['KS test-pred_val'] = KS_test_pred_val
        summary_log['KS val-pred_test'] = KS_val_pred_test
        summary_log['KS train-test'] = KS_train_test
        summary_log['KS test-pred_train median'] = KS_test_pred_train_median
        summary_log['KS test-pred_val median'] = KS_test_pred_val_median
        summary_log['KS val-pred_test median'] = KS_val_pred_test_median
        summary_log['KS train-test median'] = KS_train_test_median
        summary_log['Prediction time'] = prediction_time
        if FREQUENTISTS_RESULTS:
            summary_log['Frequentist tmu exact'] = tmuexact.tolist()
            summary_log['Frequentist tmu DNN'] = tmuDNN.tolist()
            summary_log['Frequentist tmu sample 0.01'] = tmusample001.tolist()
            summary_log['Frequentist tmu sample 0.05'] = tmusample005.tolist()
            summary_log['Frequentist tmu sample 0.1'] = tmusample01.tolist()
            summary_log['Frequentist tmu sample 0.2'] = tmusample02.tolist()
            summary_log['Frequentist mean error on tmu'] = tmu_err_mean.tolist()
        return summary_log

    def generate_title(summary_log):
        title = summary_log['Date time'] + " - "
        title = title + "n_dim: " + str(summary_log['n_dim']) + " - "
        title = title + "Nevt: " + '%.E' % Decimal(str(summary_log['Nevt-train'])) + " - "
        title = title + "Layers: " + str(len(summary_log['Hidden layers'])) + " - "
        title = title + "Nodes: " + str(summary_log['Hidden layers'][0][0]) + " - "
        title = title.replace("+","") + "Loss: " + str(summary_log['Loss'])
        return title

    def generate_summary_text(summary_log,history,FREQUENTISTS_RESULTS):
        summary_text = "Layers: " + str(summary_log['Hidden layers']) + "\n"
        summary_text = summary_text + "Pars: " + str(summary_log['Params']) + "\n"
        summary_text = summary_text + "Trainable pars: " + str(summary_log['Trainable params']) + "\n"
        summary_text = summary_text + "Non-trainable pars: " + str(summary_log['Non-trainable params']) + "\n"
        summary_text = summary_text + "Scaled X: " + str(summary_log['Scaled X']) + "\n"
        summary_text = summary_text + "Scaled Y: " + str(summary_log['Scaled Y']) + "\n"
        summary_text = summary_text + "Dropout: " + str(summary_log['Dropout']) + "\n"
        summary_text = summary_text + "Early stopping: " + str(summary_log['Early stopping']) + "\n"
        summary_text = summary_text + "Reduce LR patience: " + str(summary_log['Reduce LR patience']) + "\n"
        summary_text = summary_text + "AF out: " + str(summary_log['AF out']) + "\n"
        summary_text = summary_text + "Batch norm: " + str(summary_log['Batch norm']) + "\n"
        summary_text = summary_text + "Loss: " + str(summary_log['Loss']) + "\n"
        summary_text = summary_text + "Optimizer: " + summary_log['Optimizer'] + "\n"
        summary_text = summary_text + "Batch size: " + str(summary_log['Batch size']) + "\n"
        summary_text = summary_text + "Epochs: " + str(summary_log['Epochs']) + "\n"
        summary_text = summary_text + "GPU(s): " + str(summary_log['GPU(s)']) + "\n"
        summary_text = summary_text + "Min losses: " + '[' + '{0:1.2e}'.format(min(history['loss'])) + ',' '{0:1.2e}'.format(min(history['val_loss'])) + ']' + "\n"
        summary_text = summary_text + "Min losses scaled: " + '[' + '{0:1.2e}'.format(summary_log['Min loss scaled train']) + ',' + '{0:1.2e}'.format(summary_log['Min loss scaled val']) + ',' + '{0:1.2e}'.format(summary_log['Min loss scaled test']) + "\n"
        summary_text = summary_text + "Pred. mape on exp: " + '[' + '{0:1.2e}'.format(
            summary_log['Mape on exp train']) + ', ' + '{0:1.2e}'.format(summary_log['Mape on exp val']) + ',' + '{0:1.2e}'.format(summary_log['Mape on exp test']) + ']' + "\n"
        summary_text = summary_text + "1$\sigma$ HPI rel err: " + '[' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err train']) + ',' + '{0:1.2e}'.format(
            summary_log['1$\sigma$ HPI rel err val']) + ',' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err test']) + ',' + '{0:1.2e}'.format(summary_log['1$\sigma$ HPI rel err train-test']) + ']'  + "\n"
        summary_text = summary_text + "KS $p$-median: " + '[' + '{0:1.2e}'.format(summary_log['KS test-pred_train median']) + ',' + '{0:1.2e}'.format(
            summary_log['KS test-pred_val median']) + ',' + '{0:1.2e}'.format(summary_log['KS val-pred_test median']) + ',' + '{0:1.2e}'.format(summary_log['KS train-test median']) + ']' + "\n"
        if FREQUENTISTS_RESULTS:
            summary_text = summary_text + "Mean error on tmu: "+ str(summary_log['Frequentist mean error on tmu']) + "\n"
        summary_text = summary_text + "Train time: " + str(round(summary_log['Training time'],1)) + "s" + "\n"
        summary_text = summary_text + "Pred time: " + str(round(summary_log['Prediction time'],1)) + "s"
        return summary_text


    def generate_training_data(GENERATE_DATA, GENERATE_DATA_ON_THE_FLY, LOAD_MODEL, allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, FILE_SAMPLES, LOGPROB_THRESHOLD_INDICES_TRAIN,LOGPROB_THRESHOLD_INDICES_TEST, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y):
        if GENERATE_DATA:
            print('Generating training data')
            if len(LOGPROB_THRESHOLD_INDICES_TRAIN) < int(NEVENTS_TRAIN+NEVENTS_VAL):
                print(
                    'Please increase LOGPROB_THRESHOLD or reduce NEVENTS. There are not enough training (and validation) samples with the requires LOGPROB_THRESHOLD.')
                CONTINUE = False
            if len(LOGPROB_THRESHOLD_INDICES_TEST) < int(NEVENTS_TEST):
                print(
                    'Please increase LOGPROB_THRESHOLD or reduce NEVENTS. There are not enough test samples with the requires LOGPROB_THRESHOLD.')
                CONTINUE = False
            else:
                CONTINUE = True
            if CONTINUE:
                rnd_indices = np.random.choice(LOGPROB_THRESHOLD_INDICES_TRAIN, size=int(NEVENTS_TRAIN+NEVENTS_VAL), replace= False)
                rnd_indices_test = LOGPROB_THRESHOLD_INDICES_TEST[:NEVENTS_TEST]
                [rnd_indices_train, rnd_indices_val] = train_test_split(rnd_indices, train_size=NEVENTS_TRAIN, test_size=NEVENTS_VAL)
                #sample = [allsamples_mixed[rnd_indices], logprob_values_mixed[rnd_indices], nbI_values_mixed[rnd_indices],sample_weights]
                if GENERATE_DATA_ON_THE_FLY:
                    [X_train, Y_train] = import_XY_train(FILE_SAMPLES, rnd_indices_train)
                    [X_val, Y_val] = import_XY_train(FILE_SAMPLES, rnd_indices_val)
                    [X_test, Y_test] = import_XY_test(FILE_SAMPLES, rnd_indices_test)
                else:
                    [X_train, Y_train] = [allsamples_train[rnd_indices_train],logprob_values_train[rnd_indices_train]]
                    [X_val, Y_val] = [allsamples_train[rnd_indices_val],logprob_values_train[rnd_indices_val]]
                    [X_test, Y_test] = [allsamples_test[rnd_indices_test],logprob_values_test[rnd_indices_test]]
                if WEIGHT_SAMPLES:
                    W_train = compute_sample_weights(X_train, 500,power=1/1.3)
                    W_val = compute_sample_weights(X_val, 500,power=1/1.3)
                    W_test = compute_sample_weights(X_test, 500,power=1/1.3)
                else:
                    W_train = np.full(len(X_train), 1)
                    W_val = np.full(len(X_val), 1)
                    W_test = np.full(len(X_test), 1)
                if SCALE_X:
                    scalerX = StandardScaler(
                        with_mean=True, with_std=True)
                    scalerX.fit(X_train)
                else:
                    scalerX = StandardScaler(
                        with_mean=False, with_std=False)
                    scalerX.fit(X_train)
                if SCALE_Y:
                    scalerY = StandardScaler(
                        with_mean=True, with_std=True)
                    scalerY.fit(Y_train.reshape(-1, 1))
                else:
                    scalerY = StandardScaler(
                        with_mean=False, with_std=False)
                    scalerY.fit(Y_train.reshape(-1, 1))
        elif GENERATE_DATA == False and LOAD_MODEL != 'None':
            print('Loading data for model',LOAD_MODEL)
            [rnd_indices_train, rnd_indices_val,rnd_indices_test] = load_data_indices(LOAD_MODEL)
            if GENERATE_DATA_ON_THE_FLY:
                [X_train, Y_train]= import_XY_train(FILE_SAMPLES, rnd_indices_train)
                [X_val, Y_val]= import_XY_train(FILE_SAMPLES, rnd_indices_val)
                [X_test, Y_test]= import_XY_test(FILE_SAMPLES, rnd_indices_test)
            else:
                [X_train, Y_train] = [allsamples_train[rnd_indices_train], logprob_values_train[rnd_indices_train]]
                [X_val, Y_val] = [allsamples_train[rnd_indices_val], logprob_values_train[rnd_indices_val]]
                [X_test, Y_test] = [allsamples_train[rnd_indices_test], logprob_values_train[rnd_indices_test]]
            if WEIGHT_SAMPLES:
                W_train = compute_sample_weights(X_train, 500, power=1/1.3)
                W_val = compute_sample_weights(X_val, 500, power=1/1.3)
                W_test = compute_sample_weights(X_test, 500, power=1/1.3)
            else:
                W_train= np.full(len(X_train), 1)
                W_val= np.full(len(X_val), 1)
                W_test= np.full(len(X_test), 1)
            [scalerX, scalerY]=load_model_scaler(LOAD_MODEL)
        try:
            [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY]
            CONTINUE = True
            return [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY, CONTINUE]
        except:
            print("No training data have been generated (GENERATE_DATA=False) and LOAD_MODEL = 'None'. Please change your selection to continue training.")
            CONTINUE = False
            return ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', CONTINUE]

    #################### Custom metrics and losses
    def R2_metric(self,y_true, y_pred):
        MSE_model =  K.sum(K.square( y_true-y_pred )) 
        MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - MSE_model/(MSE_baseline + K.epsilon()))

    def Rt_metric(self,y_true, y_pred):
        MAPE_model =  K.sum(K.abs( 1-y_pred/(y_true + K.epsilon()))) 
        MAPE_baseline = K.sum(K.abs( 1-K.mean(y_true)/(y_true+ K.epsilon()) ) ) 
        return ( 1 - MAPE_model/(MAPE_baseline + K.epsilon()))


################## Generate random model name and folder
#        self.__modname__ = __modname__
#        self.__folder__ = __folder__
#
#        self.data_sample_input_filename_train = self.__folder__+self.__modname__+"_data_sample_train.pickle"
#        self.data_sample_input_filename_val = self.__folder__+self.__modname__+"_data_sample_val.pickle"
#        self.data_sample_input_filename_test = self.__folder__+self.__modname__+"_data_sample_test.pickle"
#
#        self.data_idx_train_input_filename = self.__folder__+self.__modname__+"_data_indices_train.pickle"
#        self.data_idx_val_input_filename = self.__folder__+self.__modname__+"_data_indices_val.pickle"
#        self.data_idx_test_input_filename = self.__folder__+self.__modname__+"_data_indices_test.pickle"
#
#        self.__data_sample_input_filename_train__ = __data_sample_input_filename_train__
#        self.__nevents_total_train__ = __nevents_total_train__
#        self.__data_sample_input_filename_test__ = __data_sample_input_filename_test__
#        self.__nevents_total_test__ = __nevents_total_test__
#        self.__data_idx_test_input_filename__ = __data_idx_test_input_filename__
#
#        if data_idx_input_basefilename is not None:
#            self.data_idx_train_input_filename = os.path.splitext(data_idx_input_basefilename)[0]+"_samples_indices_train.pickle"
#
#        if data_idx_train_input_filename is not None:
#            self.data_idx_train_input_filename = data_idx_train_input_filename
#            self.indices_train = self.load_data_indices(self.data_idx_train_input_filename)
#            self.indices_val = self.load_data_indices(self.data_idx_val_input_filename)
#            self.nevents_train = len(self.indices_train)
#            self.nevents_val = len(self.indices_val)
#        else:
#            if data_idx_train_output_filename is None:
#                self.data_idx_train_output_filename = r"%s" % (folder + "/" + modname + "_samples_indices.pickle"
#            self.nevents_train = nevents_train
#            self.nevents_val = nevents_val
#            rnd_indices = np.random.choice(np.arange(self.__nevents_total_train__), size=self.nevents_train+self.nevents_val, replace=False)
#            [self.indices_train, self.indices_test] = self.train_test_split(rnd_indices, train_size=self.nevents_train, test_size=self.nevents_val)
#            self.save_data_indices_train(self.data_idx_train_output_filename,self.indices_train,self.indices_test)
#        self.indices_test = self.load_data_indices_test(self.data_idx_test_input_filename)
#        self.nevents_test = len(self.indices_test)
#
############### Start defining
#
#        def save_data_indices(file,idx):
#            pickle_out = open(file, 'wb')
#            pickle.dump(idx, pickle_out, protocol=4)
#            pickle_out.close()
#
#        def load_data_indices(file):
#            file = file.replace('model.h5', 'samples_indices.pickle')
#            pickle_in = open(file, 'rb')
#            idx = pickle.load(pickle_in)
#            pickle_in.close()
#            return idx


