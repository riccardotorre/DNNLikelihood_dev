import codecs
import json
import math
import multiprocessing
import os
import pickle
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from timeit import default_timer as timer

import ipywidgets as widgets
import joblib
import keras
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import tensorflow as tf
from corner import corner, quantile
from IPython.display import Javascript, display
from jupyterthemes import jtplot
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import (EarlyStopping, History, ReduceLROnPlateau,
                             TerminateOnNaN, ModelCheckpoint, LambdaCallback)
from keras.layers import (AlphaDropout, BatchNormalization, Dense, Dropout,
                          Input)
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.utils import plot_model, multi_gpu_model
import keras2onnx
import onnx
from livelossplot import PlotLossesKeras
from pandas import DataFrame
from scipy import stats
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn import datasets
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adabound import AdaBound

# import the toy_likelihood module is needed by the tmu and related functions used here
import toy_likelihood
from toy_likelihood import *

tf.random.set_random_seed(1)
#np.random.seed(1)
pd.set_option('max_colwidth', 150)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
th_props = [
  ('font-size', '10px'),
  ('font-family', 'times')
  ]

# Set CSS properties for td elements in dataframe
td_props = [
  ('font-size', '10px'),
  ('font-family', 'times')
  ]
# Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]
# initialize seed
seed = 111511
K.set_floatx('float64')
#K.set_epsilon(1e-20)
#print('K.float = ',K.floatx())
#print('K.epsilon = ',K.epsilon())

availableCPUCoresNumber = multiprocessing.cpu_count()
print(str(availableCPUCoresNumber)+" CPU cores available")

kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)
    
def highlight_cols(s, coldict):
    if s.name in coldict.keys():
        return ['background-color: {}'.format(coldict[s.name])] * len(s)
    return [''] * len(s)


def sortby(df, column_min=None, column_max=None, color_min='Green', color_max='Red', 
           highlights_min=['loss_best', 'mse_best', 'mae_best', 'mape_best', 'me_best', 'mpe_best',
                           'val_loss_best', 'val_mse_best', 'val_mae_best', 'val_mape_best', 'val_me_best', 'val_mpe_best',
                           'test_loss_best', 'test_mse_best', 'test_mae_best', 'test_mape_best', 'test_me_best', 'test_mpe_best'],
           highlights_max=['KS test-pred_train median','KS test-pred_val median', 'KS val-pred_test median', '"KS train-test median']):
    df = df.reset_index(drop=True)
    if column_min != None and column_max != None:
        print("Cannot sort by max and min on two different columns simultaneously. Please specify either column_min or column_max")
        return None
    if column_min != None:
        df = df.sort_values(by=[column_min])
        df = df.style.apply(highlight_cols, coldict={column_min: color_min})
    if column_max != None:
        df = df.sort_values(by=[column_max],ascending=False)
        df = df.style.apply(highlight_cols, coldict={column_max: color_max})
    df_styled = df.set_table_styles(
        styles).highlight_min(subset=highlights_min, color=color_min).highlight_max(subset=highlights_max, color=color_max)
    return df_styled
    
def plot_loss_vs_time(df,xdim=16,ydim=8,labelsfontsize=12):
    #df=df[(df['Nevt']==10**5) & (df['Dropout']==0)]
    x = np.array(df[['Training time']])
    x = np.array([i[0] for i in x])
    y = np.array(df[['min_val_loss']])
    y = np.array([i[0] for i in y])
    names = df[['Hidden layers']].values
    names = np.array([[str(len(i[0]))+'; '+str(i[0][0][0])+'; '+str(i[0][0][1])][0] for i in names])
#    names = np.array([str(i[0]) for i in names])
    #sizes = np.array([sum([q[0] for q in j]) for j in [i[0] for i in df[['Hidden layers']].values]])
    #nlayers = np.array([1/len([q[0] for q in j]) for j in [i[0] for i in df[['Hidden layers']].values]])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #fig, ax = plt.subplots()
    fig = plt.figure(1,figsize=(xdim,ydim))
    #ax = fig.add_subplot(111)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #scatter = ax.plot(x, y, marker='o', linestyle='',markersize=sizes/30)#,markerfacecolor=nlayers,markersize=sizes/30)#,alpha=0.3)
    #scatter = ax.scatter(x,y,s=sizes/30,c=nlayers,alpha=0.3)
    
    for i in range(len(x)):
        plt.annotate(names[i], (x[i], y[i]), fontsize=labelsfontsize)
    
    plt.axis([min(x)*0.1,max(x)/0.9,min(y)*0.5, max(y)/0.5])
    plt.grid(linestyle="--", dashes=(5,5))
    plt.title(r"min val loss vs training time",fontsize = 20)#, color='black')
    plt.xlabel(r"Training time [s]",fontsize = 24)#, color='black')
    plt.ylabel(r"min val loss",fontsize = 24)#, color='black')
    plt.xticks(fontsize=18)#, color='black')
    plt.yticks(fontsize=18)#, color='black')
    plt.tight_layout()
    #tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=names, css=css)
    #tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=names,
    #                                   voffset=10, hoffset=10)
    plt.yscale('log')
    return fig.show()
    #mpld3.plugins.connect(fig, tooltip)
    #plt.yscale('log')
    #return mpld3.display()
    #mpld3.close()
    #fig.close()

def import_results(folders):
    mylist = []
    i=0
    for thisdir in folders:
        # r=root, d=directories, f = files
        for r, _, f in os.walk(thisdir):
            for file in f:
                if "history.json" in file:
                    current_file = os.path.join(r, file)
                    with open(current_file) as json_file: 
        #                print(json_file)
                        data = json.load(json_file)
                    mylist.append({**{'Scan': thisdir}, **{'Number': i}, **{'Name': current_file},**data})
                    i = i + 1
    for dic in mylist:
        if 'loss' in dic.keys():
            dic[dic['Loss']]=dic['loss']
            #dic['loss_last'] = dic[dic['Loss']][-1]
            #dic[dic['Loss']+"_last"] = dic[dic['Loss']][-1]
        if 'val_loss' in dic.keys():
            dic['val_'+dic['Loss']]=dic['val_loss']
            #dic['val_'+dic['Loss']+"_last"] = dic['val_'+dic['Loss']][-1]
            #dic['val_loss_last'] = dic['val_'+dic['Loss']][-1]
        if 'mean_absolute_percentage_error' in dic.keys():
            dic['mape']=dic.pop('mean_absolute_percentage_error')
            #dic['mape_last'] = dic['mape'][-1]
        if 'val_mean_absolute_percentage_error' in dic.keys():
            dic['val_mape']=dic.pop('val_mean_absolute_percentage_error')
        if 'mean_squared_error' in dic.keys():
            dic['mse']=dic.pop('mean_squared_error')
        if 'val_mean_squared_error' in dic.keys():
            dic['val_mse']=dic.pop('val_mean_squared_error')
        if 'mean_squared_logarithmic_error' in dic.keys():
            dic['msle']=dic.pop('mean_squared_logarithmic_error')
        if 'val_mean_squared_logarithmic_error' in dic.keys():
            dic['val_msle']=dic.pop('val_mean_squared_logarithmic_error')
        if 'mean_absolute_error' in dic.keys():
            dic['mae']=dic.pop('mean_absolute_error')
        if 'val_mean_absolute_error' in dic.keys():
            dic['val_mae']=dic.pop('val_mean_absolute_error')
        if 'mean_error' in dic.keys():
            dic['me'] = dic.pop('mean_error')
        if 'val_mean_error' in dic.keys():
            dic['val_me'] = dic.pop('val_mean_error')
        if 'mean_percentage_error' in dic.keys():
            dic['mpe'] = dic.pop('mean_percentage_error')
        if 'val_mean_percentage_error' in dic.keys():
            dic['val_mpe'] = dic.pop('val_mean_percentage_error')
    print(str(i)+' files imprediction_timeported')
    mydataframe = pd.DataFrame.from_dict(mylist)
    return mydataframe

def import_model(folders):
    mylist = []
    i=0
    for thisdir in folders:
        # r=root, d=directories, f = files
        for r, _, f in os.walk(thisdir):
            for file in f:
                if "model.json" in file:
                    current_file = os.path.join(r, file)
                    with open(current_file) as json_file: 
        #                print(json_file)
                        data = json.load(json_file)
                    mylist.append({**{'Scan': thisdir}, **{'Number': i}, **{'Name': current_file},**data})
                    i = i + 1
    print(str(i)+' files imported')
    mydataframe = pd.DataFrame.from_dict(mylist)
    return mydataframe

def import_models(folders, true_strings = [""], false_strings = ["-!-"], listonly=True):
    mylist = list()
    i=0
    for thisdir in folders:
        # r=root, d=directories, f = files
        for r, d, f in os.walk(thisdir):
            for file in f:
                true_strings_bool = bool(np.prod([a in file for a in true_strings]))
                false_strings_bool = bool(np.prod([a not in file for a in false_strings]))
                if "model.h5" in file and true_strings_bool and false_strings_bool:
                    current_file = os.path.join(r, file)
                    if listonly:
                        print("Importing",current_file)
                        mylist.append(current_file)
                    else:
                        model = load_model(current_file, custom_objects={'R2_metric': R2_metric, 'Rt_metric': Rt_metric})
                        scalerX, scalerY = load_model_scaler(current_file)
                        #model = model.layers[3]
                        mylist.append([model,scalerX,scalerY])
                    i = i + 1
    return mylist

def ask_new_run():
    NEW_TRAIN = False
    display(Javascript("""
    require(
        ["base/js/dialog"], 
        function(dialog) {
            dialog.modal({
                title: 'Training mode',
                body: 'Do you want to continue training or start new training',
                buttons: {
                    'New training': {click: function(){Jupyter.notebook.kernel.execute('NEW_TRAIN=True')} },
                    'Continue training': {click: function(){Jupyter.notebook.kernel.execute('NEW_TRAIN=False')} }
                }
            });
        }
    );
    """))
    return NEW_TRAIN

## All these functions always take un-scaled X as inputs
def logprob_DNN_multi(samples,model,scalerX,scalerY,batch_size=1,threshold=-400):
    nnn = len(samples)
    logprob = model_predict(model,scalerX,scalerY,np.array(samples[0:nnn]),batch_size=batch_size)[0]
    if np.bool(np.prod(logprob<threshold)):
        logprob[logprob<threshold]=-np.inf
    if np.isnan(logprob).any():
        print("Warning: nan has been replaced with -np.inf.")
        #ind = np.where(np.isnan(logprob))
        #print(samples[ind[0]])
        logprob = np.nan_to_num(logprob)
        logprob[logprob==0]=-np.inf
        return logprob
    else:
        #print(logprob)
        return logprob

def logprob_DNN(sample, model, scalerX, scalerY):
    return logprob_DNN_multi([sample], model, scalerX, scalerY, batch_size=1)[0]

def minus_logprob_DNN(sample, model, scalerX, scalerY):
    logprob = model_predict(model,scalerX,scalerY,np.array([sample]),batch_size=1)[0][0]
    return -logprob
def minus_logprob_delta_DNN(delta,mu,model, scalerX, scalerY):
    pars = np.concatenate((np.array([mu]),delta))
    logprob = model_predict(model,scalerX,scalerY,np.array([pars]),batch_size=1)[0][0]
    return -logprob
def tmu_DNN(mu,model, scalerX, scalerY):
    minimum_logprob_DNN = minimize(lambda x: minus_logprob_DNN(x,model,scalerX,scalerY), np.full(95,0),method='Powell')['x']
    L_muhat_deltahat_DNN = -minus_logprob_DNN(minimum_logprob_DNN,model,scalerX,scalerY)
    #print([mu, L_muhat_deltahat_DNN])
    minimum_logprob_delta_DNN = np.concatenate((np.array([mu]),minimize(lambda x: minus_logprob_delta_DNN(x,mu,model,scalerX,scalerY), np.full(94,0),method='Powell')['x']))
    L_mu_deltahat_DNN = -minus_logprob_DNN(minimum_logprob_delta_DNN,model,scalerX,scalerY)
    #print([mu, L_mu_deltahat_DNN,-2*(L_mu_deltahat_DNN-L_muhat_deltahat_DNN)])
    return np.array([mu,L_muhat_deltahat_DNN,L_mu_deltahat_DNN,-2*(L_mu_deltahat_DNN-L_muhat_deltahat_DNN)])

def extend_corner_range(S1,S2,ilist,percent):
    res = []
    for i in ilist:
        minn = np.min([np.min(S1[:,i]),np.min(S2[:,i])])
        maxx = np.max([np.max(S1[:,i]),np.max(S2[:,i])])
        if minn<0:
            minn = minn*(1+percent/100)
        else:
            minn = minn*(1-percent/100)
        if maxx>0:
            maxx = maxx*(1+percent/100)
        else:
            maxx = maxx*(1-percent/100)
        res.append([minn,maxx])
    return res

def get_1d_hist(i_dim, xs, nbins=25, ranges=None, weights=None, intervals=None,normalize1d=False):
    """Assumes smooth1d = True
    """
    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"
    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")
    # Parse the parameter ranges.
    if ranges is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            ranges = hist2d_kwargs.pop("extents")
        else:
            ranges = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in ranges], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))
    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        ranges = list(ranges)
        for i, _ in enumerate(ranges):
            try:
                emin, emax = ranges[i]
            except TypeError:
                q = [0.5 - 0.5*ranges[i], 0.5 + 0.5*ranges[i]]
                ranges[i] = quantile(xs[i], q, weights=weights)
    if len(ranges) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")
    # Parse the bin specifications.
    try:
        bins = [int(nbins) for _ in ranges]
    except TypeError:
        if len(nbins) != len(ranges):
            raise ValueError("Dimension mismatch between bins and range")
    x = xs[i_dim]
    # Deal with masked arrays.
    if hasattr(x, "compressed"):
            x = x.compressed()
    # Get 1D curve.
    n, b = np.histogram(
        x, bins=bins[i_dim], weights=weights, range=np.sort(ranges[i_dim]))
    if normalize1d:
        n = n/n.sum()
    n = gaussian_filter(n, True)
    x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
    y0 = np.array(list(zip(n, n))).flatten()
    # Generate 1D curves in intervals.
    result = []
    if intervals is None:
        result.append([x0, y0])
    else:
        for interval in intervals:
            #print(interval[0],interval[1])
            tmp = np.transpose(np.append(x0, y0).reshape([2, len(x0)]))
            #print((tmp[:,0]>=interval[0])*(tmp[:,0]<=interval[1]))
            tmp = tmp[(tmp[:, 0] >= interval[0])*(tmp[:, 0] <= interval[1])]
            #print(tmp1)
            result.append([tmp[:, 0], tmp[:, 1]])
    return result

def save_results(folder, model, scalerX, scalerY, title, summary_text, X_train, X_test, Y_train, Y_test, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, pars=[0], labels='None', plot_coverage=False, plot_distr=True, plot_corners=True, plot_tmu=False, batch_size=1, verbose=True):

    if plot_tmu:
        figname = 'freq_tmu'
        plt.plot(tmuexact[:,0],tmuexact[:,-1])
        plt.plot(tmuDNN[:,0],tmuDNN[:,-1])
        plt.plot(tmusample001[:,1],tmusample001[:,-1])
        plt.plot(tmusample005[:,1],tmusample005[:,-1])
        plt.plot(tmusample01[:,1],tmusample01[:,-1])
        plt.plot(tmusample02[:,1],tmusample02[:,-1])
        plt.legend(['Exact','DNN','0.01','0.05','0.1','0.2'])
        x1,x2,_,_ = plt.axis()
        plt.axis([x1, x2, -0.5, 7.9])
        plt.savefig(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf"))
        if verbose:
            #plt.show()
            print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
                  " created and saved."))
        plt.close()


#def abs_error_quantiles(data, intervals=0.68, weights=None, onesided=False):
#    intervals = np.array([intervals]).flatten()
#    if not onesided:
#        return np.array([[i, np.abs(weighted_quantile(data, (1-i)/2, weights)-weighted_quantile(data, (1-i)/2)), np.abs(weighted_quantile(data, 0.5, weights)-weighted_quantile(data, 0.5)), np.abs(weighted_quantile(data, 1-(1-i)/2, weights)-weighted_quantile(data, 1-(1-i)/2))] for i in intervals])
#    else:
#        intervals = intervals - (1-intervals)
#        data = data[data > 0]
#        return np.array([[i, 0, np.abs(weighted_quantile(data, 0.5, weights)-weighted_quantile(data, 0.5)), np.abs(weighted_quantile(data, 1-(1-i)/2, weights)-weighted_quantile(data, 1-(1-i)/2))] for i in intervals])
#
#def rel_error_quantiles(data, intervals=0.68, weights=None, onesided=False):
#    intervals = np.array([intervals]).flatten()
#    if not onesided:
#        return np.array([[i, np.abs((weighted_quantile(data, (1-i)/2, weights)-weighted_quantile(data, (1-i)/2))/weighted_quantile(data, (1-i)/2)), np.abs((weighted_quantile(data, 0.5, weights)-weighted_quantile(data, 0.5))/weighted_quantile(data, 0.5)), np.abs((weighted_quantile(data, 1-(1-i)/2, weights)-weighted_quantile(data, 1-(1-i)/2))/weighted_quantile(data, 1-(1-i)/2))] for i in intervals])
#    else:
#        intervals = intervals - (1-intervals)
#        data = data[data > 0]
#        return np.array([[i, 0, np.abs((weighted_quantile(data, 0.5, weights)-weighted_quantile(data, 0.5))/weighted_quantile(data, 0.5)), np.abs((weighted_quantile(data, 1-(1-i)/2, weights)-weighted_quantile(data, 1-(1-i)/2))/weighted_quantile(data, 1-(1-i)/2))] for i in intervals])

def compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, LOSS, NEVENTS_TRAIN, BATCH_SIZE, FREQUENTISTS_RESULTS):
    print('Computing predictions')
    start_global = timer()
    start = timer()
    metrics_names = model.metrics_names
    #Choose NEVENTS_TRAIN random indices to pick data
    [idx_train, idx_val, idx_test] = [np.random.choice(np.arange(len(X)), min(
        int(NEVENTS_TRAIN), len(X)), replace=False) for X in [X_train, X_val, X_test]]
    #Redefine train/val/test data
    X_train = X_train[idx_train]
    X_val = X_val[idx_val]
    X_test = X_test[idx_test]
    Y_train = Y_train[idx_train]
    Y_val = Y_val[idx_val]
    Y_test = Y_test[idx_test]
    #Get logprobabilities and prediction time for selected subset of data
    Y_pred_train, _ = model_predict(model, scalerX, scalerY, X_train, batch_size=BATCH_SIZE)
    Y_pred_val, _ = model_predict(model, scalerX, scalerY, X_val, batch_size=BATCH_SIZE)
    Y_pred_test, prediction_time_test = model_predict(model, scalerX, scalerY, X_test, batch_size=BATCH_SIZE)
    prediction_time = prediction_time_test
    #Get probabilities exponentiating logprobabilities for both data and prediction
    [Y_train_exp, Y_val_exp, Y_test_exp] = [np.exp(Y_train), np.exp(Y_val), np.exp(Y_test)]
    [Y_pred_train_exp, Y_pred_val_exp, Y_pred_test_exp] = [np.exp(Y_pred_train), np.exp(Y_pred_val), np.exp(Y_pred_test)]
    #Evaluate the metrics on final best model
    metrics_names_train = [i+"_best" for i in model.metrics_names]
    metrics_names_val = ["val_"+i+"_best" for i in model.metrics_names]
    metrics_names_test = ["test_"+i+"_best" for i in model.metrics_names]
    metrics_train = model_evaluate(model, scalerX, scalerY, X_train, Y_train, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_val = model_evaluate(model, scalerX, scalerY, X_val, Y_val, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_test = model_evaluate(model, scalerX, scalerY, X_test, Y_test, batch_size=BATCH_SIZE)[0:len(metrics_names)]
    metrics_true = {**dict(zip(metrics_names_train,metrics_train)),**dict(zip(metrics_names_val,metrics_val)),**dict(zip(metrics_names_test,metrics_test))}
    print(metrics_true)
    #Evaluate min loss scaled
    metrics_names_train = [i+"_best_scaled" for i in model.metrics_names]
    metrics_names_val = ["val_"+i+"_best_scaled" for i in model.metrics_names]
    metrics_names_test = ["test_"+i+"_best_scaled" for i in model.metrics_names]
    metrics_train_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_train), tf.convert_to_tensor(Y_pred_train)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_val_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_val), tf.convert_to_tensor(Y_pred_val)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_test_scaled = [keras.losses.deserialize(l)(tf.convert_to_tensor(Y_test), tf.convert_to_tensor(Y_pred_test)).eval(session=tf.Session()) for l in [s.replace("loss", LOSS) for s in model.metrics_names]]
    metrics_scaled = {**dict(zip(metrics_names_train,metrics_train_scaled)),**dict(zip(metrics_names_val,metrics_val_scaled)),**dict(zip(metrics_names_test,metrics_test_scaled))}
    end = timer()
    print('Prediction on', NEVENTS_TRAIN,
          'points done form training, validation, and test data in', end-start, 's.')
    print('Estimating Bayesian inference')
    start = timer()
    #Computing and normalizing weights
    [Weights_train, Weights_val, Weights_test] = [Y_pred_train_exp / Y_train_exp, Y_pred_val_exp/Y_val_exp, Y_pred_test_exp/Y_test_exp]
    [Weights_train, Weights_val, Weights_test] = [Weights_train/np.sum(Weights_train)*len(Weights_train), Weights_val/np.sum(Weights_val)*len(Weights_val), Weights_test/np.sum(Weights_test)*len(Weights_test)]
    #Choosing probability intervals from gaussian sigmas
    quantiles = get_CI_from_sigma([get_sigma_from_CI(0.5), 1, 2, 3, 4, 5])
    #[quantiles_train, quantiles_val, quantiles_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles), weighted_quantiles(
    #    X_val[idx_val, 0], quantiles), weighted_quantiles(X_test[idx_test, 0], quantiles)]
    #[quantiles_pred_train, quantiles_pred_val, quantiles_pred_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, Weights_train), weighted_quantiles(
    #    X_val[idx_val, 0], quantiles, Weights_val), weighted_quantiles(X_test[idx_test, 0], quantiles, Weights_test)]
    #[one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, onesided=True), weighted_quantiles(
    #    X_val[idx_val, 0], quantiles, onesided=True), weighted_quantiles(X_test[idx_test, 0], quantiles, onesided=True)]
    #[one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test] = [weighted_quantiles(X_train[idx_train, 0], quantiles, Weights_train, onesided=True), weighted_quantiles(
    #    X_val[idx_val, 0], quantiles, Weights_val, onesided=True), weighted_quantiles(X_test[idx_test, 0], quantiles, Weights_test, onesided=True)]
    #[central_quantiles_train, central_quantiles_val, central_quantiles_test] = [weighted_central_quantiles(
    #    X_train[idx_train, 0], quantiles), weighted_central_quantiles(X_val[idx_val, 0], quantiles), weighted_central_quantiles(X_test[idx_test, 0], quantiles)]
    #[central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test] = [weighted_central_quantiles(X_train[idx_train, 0], quantiles, Weights_train), weighted_central_quantiles(
    #    X_val[idx_val, 0], quantiles, Weights_val), weighted_central_quantiles(X_test[idx_test, 0], quantiles, Weights_test)]
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
         blst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
         tmuexact = np.array(list(map(tmu, blst)))
         tmuDNN = np.array(list(map(lambda x: tmu_DNN(x, model, scalerX, scalerY), blst)))
         [tmusample001, tmusample005, tmusample01, tmusample02] = [np.array(list(map(lambda x: tmu_sample(x, X_train, Y_train, binsize), blst))) for binsize in [0.01, 0.05, 0.1, 0.2]]
         tmu_err_mean = np.mean(np.abs(tmuexact[:, -1]-tmuDNN[:, -1]))
         end_tmu = timer()
         print('Frequentist inference done in', start_tmu-end_tmu, 's.')
    end_global = timer()
    print('Total time for predictions:',end_global-start_global,'s')
    return [metrics_true, metrics_scaled,
            #mean_error_train, mean_error_val, mean_error_test, min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
            #quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
            #one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
            #central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test, 
            HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
            KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median,
            tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time]

def generate_summary_log(model,now,FILE_SAMPLES,NDIM,NEVENTS_TRAIN,NEVENTS_VAL,NEVENTS_TEST,WEIGHT_SAMPLES,SCALE_X,SCALE_Y,LOSS,HID_LAYERS,DROPOUT_RATE,EARLY_STOPPING,REDUCE_LR_PATIENCE,MIN_DELTA,ACT_FUNC_OUT_LAYER,BATCH_NORM,
                         LEARNING_RATE,BATCH_SIZE,EXACT_EPOCHS,GPU_names,N_GPUS,training_time, metrics, metrics_scaled, 
                         HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                         KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, FREQUENTISTS_RESULTS,
                         tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean):
    summary_log = {**metrics,**metrics_scaled}
    if FREQUENTISTS_RESULTS:
        summary_log['Frequentist tmu exact'] = tmuexact.tolist()
        summary_log['Frequentist tmu DNN'] = tmuDNN.tolist()
        summary_log['Frequentist tmu sample 0.01'] = tmusample001.tolist()
        summary_log['Frequentist tmu sample 0.05'] = tmusample005.tolist()
        summary_log['Frequentist tmu sample 0.1'] = tmusample01.tolist()
        summary_log['Frequentist tmu sample 0.2'] = tmusample02.tolist()
        summary_log['Frequentist mean error on tmu'] = tmu_err_mean.tolist()
    for key in list(summary_log.keys()):
        summary_log[metric_name_abbreviate(key)] = summary_log.pop(key)
    return summary_log

def model_training_scan(N_RUNS,ACT_FUNC_OUT_LAYER_LIST,BATCH_NORM_LIST,BATCH_SIZE_LIST,CONTINUE_TRAINING,DROPOUT_RATE_LIST,EARLY_STOPPING,FILE_SAMPLES_LIST,
                        FOLDER,FREQUENTISTS_RESULTS,GENERATE_DATA,GENERATE_DATA_ON_THE_FLY,GPU_NAMES,HID_LAYERS_LIST,LABELS,LEARNING_RATE_LIST,LOAD_MODEL,
                        LOGPROB_THRESHOLD, LOGPROB_THRESHOLD_INDICES_TRAIN, LOGPROB_THRESHOLD_INDICES_TEST, LOSS_LIST, METRICS, MIN_DELTA_LIST, MODEL_CHEKPOINT, MONITORED_METRIC, MULTI_GPU, N_EPOCHS, NEVENTS_TRAIN_LIST, PARS, PLOTLOSSES, REDUCE_LR, REDUCE_LR_PATIENCE_LIST,
                        SCALE_X, SCALE_Y, TEST_FRACTION, VALIDATION_FRACTION, WEIGHT_SAMPLES_LIST,
                        allsamples_train='None',logprob_values_train='None',allsamples_test='None',logprob_values_test='None',
                        rnd_indices_train='None', rnd_indices_val='None', rnd_indices_test='None', X_train='None', X_val='None', X_test='None',
                        Y_train='None',Y_val='None',Y_test='None',W_train='None',W_val='None',scalerX='None',scalerY='None',
                        model='None', training_model='None',summary_log='None',history='None', training_time='None'):
    start = timer()
    overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={'width': '500px', 'height': '14px', 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
    display(overall_progress)
    iterator = 0
    for FILE_SAMPLES in FILE_SAMPLES_LIST:
        if GENERATE_DATA_ON_THE_FLY:
            #if GENERATE_DATA == False:
            #    print('When GENERATE_DATA_OM_THE_FLY = True, GENERATE_DATA flag needs also to be set to True, while it was set to False. Flag automatically changed to True.')
            #    GENERATE_DATA = True
            print('Training and test data will be generated on-the-fly for each run to save RAM')
            logprob_values_train = import_Y_train(FILE_SAMPLES,'all')
            LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                logprob_values_train >= LOGPROB_THRESHOLD)[0]
            logprob_values_train = 'None'
            logprob_values_test = import_Y_test(FILE_SAMPLES, 'all')
            LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                logprob_values_test >= LOGPROB_THRESHOLD)[0]
            logprob_values_test = 'None'
        elif GENERATE_DATA and GENERATE_DATA_ON_THE_FLY == False:
            if [allsamples_train, logprob_values_train] != ['None', 'None']:
                print('Training data already loaded')
            else:
                print('Loading training data')
                allsamples_train, logprob_values_train = import_XY_train(FILE_SAMPLES,'all')
            if [allsamples_test, logprob_values_test] != ['None', 'None']:
                print('Test data already loaded')
            else:
                print('Loading test data')
                allsamples_test, logprob_values_test = import_XY_test(FILE_SAMPLES,'all')
            LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                logprob_values_train >= LOGPROB_THRESHOLD)[0]
            LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                logprob_values_test >= LOGPROB_THRESHOLD)[0]
        elif GENERATE_DATA == False and CONTINUE_TRAINING:
            CONTINUE = True
        else:
            try:
                LOGPROB_THRESHOLD_INDICES_TRAIN = np.nonzero(
                    logprob_values_train >= LOGPROB_THRESHOLD)[0]
                LOGPROB_THRESHOLD_INDICES_TEST = np.nonzero(
                    logprob_values_test >= LOGPROB_THRESHOLD)[0]
                print('Training and test data already loaded')
            except:
                print("No training data available, please generate them by setting GENERATE_DATA=True")
                CONTINUE = False
        if LOAD_MODEL != 'None':
            if CONTINUE_TRAINING == False:
                print('When loading model CONTINUE_TRAINING flag needs to be set to True, while it was set to False. Flag automatically changed to True.')
                CONTINUE_TRAINING = True
        if len(K.tensorflow_backend._get_available_gpus()) <= 1:
            MULTI_GPU = False
            N_GPUS = len(K.tensorflow_backend._get_available_gpus())
        if MULTI_GPU:
            N_GPUS = len(K.tensorflow_backend._get_available_gpus())
        else:
            N_GPUS = 1
        if MULTI_GPU:
            BATCH_SIZE_LIST = [i*N_GPUS for i in BATCH_SIZE_LIST]
        try:
            LABELS
            CONTINUE = True
        except:
            print("Please provide labels for the features")
            CONTINUE = False
        if CONTINUE:    
            for run in range(N_RUNS):
                # Start loop over N_RUNS
                for NEVENTS_TRAIN in NEVENTS_TRAIN_LIST:
                    NEVENTS_VAL = int(NEVENTS_TRAIN*VALIDATION_FRACTION)
                    NEVENTS_TEST = int(NEVENTS_TRAIN*TEST_FRACTION)
                    for WEIGHT_SAMPLES in WEIGHT_SAMPLES_LIST:
                        if CONTINUE_TRAINING == False or GENERATE_DATA or LOAD_MODEL != 'None':
                            [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY, CONTINUE] = generate_training_data(
                                GENERATE_DATA, GENERATE_DATA_ON_THE_FLY, LOAD_MODEL, allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, FILE_SAMPLES, LOGPROB_THRESHOLD_INDICES_TRAIN,LOGPROB_THRESHOLD_INDICES_TEST, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y)
                        if X_train != "None":
                            CONTINUE = True
                        else:
                            print("No training data available, please change flags to ensure data generation.")
                            CONTINUE = False
                        if CONTINUE:
                            for LOSS in LOSS_LIST:
                                for HID_LAYERS in HID_LAYERS_LIST:
                                    for ACT_FUNC_OUT_LAYER in ACT_FUNC_OUT_LAYER_LIST:
                                        for DROPOUT_RATE in DROPOUT_RATE_LIST:
                                            for BATCH_SIZE in BATCH_SIZE_LIST:
                                                for BATCH_NORM in BATCH_NORM_LIST:
                                                    for LEARNING_RATE in LEARNING_RATE_LIST:
                                                        for REDUCE_LR_PATIENCE in REDUCE_LR_PATIENCE_LIST:
                                                            for MIN_DELTA in MIN_DELTA_LIST:
                                                                now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                                                                NDIM = len(X_train[0])
                                                                #Model title
                                                                title = generate_title(now,NDIM,NEVENTS_TRAIN,HID_LAYERS,LOSS)
                                                                if LABELS == 'None':
                                                                    LABELS = [r"$x_{%d}$"%i for i in range(NDIM)]
                                                                OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.95, beta_2=0.999, epsilon=1e-10, decay=0.0, amsgrad=False)
                                                                #OPTIMIZER = AdaBound(lr=LEARNING_RATE, final_lr=LEARNING_RATE*100, gamma=1e-03, weight_decay=0., amsbound=False)
                                                                #OPTIMIZER = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
                                                                if CONTINUE_TRAINING == False:
                                                                    if model == "None":
                                                                        model = model_define(NDIM,HID_LAYERS,DROPOUT_RATE,ACT_FUNC_OUT_LAYER,BATCH_NORM,verbose=1)
                                                                    elif type(model.input_shape) == tuple:
                                                                        model = model_define(NDIM,HID_LAYERS,DROPOUT_RATE,ACT_FUNC_OUT_LAYER,BATCH_NORM,verbose=1)
                                                                    model = model_compile(model,LOSS,OPTIMIZER,METRICS,False)
                                                                    training_model = model_compile(model,LOSS,OPTIMIZER,METRICS,MULTI_GPU)
                                                                    [history, training_time] = [{}, 0]
                                                                    #training_time = 0
                                                                #Model training
                                                                if LOAD_MODEL != 'None':
                                                                    print('Loading model',LOAD_MODEL)
                                                                    model = load_model(LOAD_MODEL, custom_objects={'R2_metric': R2_metric, 'Rt_metric':Rt_metric})
                                                                    model = model_compile(model,LOSS,OPTIMIZER,METRICS,False)
                                                                    training_model = model_compile(model,LOSS,OPTIMIZER,METRICS,MULTI_GPU)
                                                                    LOAD_MODEL = 'None'
                                                                if CONTINUE_TRAINING:
                                                                    print('Continue training of loaded model')
                                                                else:
                                                                    if LOAD_MODEL != 'None':
                                                                        print('Continue training of loaded model')
                                                                    else:
                                                                        print('Start training of new model')
                                                                #print("Train with parameters:")
                                                                #print("monitored_metric:", MONITORED_METRIC)
                                                                #print("early_stopping:",EARLY_STOPPING)
                                                                #print("reduceLR:",REDUCE_LR)
                                                                #print("reduce_LR_patience:",REDUCE_LR_PATIENCE)
                                                                #print("min_delta:",MIN_DELTA)
                                                                [h_run, training_time_run] = model_train(training_model, X_train, Y_train, X_val, Y_val, scalerX, scalerY, N_EPOCHS, BATCH_SIZE,
                                                                                                         sample_weights=W_train, folder=FOLDER, title=title, monitored_metric=MONITORED_METRIC, 
                                                                                                         plotlosses = PLOTLOSSES, model_checkpoint=MODEL_CHEKPOINT, early_stopping=EARLY_STOPPING,restore_best_weights=True,
                                                                                                         reduceLR=REDUCE_LR,reduce_LR_patience=REDUCE_LR_PATIENCE, min_delta=MIN_DELTA, verbose=2)
                                                                if CONTINUE_TRAINING == False:# and LOAD_MODEL == 'None':
                                                                    [history, training_time] = [h_run.history, training_time_run]
                                                                else:
                                                                    if LOAD_MODEL != 'None':
                                                                        with open(LOAD_MODEL.replace('model.h5','history.json')) as json_file:
                                                                            history = json.load(json_file)
                                                                        history_full = {}
                                                                        history_run = h_run.history
                                                                        for key in history_run.keys():
                                                                            history_full[key] = history[key] + history_run[key]
                                                                        [history, training_time] = [history_full, training_time + training_time_run]
                                                                        #training_time = training_time + training_time_run
                                                                        del(history_full,history_run)
                                                                    else:    
                                                                        history_full = {}
                                                                        history_run = h_run.history
                                                                        for key in history_run.keys():
                                                                            history_full[key] = history[key] + history_run[key]
                                                                        [history, training_time] = [history_full, training_time + training_time_run]
                                                                        #training_time = training_time + training_time_run
                                                                        del(history_full,history_run)
                                                                # compute predictions
                                                                [metrics, metrics_scaled,
                                                                HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                                KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median,
                                                                tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time] = compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, LOSS, NEVENTS_TRAIN, BATCH_SIZE, FREQUENTISTS_RESULTS)
                                                                #print('Relative error on credibility intervals validation:', prob_intervals_pred_val_rel_error[1])
                                                                #print('KS test-pred_train/KS test-pred_val/KS val-pred_test/KS train-test median:', str(KS_test_pred_train_median),'/',str(KS_test_pred_val_median),'/',str(KS_val_pred_test_median),'/',str(KS_train_test_median))
                                                                #print('MAPE on exp (train/test/val):',mape_on_exp_train,'/',mape_on_exp_val,'/',mape_on_exp_test)
                                                                EXACT_EPOCHS = len(history['loss'])
                                                                #Model log
                                                                summary_log = generate_summary_log(model,now, FILE_SAMPLES, NDIM, NEVENTS_TRAIN, NEVENTS_VAL, NEVENTS_TEST, WEIGHT_SAMPLES, SCALE_X, SCALE_Y, LOSS, HID_LAYERS, DROPOUT_RATE, EARLY_STOPPING, REDUCE_LR_PATIENCE, MIN_DELTA, ACT_FUNC_OUT_LAYER, BATCH_NORM,
                                                                                                   LEARNING_RATE, BATCH_SIZE, EXACT_EPOCHS, GPU_NAMES, N_GPUS, training_time, metrics, metrics_scaled,
                                                                                                   HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                                                                   KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, FREQUENTISTS_RESULTS,
                                                                                                   tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean)
                                                                #Summary
                                                                summary_text = generate_summary_text(
                                                                    summary_log, history, FREQUENTISTS_RESULTS)
                                                                #Summary figure saving 
                                                                print('Saving model')
                                                                model_save_fig(FOLDER,history,title,summary_text,metrics=[LOSS,MONITORED_METRIC],yscale='log')
                                                                model_store(FOLDER, rnd_indices_train, rnd_indices_val, rnd_indices_test,
                                                                            model, scalerX, scalerY, history, title, summary_log)
                                                                print('Saving results')
                                                                save_results(FOLDER, model, scalerX, scalerY, title, summary_text, X_train, X_test, Y_train, Y_test, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, pars=PARS, 
                                                                        labels=LABELS, plot_coverage=False, plot_distr=True, plot_corners=True, plot_tmu=FREQUENTISTS_RESULTS, batch_size=BATCH_SIZE, verbose=True)
                                                                iterator = iterator + 1
                                                                overall_progress.value = float(iterator)/(len(FILE_SAMPLES_LIST)*len(NEVENTS_TRAIN_LIST)*len(LEARNING_RATE_LIST)*len(BATCH_NORM_LIST)*len(
                                                                    LOSS_LIST)*len(HID_LAYERS_LIST)*len(ACT_FUNC_OUT_LAYER_LIST)*len(DROPOUT_RATE_LIST)*len(BATCH_SIZE_LIST)*len(REDUCE_LR_PATIENCE_LIST)*len(MIN_DELTA_LIST)*N_RUNS)
                                                                print("Processed NN:" + summary_text.replace("\n"," / "))
                                                                #del history
                                                                #del model
                                                                #gc.collect()
                                                                #K.clear_sesssion()
        end = timer()
        if CONTINUE:
            print("Processed " + str(len(FILE_SAMPLES_LIST)*len(NEVENTS_TRAIN_LIST)*len(LEARNING_RATE_LIST)*len(BATCH_NORM_LIST)*len(LOSS_LIST)*len(HID_LAYERS_LIST) *
                                     len(ACT_FUNC_OUT_LAYER_LIST)*len(DROPOUT_RATE_LIST)*len(BATCH_SIZE_LIST)*len(REDUCE_LR_PATIENCE_LIST)*len(MIN_DELTA_LIST)*N_RUNS) + " models in " + str(int(end-start)) + " s")
    return [allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, LOGPROB_THRESHOLD_INDICES_TRAIN, LOGPROB_THRESHOLD_INDICES_TEST, rnd_indices_train, rnd_indices_val, rnd_indices_test, X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, scalerX, scalerY, model, training_model, summary_log, history, training_time]
#    hist = np.histogram(data, nbins)
#    binwidth = hist[1][1]-hist[1][0]
#    #print(len(np.histogram(allsamples[:,0], 100)[0]/len(allsamples)))
#    #print(len((np.histogram(allsamples[:,0], 100)[1]+binwidth)[0:-1]))
#    arr = np.transpose(np.concatenate(
#        ([hist[0]/len(data)], [(hist[1]+binwidth)[0:-1]])))
#    arr = np.flip(arr[arr[:, 0].argsort()], axis=0)
#    q = 0
#    for i in range(nbins):
#        if q <= interval:
#            q = q + arr[i, 0]
#        else:
#            upper = np.max(arr[:i-1, 1])
#            lower = np.min(arr[:i-1, 1])
#            return [lower, upper]

#model = load_model('results_DNNLik_Bayesian/2019-08-23--12-20-42 - Data_Normal - Ndim_95 - Nevt_2E05 - Weighted_False - Loss_mae_model.h5',
#                   custom_objects={'R2_metric': R2_metric, 'Rt_metric': Rt_metric})
#scalerY = joblib.load('results_DNNLik_Bayesian/2019-08-23--12-20-42 - Data_Normal - Ndim_95 - Nevt_2E05 - Weighted_False - Loss_mae_scaler.jlib')
#
#def logprob_DNN(sample):#,model,scalerY):
#    if scalerX != 'None':
#        sample = scalerX.transform(sample)
#    if scalerY == 'None':
#        logprob = model_predict(model,np.array([sample]),batch_size=1)[0]
#    else:
#        logprob = scalerY.inverse_transform(model_predict(model,np.array([sample]),batch_size=1)[0])[0,0]
#    return logprob


def plot_corners(ilist, nbins, samp1, samp2, w1=None, w2=None, levels1=None, levels2=None, HPI_intervals1=None, HPI_intervals2=None, ranges=None, title1=None, title2=None, color1='green', color2='red', plot_title= "Params contours", legend_labels=None, figdir=None, figname=None):
    jtplot.reset()
    plt.style.use('matplotlib.mplstyle')

    start = timer()
    linewidth = 1.3
    nndim = len(ilist)
    if ilist[0] == 0:
        labels = ['$\mu$']
        for i in ilist[1:]:
            labels = np.append(labels, ['$\delta_{'+str(i+1)+'}$'])
    else:
        labels = ['$\delta_{'+str(ilist[0])+'}$']
        for i in ilist[1:]:
            labels = np.append(labels, ['$\delta_{'+str(i+1)+'}$'])
    fig, axes = plt.subplots(nndim, nndim, figsize=(3*nndim, 3*nndim))
    figure1 = corner(samp1, bins=nbins, weights=w1, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color1, plot_contours=True, smooth=True, smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color1, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels1, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color1], "alpha": 1})  # , levels=(0.393,0.68,))
    #,levels=[300],levels_lists=levels1)#,levels=[120])
    figure2 = corner(samp2, bins=nbins, weights=w2, labels=[r"%s" % s for s in labels], fig=fig, max_n_ticks=6, color=color2, plot_contours=True, smooth=True, range=ranges, smooth1d=True, plot_datapoints=True, plot_density=False, fill_contours=False, normalize1d=True,
                     hist_kwargs={'color': color2, 'linewidth': '1.5'}, label_kwargs={'fontsize': 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels2, data_kwargs={"alpha": 1}, contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][0:len(HPI_intervals1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPI_intervals1[0])]},
                     no_fill_contours=False, contourf_kwargs={"colors": ["white", "tomato", color2], "alpha": 1})  # , quantiles = (0.16, 0.84), levels=(0.393,0.68,))
    #, levels=[300],levels_lists=levels2)#,levels=[120])
    axes = np.array(figure1.axes).reshape((nndim, nndim))
    #print(get_hist(axes[0,0]))
    for i in range(nndim):
        ax = axes[i, i]
        title = ""
        #ax.axvline(value1[i], color="green",alpha=1)
        #ax.axvline(value2[i], color="red",alpha=1)
        ax.grid(True, linestyle='--', linewidth=1, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        HPI681 = HPI_intervals1[i][0][1]
        HPI951 = HPI_intervals1[i][1][1]
        HPI3s1 = HPI_intervals1[i][2][1]
        HPI682 = HPI_intervals2[i][0][1]
        HPI952 = HPI_intervals2[i][1][1]
        HPI3s2 = HPI_intervals2[i][2][1]
        hists_1d_1 = get_1d_hist(i, samp1, nbins=nbins, ranges=ranges,
                                 weights=w1, normalize1d=True)[0]  # ,intervals=HPI681)
        hists_1d_2 = get_1d_hist(i, samp2, nbins=nbins, ranges=ranges,
                                 weights=w2, normalize1d=True)[0]  # ,intervals=HPI682)
        for j in HPI3s1:
            #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor='lightgreen', alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle=":", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle=":", linewidth=linewidth)
        for j in HPI3s2:
            #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor='tomato', alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle=":", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle=":", linewidth=linewidth)
        for j in HPI951:
            #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor='lightgreen', alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
        for j in HPI952:
            #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor='tomato', alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
        for j in HPI681:
            #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor='white', alpha=1)#facecolor=(0,1,0,.5))#
            ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                       color=color1, alpha=1, linestyle="--", linewidth=linewidth)
            ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                       color=color1, alpha=1, linestyle="--", linewidth=linewidth)
            title = title+title1 + \
                ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
        title = title+"\n"
        for j in HPI682:
            #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor='white', alpha=1)#facecolor=(1,0,0,.4))#
            ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                       color=color2, alpha=1, linestyle="--", linewidth=linewidth)
            ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                       color=color2, alpha=1, linestyle="--", linewidth=linewidth)
            title = title+title2 + \
                ": ["+'{0:1.2e}'.format(j[0])+","+'{0:1.2e}'.format(j[1])+"]"
        #for j in HPI681:
        #    ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor='green', alpha=0.2)#facecolor=(0,1,0,.5))#
        #for j in HPI682:
        #    ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor='red', alpha=0.2)#facecolor=(1,0,0,.4))#
        if i == 0:
            x1, x2, _, _ = ax.axis()
            ax.set_xlim(x1*1.3, x2)
        ax.set_title(title, fontsize=12)
    for yi in range(nndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            if xi == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.grid(True, linestyle='--', linewidth=1)
            ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    fig.text(0.53,0.97,r'%s'%plot_title, fontsize=26)#, transform = ax.transAxes, ha='right',ma='left')
    colors = [color1,color2,'black', 'black', 'black']
    red_patch = matplotlib.patches.Patch(color=colors[0])#, label='The red data')
    blue_patch = matplotlib.patches.Patch(color=colors[1])#, label='The blue data')
    line1 = matplotlib.lines.Line2D([0], [0], color=colors[0], lw=12)
    line2 = matplotlib.lines.Line2D([0], [0], color=colors[1], lw=12)
    line3 = matplotlib.lines.Line2D([0], [0], color=colors[2], linewidth=3, linestyle='--')
    line4 = matplotlib.lines.Line2D([0], [0], color=colors[3], linewidth=3, linestyle='-.')
    line5 = matplotlib.lines.Line2D([0], [0], color=colors[4], linewidth=3, linestyle=':')
    lines = [line1,line2,line3,line4,line5]
    #lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='--'),matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle=':'),matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='-.')]
    #legend_labels = [r"Train set ($10^{7}$ points)",r"Test set ($10^{6}$ points)",r'$68.27\%$ HPDI', r'$95.45\%$ HPDI', r'$99.73\%$ HPDI']
    fig.legend(lines, legend_labels, fontsize=26, loc=(0.53,0.8))#, bbox_transform=ax.transAxes)
    plt.savefig(figdir + figname, dpi=50)  # ,dpi=200)
    plt.show()
    #plt.close
    end = timer()
    print("Plot done and saved in", end-start, "s.")
