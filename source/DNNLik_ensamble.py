__all__ = ["DNNLik_ensamble"]

import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

from .data_sample import Data_sample
from .DNNLik import DNNLik

#def __init__(self,
#             data_X=None,
#             data_logprob=None,
#             name=None,
#             data_sample_input_filename=None,
#             data_sample_output_filename=None,
#             import_from_file=False,
#             npoints=None,
#             shuffle=False
#             ):

class DNNLik_ensamble():
    def __init__(self, 
                 n_runs=1,
                 act_func_out_layer_list,
                 batch_norm_list,
                 batch_size_list,
                 continue_training,
                 dropout_rate_list,
                 early_stopping,
                 file_samples,
                 folder,
                 frequentist_results,
                 generate_data,
                 generate_data_on_the_fly,
                 gpu_names,
                 hid_layers_list,
                 labels,
                 learning_rate_list,
                 load_model,
                 logprob_threshold,
                 logprob_threshold_indices,
                 loss_list,
                 metrics,
                 multi_gpu,
                 n_epochs,
                 n_events_train_list,
                 pars,
                 reduce_LR_patience,
                 scale_X,
                 scale_Y,
                 test_fraction,
                 validation_fraction,
                 weight_samples_list,
                 allsamples_train='None',
                 logprob_values_train='None',
                 allsamples_test='None',
                 logprob_values_test='None',
                 idx_train='None',
                 idx_val='None',
                 idx_test='None',
                 X_train='None',
                 X_val='None',
                 X_test='None',
                 Y_train='None',
                 Y_val='None',
                 Y_test='None',
                 W_train='None',
                 W_val='None',
                 scalerX='None',
                 scalerY='None',
                 model='None',
                 training_model='None',
                 summary_log='None',
                 history='None',
                 training_time='None'
                 ):
################## Generate random model name and folder
#        self.modbasename = modbasename
#        self.folder = folder
#        self.data_sample_input_filename_train = data_sample_input_filename_train
#        self.nevents_total_train = nevents_total_train
#        self.data_sample_input_filename_test = data_sample_input_filename_test
#        self.nevents_total_test = nevents_total_test
##        self.data_idx_test_input_filename = data_idx_test_input_filename
##        self.data_idx_test_output_filename = data_idx_test_output_filename
#
#        if data_idx_test_input_filename is not None:
#            self.data_idx_test_input_filename = data_idx_test_input_filename
#            self.indices_test = DNNLik.load_data_indices_test(self.data_idx_test_input_filename)
#            self.n_events_test = len(self.indices_train)
#            self.n_events_val = len(self.indices_val)
#        else:
#            if data_idx_train_output_filename is None:
#                self.data_idx_train_output_filename = r"%s" % (folder + "/" + modname + "_samples_indices.pickle"
#            self.n_events_train = n_events_train
#            self.n_events_val = n_events_val
#            rnd_indices = np.random.choice(np.arange(self.__nevents_total_train__), size=self.n_events_train+self.n_events_val, replace=False)
#            [self.indices_train, self.indices_test] = self.train_test_split(rnd_indices, train_size=self.n_events_train, test_size=self.n_events_val)
#            self.save_data_indices_train(self.data_idx_train_output_filename,self.indices_train,self.indices_test)
#        self.indices_test = self.load_data_indices_test(self.data_idx_test_input_filename)
#        self.n_events_test = len(self.indices_test)
#
############### Start defining 
#
#        
#        self.data_train, self.data_val, self.data_test = generate_training_data()


def model_training_scan(self,
                        n_runs=1,
                        act_func_out_layer_list,
                        batch_norm_list, 
                        batch_size_list, 
                        continue_training, 
                        dropout_rate_list, 
                        early_stopping, 
                        file_samples,
                        folder, 
                        frequentist_results, 
                        generate_data, 
                        generate_data_on_the_fly, 
                        gpu_names, 
                        hid_layers_list, 
                        labels, 
                        learning_rate_list, 
                        load_model,
                        logprob_threshold, 
                        logprob_threshold_indices, 
                        loss_list, 
                        metrics, 
                        multi_gpu, 
                        n_epochs, 
                        n_events_train_list, 
                        pars, 
                        reduce_LR_patience,
                        scale_X, 
                        scale_Y, 
                        test_fraction, 
                        validation_fraction, 
                        weight_samples_list,
                        allsamples_train='None', 
                        logprob_values_train='None', 
                        allsamples_test='None', 
                        logprob_values_test='None',
                        rnd_indices_train='None', 
                        rnd_indices_val='None', 
                        rnd_indices_test='None', 
                        X_train='None', 
                        X_val='None', 
                        X_test='None',
                        Y_train='None', 
                        Y_val='None', 
                        Y_test='None', 
                        W_train='None', 
                        W_val='None', 
                        scalerX='None', 
                        scalerY='None',
                        model='None', 
                        training_model='None', 
                        summary_log='None', 
                        history='None', 
                        training_time='None'):
    start = timer()
    overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                             'width': '500px', 'height': '14px', 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
    display(overall_progress)
    iterator = 0
    if generate_data_on_the_fly:
        #if generate_data == False:
        #    print('When generate_data_OM_THE_FLY = True, generate_data flag needs also to be set to True, while it was set to False. Flag automatically changed to True.')
        #    generate_data = True
        print(
            'Training and test data will be generated on-the-fly for each run to save RAM')
        logprob_values_train = import_Y_train(file_samples, 'all')
        logprob_threshold_indices_train = np.nonzero(
            logprob_values_train >= logprob_threshold)[0]
        logprob_values_train = 'None'
        logprob_values_test = import_Y_test(file_samples, 'all')
        logprob_threshold_indices_test = np.nonzero(
            logprob_values_test >= logprob_threshold)[0]
        logprob_values_test = 'None'
    elif generate_data and generate_data_on_the_fly == False:
        if [allsamples_train, logprob_values_train] != ['None', 'None']:
            print('Training data already loaded')
        else:
            print('Loading training data')
            allsamples_train, logprob_values_train = import_XY_train(
                file_samples, 'all')
        if [allsamples_test, logprob_values_test] != ['None', 'None']:
            print('Test data already loaded')
        else:
            print('Loading test data')
            allsamples_test, logprob_values_test = import_XY_test(
                file_samples, 'all')
        logprob_threshold_indices_train = np.nonzero(
            logprob_values_train >= logprob_threshold)[0]
        logprob_threshold_indices_test = np.nonzero(
            logprob_values_test >= logprob_threshold)[0]
    elif generate_data == False and continue_training:
        continue = True
    else:
        try:
            logprob_threshold_indices_train = np.nonzero(
                logprob_values_train >= logprob_threshold)[0]
            logprob_threshold_indices_test = np.nonzero(
                logprob_values_test >= logprob_threshold)[0]
            print('Training and test data already loaded')
        except:
            print(
                "No training data available, please generate them by setting generate_data=True")
            continue = False
    if load_model != 'None':
        if continue_training == False:
            print('When loading model continue_training flag needs to be set to True, while it was set to False. Flag automatically changed to True.')
            continue_training = True
    if len(K.tensorflow_backend._get_available_gpus()) <= 1:
        multi_gpu = False
        n_gpus = len(K.tensorflow_backend._get_available_gpus())
    if multi_gpu:
        n_gpus = len(K.tensorflow_backend._get_available_gpus())
    else:
        n_gpus = 1
    if multi_gpu:
        batch_size_list = [i*n_gpus for i in batch_size_list]
    try:
        labels
        continue = True
    except:
        print("Please provide labels for the features")
        continue = False
    if continue:
        for run in range(n_runs):
            # Start loop over n_runs
            for n_events_train in n_events_train_list:
                n_events_val = int(n_events_train*validation_fraction)
                n_events_test = int(n_events_train*test_fraction)
                for weight_samples in weight_samples_list:
                    if continue_training == False or generate_data or load_model != 'None':
                        [X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, rnd_indices_train, rnd_indices_val, rnd_indices_test, scalerX, scalerY, continue] = generate_training_data(
                            generate_data, generate_data_on_the_fly, load_model, allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, file_samples, logprob_threshold_indices_train, logprob_threshold_indices_test, n_events_train, n_events_val, n_events_test, weight_samples, scale_X, scale_Y)
                    if X_train != "None":
                        continue = True
                    else:
                        print(
                            "No training data available, please change flags to ensure data generation.")
                        continue = False
                    if continue:
                        for loss in loss_list:
                            for hid_layers in hid_layers_list:
                                for act_func_out_layer in act_func_out_layer_list:
                                    for dropout_rate in dropout_rate_list:
                                        for batch_size in batch_size_list:
                                            for batch_norm in batch_norm_list:
                                                for learning_rate in learning_rate_list:
                                                    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                                                    n_dim = len(X_train[0])
                                                    if labels == 'None':
                                                        labels = [
                                                            r"$x_{%d}$" % i for i in range(n_dim)]
                                                    optimizer = optimizers.Adam(
                                                        lr=learning_rate, beta_1=0.95, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                                                    #optimizer = AdaBound(lr=learning_rate, final_lr=0.1, sgamma=1e-03, weight_decay=0., amsbound=False)
                                                    #optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
                                                    if continue_training == False:
                                                        model = model_define(
                                                            n_dim, hid_layers, dropout_rate, act_func_out_layer, batch_norm, verbose=1)
                                                        model = model_compile(
                                                            model, loss, optimizer, metrics, False)
                                                        training_model = model_compile(
                                                            model, loss, optimizer, metrics, multi_gpu)
                                                        [history, training_time] = [
                                                            {}, 0]
                                                        #training_time = 0
                                                    #Model training
                                                    if load_model != 'None':
                                                        print(
                                                            'Loading model', load_model)
                                                        model = load_model(load_model, custom_objects={
                                                                           'R2_metric': R2_metric, 'Rt_metric': Rt_metric})
                                                        model = model_compile(
                                                            model, loss, optimizer, metrics, False)
                                                        training_model = model_compile(
                                                            model, loss, optimizer, metrics, multi_gpu)
                                                        load_model = 'None'
                                                    if continue_training:
                                                        print(
                                                            'continue training of loaded model')
                                                    else:
                                                        if load_model != 'None':
                                                            print(
                                                                'continue training of loaded model')
                                                        else:
                                                            print(
                                                                'Start training of new model')
                                                    [h_run, training_time_run] = model_fit(training_model, X_train, Y_train, X_val, Y_val, scalerX, scalerY, n_epochs,
                                                                                           batch_size, sample_weights=W_train, early_stopping=early_stopping, reduce_LR_patience=reduce_LR_patience, verbose=2)
                                                    if continue_training == False:  # and load_model == 'None':
                                                        [history, training_time] = [
                                                            h_run.history, training_time_run]
                                                    else:
                                                        if load_model != 'None':
                                                            with open(load_model.replace('model.h5', 'history.json')) as json_file:
                                                                history = json.load(
                                                                    json_file)
                                                            history_full = {}
                                                            history_run = h_run.history
                                                            for key in history_run.keys():
                                                                history_full[key] = history[key] + \
                                                                    history_run[key]
                                                            [history, training_time] = [
                                                                history_full, training_time + training_time_run]
                                                            #training_time = training_time + training_time_run
                                                            del(history_full,
                                                                history_run)
                                                        else:
                                                            history_full = {}
                                                            history_run = h_run.history
                                                            for key in history_run.keys():
                                                                history_full[key] = history[key] + \
                                                                    history_run[key]
                                                            [history, training_time] = [
                                                                history_full, training_time + training_time_run]
                                                            #training_time = training_time + training_time_run
                                                            del(history_full,
                                                                history_run)
                                                    # compute predictions
                                                    [min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
                                                     quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
                                                     one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
                                                     central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test,
                                                     HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                     KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean, prediction_time] = compute_predictions(model, scalerX, scalerY, X_train, X_val, X_test, Y_train, Y_val, Y_test, loss, n_events_train, batch_size, frequentist_results)
                                                    #print('Relative error on credibility intervals validation:', prob_intervals_pred_val_rel_error[1])
                                                    print('KS test-pred_train/KS test-pred_val/KS val-pred_test/KS train-test median:', str(KS_test_pred_train_median), '/', str(
                                                        KS_test_pred_val_median), '/', str(KS_val_pred_test_median), '/', str(KS_train_test_median))
                                                    print(
                                                        'MAPE on exp (train/test/val):', mape_on_exp_train, '/', mape_on_exp_val, '/', mape_on_exp_test)
                                                    EXACT_EPOCHS = len(
                                                        history['loss'])
                                                    #Model log
                                                    summary_log = generate_summary_log(model, now, file_samples, n_dim, n_events_train, n_events_val, weight_samples, scale_X, scale_Y, loss, hid_layers, dropout_rate, early_stopping, reduce_LR_patience, act_func_out_layer, batch_norm,
                                                                                       learning_rate, batch_size, EXACT_EPOCHS, gpu_names, n_gpus, training_time, min_loss_scaled_train, min_loss_scaled_val, min_loss_scaled_test, mape_on_exp_train, mape_on_exp_val, mape_on_exp_test,
                                                                                       quantiles_train, quantiles_val, quantiles_test, quantiles_pred_train, quantiles_pred_val, quantiles_pred_test,
                                                                                       one_sised_quantiles_train, one_sised_quantiles_val, one_sised_quantiles_test, one_sised_quantiles_pred_train, one_sised_quantiles_pred_val, one_sised_quantiles_pred_test,
                                                                                       central_quantiles_train, central_quantiles_val, central_quantiles_test, central_quantiles_pred_train, central_quantiles_pred_val, central_quantiles_pred_test,
                                                                                       HPI_train, HPI_val, HPI_test, HPI_pred_train, HPI_pred_val, HPI_pred_test, one_sigma_HPI_rel_err_train, one_sigma_HPI_rel_err_val, one_sigma_HPI_rel_err_test, one_sigma_HPI_rel_err_train_test,
                                                                                       KS_test_pred_train, KS_test_pred_val, KS_val_pred_test, KS_train_test, KS_test_pred_train_median, KS_test_pred_val_median, KS_val_pred_test_median, KS_train_test_median, prediction_time, frequentist_results,
                                                                                       tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, tmu_err_mean)
                                                    #Model title
                                                    title = generate_title(
                                                        summary_log)
                                                    #Summary
                                                    summary_text = generate_summary_text(
                                                        summary_log, history, frequentist_results)
                                                    #Summary figure saving
                                                    print('Saving model')
                                                    model_save_fig(
                                                        folder, history, title, summary_text, metric='loss', yscale='log')
                                                    model_store(folder, rnd_indices_train, rnd_indices_val, rnd_indices_test,
                                                                model, scalerX, scalerY, history, title, summary_log)
                                                    print('Saving results')
                                                    save_results(folder, model, scalerX, scalerY, title, summary_text, X_train, X_test, Y_train, Y_test, tmuexact, tmuDNN, tmusample001, tmusample005, tmusample01, tmusample02, pars=pars,
                                                                 labels=labels, plot_coverage=False, plot_distr=True, plot_corners=True, plot_tmu=frequentist_results, batch_size=batch_size, verbose=True)
                                                    iterator = iterator + 1
                                                    overall_progress.value = float(iterator)/(len(n_events_train_list)*len(learning_rate_list)*len(batch_norm_list)*len(
                                                        loss_list)*len(hid_layers_list)*len(act_func_out_layer_list)*len(dropout_rate_list)*len(batch_size_list)*n_runs)
                                                    print(
                                                        "Processed NN:" + summary_text.replace("\n", " / "))
                                                    #del history
                                                    #del model
                                                    #gc.collect()
                                                    #K.clear_sesssion()
        end = timer()
        if continue:
            print("Processed " + str(len(n_events_train_list)*len(learning_rate_list)*len(batch_norm_list)*len(loss_list)*len(hid_layers_list)
                                     * len(act_func_out_layer_list)*len(dropout_rate_list)*len(batch_size_list)*n_runs) + " models in " + str(int(end-start)) + " s")
    return [allsamples_train, logprob_values_train, allsamples_test, logprob_values_test, logprob_threshold_indices, rnd_indices_train, rnd_indices_val, rnd_indices_test, X_train, X_val, X_test, Y_train, Y_val, Y_test, W_train, W_val, scalerX, scalerY, model, training_model, summary_log, history, training_time]
