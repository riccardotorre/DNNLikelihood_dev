.. _DnnLik_methods:

Methods
"""""""

.. currentmodule:: DNNLikelihood

.. automethod:: DnnLik.__init__

.. automethod:: DnnLik._DnnLik__set_resources

.. automethod:: DnnLik._DnnLik__check_define_input_files

.. automethod:: DnnLik._DnnLik__check_define_output_files

.. automethod:: DnnLik._DnnLik__check_define_name

.. automethod:: DnnLik._DnnLik__check_npoints

.. automethod:: DnnLik._DnnLik__check_define_model_data_inputs

.. automethod:: DnnLik._DnnLik__check_define_model_define_inputs

.. automethod:: DnnLik._DnnLik__check_define_model_compile_inputs

.. automethod:: DnnLik._DnnLik__check_define_model_train_inputs

.. automethod:: DnnLik._DnnLik__check_define_ensemble_folder

.. automethod:: DnnLik._DnnLik__set_seed

.. automethod:: DnnLik._DnnLik__set_dtype

.. automethod:: DnnLik._DnnLik__set_data

.. automethod:: DnnLik._DnnLik__set_pars_info

.. automethod:: DnnLik._DnnLik__set_model_hyperparameters

.. automethod:: DnnLik._DnnLik__set_tf_objects

.. automethod:: DnnLik._DnnLik__load_json_and_log

.. automethod:: DnnLik._DnnLik__load_history

.. automethod:: DnnLik._DnnLik__load_model

.. automethod:: DnnLik._DnnLik__load_scalers

.. automethod:: DnnLik._DnnLik__load_data_indices

.. automethod:: DnnLik._DnnLik__load_predictions

.. automethod:: DnnLik._DnnLik__set_optimizer

.. automethod:: DnnLik._DnnLik__set_loss

.. automethod:: DnnLik._DnnLik__set_metrics

.. automethod:: DnnLik._DnnLik__set_callbacks

.. automethod:: DnnLik._DnnLik__set_epochs_to_run

.. automethod:: DnnLik._DnnLik__set_pars_labels

.. automethod:: DnnLik.compute_sample_weights

.. automethod:: DnnLik.define_rotation

.. automethod:: DnnLik.define_scalers

.. automethod:: DnnLik.generate_train_data

.. automethod:: DnnLik.generate_test_data

.. automethod:: DnnLik.model_define

.. automethod:: DnnLik.model_compile

.. automethod:: DnnLik.model_build

.. automethod:: DnnLik.model_train

.. automethod:: DnnLik.model_predict

.. automethod:: DnnLik.model_predict_scalar

.. automethod:: DnnLik.compute_maximum_model

.. automethod:: DnnLik.compute_profiled_maximum_model

.. automethod:: DnnLik.model_evaluate

.. automethod:: DnnLik.generate_fig_base_title

.. automethod:: DnnLik.update_figures

.. automethod:: DnnLik.plot_training_history

.. automethod:: DnnLik.plot_pars_coverage

.. automethod:: DnnLik.plot_lik_distribution

.. automethod:: DnnLik.plot_corners_1samp

.. automethod:: DnnLik.plot_corners_2samp

.. automethod:: DnnLik.model_compute_predictions

.. automethod:: DnnLik.save_log

.. automethod:: DnnLik.save_data_indices

.. automethod:: DnnLik.save_model_json

.. automethod:: DnnLik.save_model_h5

.. automethod:: DnnLik.save_model_onnx

.. automethod:: DnnLik.save_history_json

.. automethod:: DnnLik.save_json

.. automethod:: DnnLik.generate_summary_text

.. automethod:: DnnLik.save_predictions_h5

.. automethod:: DnnLik.save_scalers

.. automethod:: DnnLik.save_model_graph_pdf

.. automethod:: DnnLik.save

.. automethod:: DnnLik.show_figures

.. py:method:: DNNLikelihood.DnnLik.get_available_gpus

   Method inherited from the :class:`Resources <DNNLikelihood.Resources>` object.
   See the documentation of :meth:`Resources.get_available_gpus <DNNLikelihood.Resources.get_available_gpus>`.

.. py:method:: DNNLikelihood.DnnLik.get_available_cpu

   Method inherited from the :class:`Resources <DNNLikelihood.Resources>` object.
   See the documentation of :meth:`Resources.get_available_cpu <DNNLikelihood.Resources.get_available_cpu>`.

.. py:method:: DNNLikelihood.DnnLik.set_gpus

   Method inherited from the :class:`Resources <DNNLikelihood.Resources>` object.
   See the documentation of :meth:`Resources.set_gpus <DNNLikelihood.Resources.set_gpus>`.
   
.. py:method:: DNNLikelihood.DnnLik.set_gpus_env

   Method inherited from the :class:`Resources <DNNLikelihood.Resources>` object.
   See the documentation of :meth:`Resources.set_gpus_env <DNNLikelihood.Resources.set_gpus_env>`.

.. py:method:: DNNLikelihood.DnnLik.set_verbosity

   Method inherited from the :class:`Verbosity <DNNLikelihood.Verbosity>` object.
   See the documentation of :meth:`Verbosity.set_verbosity <DNNLikelihood.Verbosity.set_verbosity>`.

.. include:: ../external_links.rst