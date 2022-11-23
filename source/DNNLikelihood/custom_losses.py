import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

try:
    from livelossplot import PlotLossesKerasTF as PlotLossesKeras
    from livelossplot.outputs import MatplotlibPlot
except:
    print(header_string,"\nNo module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.\n")

def mean_error(y_true, y_pred):
    """
    Bla  bla  bla
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    ME_model = K.mean(y_true-y_pred)
    return K.abs(ME_model)

def mean_percentage_error(y_true, y_pred):
    """
    Bla  bla  bla
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    MPE_model = K.mean((y_true-y_pred)/(K.sign(y_true)*K.clip(K.abs(y_true),
                                                              K.epsilon(),
                                                              None)))
    return 100. * K.abs(MPE_model)

def R2_metric(y_true, y_pred):
    """
    Bla  bla  bla
    """
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    MSE_model =  K.sum(K.square( y_true-y_pred )) 
    MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (1 - MSE_model/(MSE_baseline + K.epsilon()))

def losses():
    return {"mean_error": mean_error,
            "mean_percentage_error": mean_percentage_error,
            "R2_metric": R2_metric}


def metric_name_abbreviate(metric_name):
    name_dict = {"accuracy": "acc", "mean_error": "me", "mean_percentage_error": "mpe", "mean_squared_error": "mse",
                 "mean_absolute_error": "mae", "mean_absolute_percentage_error": "mape", "mean_squared_logarithmic_error": "msle"}
    for key in name_dict:
        metric_name = metric_name.replace(key, name_dict[key])
    return metric_name


def metric_name_unabbreviate(metric_name):
    name_dict = {"acc": "accuracy", "me": "mean_error", "mpe": "mean_percentage_error", "mse": "mean_squared_error",
                 "mae": "mean_absolute_error", "mape": "mean_absolute_percentage_error", "msle": "mean_squared_logarithmic_error"}
    for key in name_dict:
        metric_name = metric_name.replace(key, name_dict[key])
    return metric_name

## Custom Matplotlib functions for PlotLossesKeras

def custom_after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart
        Args:
            ax: matplotlib Axes
            group_name: name of metrics group (eg. Accuracy, Recall)
            x_label: label of x axis (eg. epoch, iteration, batch)
    """
    ax.set_yscale('log')
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc='center right')

#def custom_before_subplot(fig: plt.Figure, axes: np.ndarray, num_of_log_groups: int):
#    """Set matplotlib window properties
#    Args:
#        fig: matplotlib Figure
#        num_of_log_groups: number of log groups
#    """
#    clear_output(wait=True)
#    figsize_x = self.max_cols * self.cell_size[0]
#    figsize_y = ((num_of_log_groups + 1) // self.max_cols + 1) * self.cell_size[1]
#    fig.set_size_inches(figsize_x, figsize_y)
#    if num_of_log_groups < axes.size:
#        for idx, ax in enumerate(axes[-1]):
#            if idx >= (num_of_log_groups + len(self.extra_plots)) % self.max_cols:
#                ax.set_visible(False)

#def custom_after_plot(fig: plt.Figure):
#    fig.tight_layout()
#    if self.figpath is not None:
#        fig.savefig(self.figpath.format(i=self.file_idx))
#        self.file_idx += 1
#    plt.show()