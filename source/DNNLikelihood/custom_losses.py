import tensorflow as tf
from tensorflow.keras import backend as K

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
