import inspect
import itertools
import json
import math
import os
from fpdf import FPDF
from PIL import Image
import sys
import h5py
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

#class InputError(Exception):
#    """Base class for data error exceptions"""
#    pass#

#class DataError(Exception):
#    """Base class for data error exceptions"""
#    pass#

#class MissingModule(Exception):
#    """Base class for missing package exceptions"""
#    pass

#def flatten_list(l):
#    l = [item for sublist in l for item in sublist]
#    return l

def flatten_list(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out

def make_pdf_from_img(img):
    """Make pdf from image
    Used to circumvent bud in plot_model which does not allow to export pdf"""
    img_pdf = os.path.splitext(img)[0]+".pdf"
    cover = Image.open(img)
    width, height = cover.size
    pdf = FPDF(unit = "pt", format = [width, height])
    pdf.add_page()
    pdf.image(img, 0, 0)
    pdf.output(img_pdf, "F")

def chunks(lst, n):
    """Return list of chunks from lst."""
    res = []
    for i in range(0, len(lst), n):
        res.append(lst[i:i + n])
    return res

def check_create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        #print("Folder",path,"has been created.")

def check_rename_file(path):
    if os.path.exists(path):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("The file", path, "already exists.")
        file, extension = os.path.splitext(path)
        path = file+"_"+now+extension
        print("New file name set to", path)
    return path

def check_rename_folder(path):
    if os.path.exists(path):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("The folder", path, "already exists.")
        path = path+"_"+now
        print("New folder name set to", path)
    return path

def save_samples(allsamples, logpdf_values, data_sample_filename, name):
    start = timer()
    data_sample_filename = check_rename_file(data_sample_filename)
    data_sample_shape = np.shape(allsamples)
    #data_sample_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #data_sample_name = name+"_"+data_sample_timestamp
    h5_out = h5py.File(data_sample_filename, "w")
    grp = h5_out.create_group(name)
    grp["shape"] = data_sample_shape
    grp["allsamples"] = allsamples
    grp["logpdf_values"] = logpdf_values
    h5_out.close()
    statinfo = os.stat(data_sample_filename)
    end = timer()
    print("File saved in", end-start,"seconds.\nFile size is", statinfo.st_size, ".")

#def set_param(obj_name, par_name):
#    if eval(par_name) is None:
#        exec("%s = %s" % (par_name, obj_name+"."+par_name))
#    else:
#        setattr(eval(obj_name), par_name, eval(par_name))
#    return eval(par_name)

def check_repeated_elements_at_start(list):
    x0 = list[0]
    n = 0
    for x in list[1:]:
        if x == x0:
            n += 1
        else:
            return n
    return n

def get_spaced_elements(array, numElems=5):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out

def next_power_of_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def convert_types_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            convert_types_dict(v)
        elif type(v) is np.ndarray:
            d[k] = v.tolist()
        elif type(v) is list:
            if str in [type(q) for q in flatten_list(v)]:
                d[k] = np.array(v, dtype=object).tolist()
            else:
                d[k] = np.array(v).tolist()
        else:
            d[k] = np.array(v).tolist()
    return d

def sort_dict(d):
    return json.loads(json.dumps(d,sort_keys=True))

#def convert_types_dic(dic):
#    new_dic = {}
#    for key in list(dic.keys()):
#        if type(dic[key]) == np.ndarray:
#            new_dic[key] == dic[key].tolist()
#        elif type(dic[key]) == list:
#            #"float" in str(
#            if "numpy.float" in str(type(dic[key][0])):
#                new_dic[key] = list(map(float, dic[key]))
#            if "numpy.int" in str(type(dic[key][0])):
#                new_dic[key] = list(map(int, dic[key]))
#            else:
#                new_dic[key] = dic[key]
#        elif "numpy.float" in str(type(dic[key])):
#            new_dic[key] = float(dic[key])
#        elif "numpy.int" in str(type(dic[key])):
#            new_dic[key] = int(dic[key])
#        else:
#            new_dic[key] = dic[key]
#    return new_dic

def normalize_weights(x):
    return x/np.sum(x)*len(x)

def closest_power_of_two(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def dic_minus_keys(dictionary, keys):
    if type(keys) is str:
        shallow_copy = dict(dictionary)
        try:
            del shallow_copy[keys]
        except:
            pass
        return shallow_copy
    elif type(keys) is list:
        shallow_copy = dict(dictionary)
        for i in keys:
            try:
                del shallow_copy[i]
            except:
                pass
        return shallow_copy

def metric_name_abbreviate(name):
    name_dict = {"accuracy": "acc", "mean_error": "me", "mean_percentage_error": "mpe", "mean_squared_error": "mse",
                 "mean_absolute_error": "mae", "mean_absolute_percentage_error": "mape", "mean_squared_logarithmic_error": "msle"}
    for key in name_dict:
        name = name.replace(key, name_dict[key])
    return name

def metric_name_unabbreviate(name):
    name_dict = {"acc": "accuracy", "me": "mean_error", "mpe": "mean_percentage_error", "mse": "mean_squared_error",
                 "mae": "mean_absolute_error", "mape": "mean_absolute_percentage_error", "msle": "mean_squared_logarithmic_error"}
    for key in name_dict:
        name = name.replace(key, name_dict[key])
    return name
