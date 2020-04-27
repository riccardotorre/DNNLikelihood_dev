import inspect
import itertools
import json
import math
import os
import shutil
import re
from fpdf import FPDF
from PIL import Image
import sys
import h5py
import numpy as np
from datetime import datetime
from timeit import default_timer as timer

from . import show_prints
from .show_prints import print

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

def flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out

def append_without_duplicate(list,element):
    if element not in list:
        list.append(element)
    return list

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
    return path

def filename_without_datetime(name):
    file, extension = os.path.splitext(name)
    try:
        match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', file).group()
    except:
        match = ""
    if match is not "":
        file = file.replace(match, "")+extension
    else:
        file = file+"_"+extension

def check_rename_file(path,timestamp=None,verbose=True):
    if os.path.exists(path):
        if timestamp is None:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            now = timestamp
        print("The file", path, "already exists. Renaming the old file.",show=verbose)
        file, extension = os.path.splitext(path)
        try:
            match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', file).group()
        except:
            match = ""
        if match is not "":
            new_path = file.replace(match,now)+extension
        else:
            new_path = file+"_old_"+now+extension
        shutil.move(path, new_path)
        #print("New file name set to", path)
    #return path

def check_rename_folder(path, timestamp=None):
    if os.path.exists(path):
        if timestamp is None:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            now = timestamp
        print("The folder", path, "already exists.")
        try:
            match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', path).group()
        except:
            match = ""
        if match is not "":
            new_path = path.replace(match, now)
        else:
            new_path = path+"_"+now
        shutil.move(path, new_path)
        #print("New folder name set to", path)
    #return path

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

def check_repeated_elements_at_start(lst):
    x0 = lst[0]
    n = 0
    for x in lst[1:]:
        if x == x0:
            n += 1
        else:
            return n
    return n

def show_figures(fig_list):
    fig_list = np.array(fig_list).flatten().tolist()
    for fig in fig_list:
        try:
            os.startfile(r'%s'%fig)
            print('File', fig, 'opened.')
        except:
            print('File',fig,'not found.')

def get_spaced_elements(array, numElems=5):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out

def next_power_of_two(x):
    i = 1
    while i < x:
        i = i << 1
    return i

def closest_power_of_two(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))

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

def normalize_weights(w):
    return w/np.sum(w)*len(w)

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

def string_split_at_char(s, c):
    mid = len(s)//2
    try:
        break_at = mid + min(-s[mid::-1].index(c), s[mid:].index(c), key=abs)
    except ValueError:  # if '\n' not in s
        break_at = len(s)
    firstpart, secondpart = s[:break_at +
                              1].rstrip(), s[break_at:].lstrip(c).rstrip()
    return [firstpart, secondpart]

def string_add_newline_at_char(s, c):
    firstpart, secondpart = string_split_at_char(s, c)
    return firstpart+"\n"+"\t"+secondpart

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

def strip_suffix(s, suff):
    if s.endswith(suff):
        return s[:len(s)-len(suff)]
    return s

def check_add_suffix(s, suff):
    if s.endswith(suff):
        return s
    else:
        return s+suff

def strip_prefix(s, pref):
    if s.startswith(pref):
        return s[len(s)-len(pref):]
    return s

def check_add_prefix(s, pref):
    if s.startswith(pref):
        return s
    else:
        return pref+s
    
def get_sorted_grid(pars_ranges, spacing="grid"):
    totpoints = np.product(np.array(pars_ranges)[:, -1])
    npars = len(pars_ranges)
    if spacing == "random":
        grid = [np.random.uniform(*par) for par in pars_ranges]
    else:
        grid = [np.linspace(*par) for par in pars_ranges]
    #np.meshgrid(*grid)
    #np.vstack(np.meshgrid(*grid)).reshape(npoints**len(pars),-1).T
    #np.meshgrid(*grid)#.reshape(125,3)
    pars_vals = np.stack(np.meshgrid(*grid), axis=npars).reshape(totpoints, -1)
    q = npars-1
    for i in range(npars):
        pars_vals = pars_vals[pars_vals[:, q].argsort(kind='mergesort')]
    q = q-1
    return pars_vals

def define_generic_pars_labels(pars_pos_poi, pars_pos_nuis):
    generic_pars_labels = []
    i_poi = 1
    i_nuis = 1
    for i in range(len(pars_pos_poi)+len(pars_pos_nuis)):
        if i in pars_pos_poi:
            generic_pars_labels.append(r"$\theta_{%d}$" % i_poi)
            i_poi = i_poi+1
        else:
            generic_pars_labels.append(r"$\nu_{%d}$" % i_nuis)
            i_nuis = i_nuis+1
    return generic_pars_labels
