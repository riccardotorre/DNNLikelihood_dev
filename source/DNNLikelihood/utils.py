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

class _FunctionWrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    Copied from emcee.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:  # pragma: no cover
            import traceback

            print("emcee: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise

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
        if isinstance(item, (list, tuple, np.ndarray)):
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
    os.makedirs(path, exist_ok=True)
    #if not os.path.exists(path):
    #    os.mkdir(path)
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

def check_delete_file(path):
    if os.path.exists(path):
        os.remove(path)

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
    totpoints = int(np.product(np.array(pars_ranges)[:, -1]))
    npars = len(pars_ranges)
    if spacing == "random":
        grid = [np.random.uniform(*par) for par in pars_ranges]
    elif spacing == "grid":
        grid = [np.linspace(*par) for par in pars_ranges]
    else:
        print("Invalid spacing argument. It should be one of: 'random' and 'grid'. Continuing with 'grid'.")
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

def define_pars_labels_auto(pars_pos_poi, pars_pos_nuis):
    pars_labels_auto = []
    i_poi = 1
    i_nuis = 1
    for i in range(len(pars_pos_poi)+len(pars_pos_nuis)):
        if i in pars_pos_poi:
            pars_labels_auto.append(r"$\theta_{%d}$" % i_poi)
            i_poi = i_poi+1
        else:
            pars_labels_auto.append(r"$\nu_{%d}$" % i_nuis)
            i_nuis = i_nuis+1
    return pars_labels_auto

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == "1":
            return r"10^{{{0}}}".format(int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def dict_structure(dic):
    excluded_keys = []
    def dict_structure_sub(dic,excluded_keys = []):
        res = {}
        for key, value in dic.items():
            if isinstance(value, dict):
                res[key], kk = dict_structure_sub(value,excluded_keys)
                excluded_keys = np.unique(excluded_keys+kk+list(value.keys())).tolist()
                for k in excluded_keys:
                    try:
                        res.pop(k)
                    except:
                        for i in res.keys():
                            if res[i] == {}:
                                res[i] = "..."
            else:
                res[key] = type(value)
                for k in excluded_keys:
                    try:
                        res.pop(k)
                    except:
                        for i in res.keys():
                            if res[i] == {}:
                                res[i] = "..."
        return res, excluded_keys
    res = {}
    for key, value in dic.items():
        if isinstance(value, dict):
            res[key], kk = dict_structure_sub(value,excluded_keys)
            excluded_keys = np.unique(excluded_keys+kk+list(value.keys())).tolist()
            for k in excluded_keys:
                try:
                    res.pop(k)
                except:
                    for i in res.keys():
                        if res[i] == {}:
                            res[i] = "..."
        else:
            res[key] = type(value)
            for k in excluded_keys:
                try:
                    res.pop(k)
                except:
                    for i in res.keys():
                        if res[i] == {}:
                            res[i] = "..."
    return res

def compare_objects(obj1,obj2,string="",verbose=False):
    print("Comparing obejects", string,".", show=verbose)
    dict1=obj1.__dict__
    dict2=obj2.__dict__
    diffs = compare_dictionaries(dict1,dict2,string,verbose=verbose)
    return diffs
    
def compare_dictionaries(dict1,dict2,string="",verbose=False):
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    print("Comparing dictionaries", string, ".", show=verbose_sub)
    diffs = []
    def intersection(lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3
    keys1 = sorted(dict1.keys())#,key=str.lower)
    keys2 = sorted(dict2.keys())#,key=str.lower)
    diff1 = list(set(keys1) - set(keys2))
    diff2 = list(set(keys2) - set(keys2))
    keys = intersection(keys1, keys2)
    if diff1 != []:
        print("DIFFERENCE: ",string,": Keys",diff1,"are in dict1 but not in dict2.\n",show=verbose)
        diffs.append([string,keys1,keys2])
    if diff2 != []:
        print("DIFFERENCE: ",string,": Keys",diff2,"are in dict2 but not in dict1.\n",show=verbose)
        diffs.append([string,keys1,keys2])
    #if diff1 == [] and diff2 == []:
    #    print(tabstr,"OK: Keys in the two dictionaries are equal.")
    for k in keys:
        prestring = string + " - " + str(k)
        print("Comparing keys", prestring, ".", show=verbose_sub)
        #print(tabstr,"Checking key",k,".")
        areobjects=False
        try:
            dict1[k].__dict__
            dict2[k].__dict__
            areobjects=True
        except:
            pass
        if areobjects:
            print("Keys", prestring, "are objects.", show=verbose_sub)
            diffs=diffs + compare_objects(dict1[k],dict2[k],prestring,verbose=verbose_sub)
        elif isinstance(dict1[k],dict) and isinstance(dict2[k],dict):
            print("Keys", prestring, "are dictionaries.", show=verbose_sub)
            diffs=diffs +compare_dictionaries(dict1[k],dict2[k],prestring,verbose=verbose_sub)
        elif isinstance(dict1[k],(np.ndarray,list,tuple)) and isinstance(dict2[k],(np.ndarray,list,tuple)):
            print("Keys", prestring, "are lists, numpy arrays, or tuple.", show=verbose_sub)
            diffs=diffs +compare_lists_arrays_tuple(dict1[k], dict2[k], prestring,verbose=verbose_sub)
        else:
            try:
                if not dict1[k] == dict2[k]:
                    print("DIFFERENCE: ",prestring,": Values are",dict1[k],"and",dict2[k],".\n",show=verbose)
                    diffs.append([prestring,dict1[k],dict2[k]])
                else:
                    print("OK: ",prestring,": Values are equal.\n",show=verbose_sub)
            except:
                print("FAILED: ",prestring,": Values could not be compared. Values are",dict1[k],"and",dict2[k],".\n",show=verbose)
                diffs.append([prestring+" - FAILED TO COMPARE",dict1[k],dict2[k]])
    return diffs

def compare_lists_arrays_tuple(list1,list2,string="",verbose=False):
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    print("Comparing list or arrays", string, ".", show=verbose)
    diffs = []
    arequal = False
    try:
        arr1 = np.array(list1)
        arr2 = np.array(list2)
        min_dtype = np.min([arr1.dtype,arr2.dtype])
        arequal = np.all(np.equal(list1,list2, dtype=min_dtype))
    except:
        pass
    if arequal:
        print("OK: ", string, ": Lists are equal.\n", show=verbose_sub)
    if not arequal:
        if len(list1)!=len(list2):
            print("DIFFERENCE: ",string,": Lists have different length.\n",show=verbose)
            diffs.append([string, list1, list2])
        else:
            for i in range(len(list1)):
                prestring = string + " - list entry " + str(i)
                print("Comparing", prestring, ".", show=verbose_sub)
                areobjects=False
                try:
                    list1[i].__dict__
                    list2[i].__dict__
                    areobjects=True
                except:
                    pass
                if areobjects:
                    print("Items", prestring, "are objects.", show=verbose_sub)
                    diffs = diffs + compare_objects(list1[i],list2[i],prestring)
                elif isinstance(list1[i],dict) and isinstance(list2[i],dict):
                    print("Items", prestring, "are dictionaries.", show=verbose_sub)
                    diffs = diffs + compare_dictionaries(list1[i],list2[i],prestring)
                elif isinstance(list1[i],(np.ndarray,list,tuple)) and isinstance(list2[i],(np.ndarray,list,tuple)):
                    print("Items", prestring,
                          "are lists, numpy arrays, or tuple.", show=verbose_sub)
                    diffs = diffs + compare_lists_arrays_tuple(list1[i], list2[i], prestring)
                else:
                    try:
                        if not list1[i] == list2[i]:
                            print("DIFFERENCE: ",prestring,": Values are",list1[i],"and",list2[i],".\n",show=verbose)
                            diffs.append([prestring,list1[i],list2[i]])
                        else:
                            print("OK: ", prestring, " Items are equal.\n", show=verbose_sub)
                    except:
                        print("FAILED: ", prestring, ": Values could not be compared. Values are",
                              list1[i], "and", list2[i], ".\n", show=verbose)
                        diffs.append([prestring,list1[i],list2[i]])
    return diffs
