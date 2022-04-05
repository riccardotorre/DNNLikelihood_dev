from distutils import dir_util
import inspect
import itertools
import numpy as np
from numpy import typing as npt
from typing import Union, List, Dict, Tuple, Optional, NewType, Type, Generic, Any
import json
import math
import os
import shutil
import re
from fpdf import FPDF
from PIL import Image
import sys
import h5py
from matplotlib import pyplot as plt
from datetime import datetime
from timeit import default_timer as timer
from tensorflow.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses

from . import show_prints
from .show_prints import print

header_string = "=============================="
footer_string = "------------------------------"

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

def generate_timestamp():
    return "datetime_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]

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


def savefig(path,**kwargs):
    """
    Function that patches the ``plt.savefig`` method for long filenames on Windows platforms.
    """
    if 'win32' in sys.platform or "win64" in sys.platform:
        plt.savefig("\\\\?\\" + path, **kwargs)
    else:
        plt.savefig(path, **kwargs)

def build_method_string_from_dict(class_name=None, method_name=None, args=None, kwargs=None):
    if class_name is not None:
        method_string = class_name+"."+method_name+"("
    else:
        method_string = method_name+"("
    if args != []:
        for arg in args:
            if type(arg) == str:
                if "(" in arg:
                    method_string = method_string+arg+", "
                else:
                    method_string = method_string+"'"+arg+"', "
            else:
                method_string = method_string+str(arg)+", "
    for key, val in kwargs.items():
        if type(val) != dict:
            if "(" in str(val):
                if "initializer" in key:
                    val = str(val).lstrip("initializers.")
                    try:
                        eval("initializers."+val)
                        method_string = method_string+key + "="+"initializers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "regularizer" in key:
                    val = str(val).lstrip("regularizer.")
                    try:
                        eval("regularizers."+val)
                        method_string = method_string+key + "="+"regularizers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "constraint" in key:
                    val = str(val).lstrip("constraints.")
                    try:
                        eval("constraints."+val)
                        method_string = method_string+key + "="+"constraints."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "callback" in key:
                    val = str(val).lstrip("callbacks.")
                    try:
                        eval("callbacks."+val)
                        method_string = method_string+key + "="+"callbacks."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "optimizers" in key:
                    val = str(val).lstrip("optimizers.")
                    try:
                        eval("optimizers."+val)
                        method_string = method_string+key + "="+"optimizers."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "losses" in key:
                    val = str(val).lstrip("losses.")
                    try:
                        eval("losses."+val)
                        method_string = method_string+key + "="+"losses."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                elif "metrics" in key:
                    val = str(val).lstrip("metrics.")
                    try:
                        eval("metrics."+val)
                        method_string = method_string+key + "="+"metrics."+val+", "
                    except:
                        method_string = method_string+key+"='"+val+"', "
                else:
                    try:
                        str_val = str(eval(str(val)))
                    except:
                        str_val = ""
                    if "<built-in function" in str_val:
                        method_string = method_string+key+"='"+val+"', "
                    else:
                        method_string = method_string+key+"="+val+", "
            elif type(val) == str:
                try:
                    str_val = str(eval(str(val)))
                except:
                    str_val = ""
                if "<built-in function" in str_val:
                    method_string = method_string+key+"='"+val+"', "
                else:
                    try:
                        eval(method_string+key+"="+val+")")
                        method_string = method_string+key+"="+val+", "
                    except Exception as e:
                        if "\\" in val:
                            method_string = method_string+key+"=r'"+val+"', "
                        else:
                            method_string = method_string+key+"='"+val+"', "
            else:
                method_string = method_string+key+"="+str(val)+", "
            #else:
            #    try:
            #        eval(str(val))
            #        method_string = method_string+key+"="+str(val)+", "
            #    except:
            #        method_string = method_string+key+"='"+str(val)+"', "
        else:
            if "initializer" in key:
                method_string = method_string+key+"=initializers."
            elif "regularizer" in key:
                method_string = method_string+key+"=regularizers."
            elif "constraint" in key:
                method_string = method_string+key+"=constraints."
            elif "callback" in key:
                method_string = method_string+key+"=callbacks."
            elif "optimizer" in key:
                method_string = method_string+key+"=optimizers."
            method_string = method_string+build_method_string_from_dict(
                class_name=None, method_name=val["name"], args=val["args"], kwargs=val["kwargs"])+", "
    method_string = method_string.rstrip(", ")+")"
    return method_string

def check_set_dict_keys(dic, keys, vals,verbose=None):
    #keys = np.array([keys]).flatten()
    #vals = np.array([vals]).flatten()
    if len(keys) != len(vals):
        raise Exception("Keys and values should have the same dimension.")
    for i in range(len(keys)):
        try:
            dic[keys[i]]
        except:
            dic[keys[i]] = vals[i]
            print("The key '"+str(keys[i])+"' was not specified and has been set to the default value '"+str(vals[i])+"'.",show=verbose)

def check_repeated_elements_at_start(lst):
    x0 = lst[0]
    n = 0
    for x in lst[1:]:
        if x == x0:
            n += 1
        else:
            return n
    return n

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
    dd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            dd[k] = convert_types_dict(v)
        elif type(v) == np.ndarray:
            dd[k] = v.tolist()
        elif type(v) == list:
            if str in [type(q) for q in flatten_list(v)]:
                dd[k] = np.array(v, dtype=object).tolist()
            else:
                dd[k] = np.array(v).tolist()
        else:
            dd[k] = np.array(v).tolist()
    return dd

def sort_dict(dictionary):
    new_dict={}
    keys_sorted = sorted(list(dictionary.keys()),key= lambda v: (str(v).upper(), str(v)[0].islower()))
    for k in keys_sorted:
        if isinstance(dictionary[k], dict):
            new_dict[k] = sort_dict(dictionary[k])
        else:
            new_dict[k] = dictionary[k]
    return new_dict

def normalize_weights(w):
    return w/np.sum(w)*len(w)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def dic_minus_keys(dictionary: Dict[Any, Any],
                   keys: List[str],
                  ) -> Dict[Any, Any]:
    if type(keys) == str:
        shallow_copy = dict(dictionary)
        try:
            del shallow_copy[keys]
        except:
            pass
        return shallow_copy
    elif type(keys) == list:
        shallow_copy = dict(dictionary)
        for i in keys:
            try:
                del shallow_copy[i]
            except:
                pass
    else:
        raise ValueError("Invalid value for 'keys' argument. The argument should be either a string or a list.")
    return shallow_copy

def string_split_at_char(s, c):
    mid = len(s)//2
    try:
        break_at = mid + min(-s[mid::-1].index(c), s[mid:].index(c), key=abs)
    except ValueError:  # if '\n' not in s
        break_at = len(s)
    firstpart, secondpart = s[:break_at + 1].rstrip(), s[break_at:].lstrip(c).rstrip()
    return [firstpart, secondpart]

def string_add_newline_at_char(s, c):
    firstpart, secondpart = string_split_at_char(s, c)
    return firstpart+"\n"+"\t"+secondpart

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
        print(header_string,"\nInvalid spacing argument. It should be one of: 'random' and 'grid'. Continuing with 'grid'.\n")
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
    verbose_sub = verbose
    if verbose < 0:
        verbose_sub = 0
    print("Comparing obejects", string, ".", show=verbose_sub)
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
            diffs=diffs + compare_objects(dict1[k],dict2[k],prestring,verbose=verbose)
        elif isinstance(dict1[k],dict) and isinstance(dict2[k],dict):
            print("Keys", prestring, "are dictionaries.", show=verbose_sub)
            diffs=diffs +compare_dictionaries(dict1[k],dict2[k],prestring,verbose=verbose)
        elif isinstance(dict1[k],(np.ndarray,list,tuple)) and isinstance(dict2[k],(np.ndarray,list,tuple)):
            print("Keys", prestring, "are lists, numpy arrays, or tuple.", show=verbose_sub)
            diffs=diffs +compare_lists_arrays_tuple(dict1[k], dict2[k], prestring,verbose=verbose)
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
    print("Comparing list or arrays", string, ".", show=verbose_sub)
    diffs = []
    arequal = False
    try:
        arr1 = np.array(list1, dtype=object)
        arr2 = np.array(list2, dtype=object)
        arequal = np.all(np.equal(arr1, arr2))
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
                    diffs = diffs + compare_objects(list1[i],list2[i],prestring,verbose=verbose)
                elif isinstance(list1[i],dict) and isinstance(list2[i],dict):
                    print("Items", prestring, "are dictionaries.", show=verbose_sub)
                    diffs = diffs + compare_dictionaries(list1[i],list2[i],prestring,verbose=verbose)
                elif isinstance(list1[i],(np.ndarray,list,tuple)) and isinstance(list2[i],(np.ndarray,list,tuple)):
                    print("Items", prestring,
                          "are lists, numpy arrays, or tuple.", show=verbose_sub)
                    diffs = diffs + compare_lists_arrays_tuple(list1[i], list2[i], prestring,verbose=verbose)
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
