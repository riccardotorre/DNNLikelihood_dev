__all__ = ["histfactory"]

import sys
import builtins
from os import listdir, path, stat
import ipywidgets as widgets
import pickle
import cloudpickle
from datetime import datetime
#from os.path import abspath, isdir, isfile, join
import ipywidgets as widgets
import numpy as np
import json, jsonpatch, requests, jsonschema
from timeit import default_timer as timer
from jsonpatch import JsonPatch

sys.path.insert(0, '../')
import pyhf

from . import utility
from .Lik import Lik

ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

class histfactory(object):
    """Basic class to import ATLAS HistFactory format likelihoods"""
    def __init__(self,
                 workspace_folder = None,
                 histfactory_name = None,
                 regions_folders_base_name = "Region",
                 bkg_files_base_name = "Bkg",
                 patch_files_base_name ="patch",
                 histfactory_input_file = None):
        if histfactory_input_file is None:
            self.workspace_folder = path.abspath(workspace_folder)
            if histfactory_name is None:
                self.histfactory_name = path.split(self.workspace_folder)[1]
            else:
                self.histfactory_name = histfactory_name
            self.regions_folders_base_name = regions_folders_base_name
            self.bkg_files_base_name = bkg_files_base_name
            self.patch_files_base_name = patch_files_base_name
            self.output_file_base_name = path.abspath("histfactory_"+histfactory_name)
            subfolders = [path.join(self.workspace_folder,f) for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder,f))]
            regions = [f.replace(regions_folders_base_name, "") for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder, f))]
            self.regions = dict(zip(regions,subfolders))
            self.__import_likelihoods()
        else:
            self.load_likelihoods(in_file=histfactory_input_file, verbose=True)

    def __import_likelihoods(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        likelihoods_dict = {}
        for region in self.regions.keys():
            region_path = self.regions[region]
            regionfiles = [path.join(region_path, f) for f in listdir(region_path) if path.isfile(path.join(region_path, f))]
            bgonly_file = [x for x in regionfiles if self.bkg_files_base_name in x][0]
            signal_patch_files = [x for x in regionfiles if self.patch_files_base_name in x]
            if len(bgonly_file) > 0 and len(signal_patch_files) > 0:
                range_region = list(range(len(likelihoods_dict.keys()), len(likelihoods_dict.keys())+len(signal_patch_files)))
                dict_region = dict(zip(range_region, [{"signal_region": region, 
                                                       "bg_only_file": bgonly_file, 
                                                       "patch_file": x,
                                                       "name": "region_"+ region + "_patch_" + path.split(x)[1].split(".")[1],
                                                       "model_loaded": False} for x in signal_patch_files]))
                likelihoods_dict = {**likelihoods_dict, **dict_region}
            else:
                print("Likelihoods import from folder",self.regions[region]," failed. Please check background and patch files base name.")
        self.likelihoods_dict = likelihoods_dict
        print("Successfully imported", len(list(self.likelihoods_dict.keys())),"likelihoods from", len(list(self.regions.keys())),"regions.")

    def import_likelihoods(self,lik_number_list=None,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        if lik_number_list is None:
            lik_number_list = list(self.likelihoods_dict.keys())    
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px', 
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
        display(overall_progress)
        iterator = 0
        for n in lik_number_list:
            if self.likelihoods_dict[n]["model_loaded"]:
                print(self.likelihoods_dict[n]["patch_file"], "already loaded.")
            else:
                start_patch = timer()
                with open(self.likelihoods_dict[n]["bg_only_file"]) as json_file:
                    bgonly = json.load(json_file)
                with open(self.likelihoods_dict[n]["patch_file"]) as json_file:
                    patch = JsonPatch(json.load(json_file))
                spec = jsonpatch.apply_patch(bgonly, patch)
                ws = pyhf.Workspace(spec)
                model = ws.model()
                self.likelihoods_dict[n]["model"] = model
                pars_mapping = {}
                pars_settings = {}
                ii = 0
                for k, v in model.config.par_map.items():
                    pars_int = range(model.config.npars)
                    for j in list(pars_int[v["slice"]]):
                        pars_mapping = {**pars_mapping, **{ii: k+"_"+str(j)}}
                        pars_settings = {**pars_settings, **{ii: model.config.suggested_bounds()[ii]}}
                        ii = ii+1
                obs_data = pyhf.tensorlib.astensor(ws.data(model))
                self.likelihoods_dict[n]["obs_data"] = obs_data
                self.likelihoods_dict[n]["pars_init"] = np.array(model.config.suggested_init())
                self.likelihoods_dict[n]["pars_bounds"] = np.array(list(pars_settings.values()))
                self.likelihoods_dict[n]["pars_labels"] = list(pars_mapping.values())
                self.likelihoods_dict[n]["pars_pos_poi"] = np.array([model.config.poi_index]).flatten()
                self.likelihoods_dict[n]["pars_pos_nuis"] = np.array([i for i in range(len(self.likelihoods_dict[n]["pars_init"])) if i not in np.array([model.config.poi_index]).flatten().tolist()])
                self.likelihoods_dict[n]["model_loaded"] = True
                self.likelihoods_dict[n]["logpdf"] = self.get_logpdf(n)
                schema = requests.get('https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json').json()
                jsonschema.validate(instance=spec, schema=schema)
                end_patch = timer()
                print(self.likelihoods_dict[n]["patch_file"], "processed in", str(end_patch-start_patch), "s.")
            iterator = iterator + 1
            overall_progress.value = float(iterator)/(len(lik_number_list))
        end = timer()
        print("Imported",len(lik_number_list),"likelihoods in ", str(end-start), "s.")

    def get_logpdf(self,n):
        if not self.likelihoods_dict[n]["model_loaded"]:
            print("Model for likelihood", n,"not loaded. Attempting to load it.")
            self.import_likelihoods(lik_number_list=[n], verbose=True)
        model = self.likelihoods_dict[n]["model"]
        obs_data = self.likelihoods_dict[n]["obs_data"]
        return lambda x: model.logpdf(x, obs_data)[0]

    def save_likelihoods(self, lik_number_list=None, out_file=None, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        if lik_number_list is None:
            sub_dict = dict(self.likelihoods_dict)
        else:
            tmp1 = {i: self.likelihoods_dict[i] for i in lik_number_list}
            tmp2 = utility.dic_minus_keys(self.likelihoods_dict,lik_number_list)
            for key in tmp2.keys():
                tmp2[key] = utility.dic_minus_keys(tmp2[key], ["model", "obs_data", "pars_init", 
                                                           "pars_bounds", "pars_labels", "pars_pos_poi",
                                                           "pars_pos_poi","pars_pos_nuis", "logpdf"])
                tmp2[key]["model_loaded"] = False
            sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if out_file is None:
            out_file = self.output_file_base_name+"_"+timestamp+".pickle"
        else:
            out_file = out_file.replace(".pickle", "")+".pickle"
        pickle_out = open(out_file, 'wb')
        cloudpickle.dump(self.workspace_folder, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.histfactory_name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.regions_folders_base_name, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.bkg_files_base_name, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.patch_files_base_name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.output_file_base_name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(self.regions, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        cloudpickle.dump(sub_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihoods saved in file', out_file,"in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def load_likelihoods(self, in_file=None, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        if in_file is None:
            print("Please specify a file to import.")
            return
        pickle_in = open(in_file, 'rb')
        self.workspace_folder = pickle.load(pickle_in)
        self.histfactory_name = pickle.load(pickle_in)
        self.regions_folders_base_name = pickle.load(pickle_in)
        self.bkg_files_base_name = pickle.load(pickle_in)
        self.patch_files_base_name = pickle.load(pickle_in)
        self.output_file_base_name = pickle.load(pickle_in)
        self.regions = pickle.load(pickle_in)
        self.likelihoods_dict = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(in_file)
        end = timer()
        print('Likelihoods loaded in', str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def get_lik_object(self, lik_number=0):
        lik = self.likelihoods_dict[lik_number]
        if not lik["model_loaded"]:
            print("Model for likelihood",lik_number,"not loaded. Attempting to load it.")
            self.import_likelihoods(lik_number_list=[lik_number], verbose=True)
        lik_obj = Lik(lik_name = lik["name"],
                      logpdf=lik["logpdf"],
                      pars_pos_poi = lik["pars_pos_poi"],
                      pars_pos_nuis = lik["pars_pos_nuis"],
                      pars_init=lik["pars_init"],
                      pars_labels=lik["pars_labels"],
                      pars_bounds=lik["pars_bounds"],
                      lik_input_file=None)
        return lik_obj


