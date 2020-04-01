__all__ = ["Histfactory"]

import sys
import copy
import builtins
from os import listdir, path, stat
import ipywidgets as widgets
import pickle
#import cloudpickle as pickle
from datetime import datetime
import ipywidgets as widgets
import numpy as np
import json, jsonpatch, requests, jsonschema
from timeit import default_timer as timer
from jsonpatch import JsonPatch

sys.path.insert(0, '../')
import pyhf

from . import utils
from .likelihood import Likelihood

ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

class Histfactory(object):
    """
    **The HistFactory object**

    This class is a container for a ATLAS histfactory object which allows one to import histfactory workspaces, 
    read parameters and logpdf using the pyhf package, create ``likelihood`` objects (see :ref:`The Likelihood object <likelihood_class>`) 
    and save them for later use.
    """
    def __init__(self,
                 workspace_folder = None,
                 name = None,
                 regions_folders_base_name = "Region",
                 bkg_files_base_name="BkgOnly",
                 patch_files_base_name ="patch",
                 out_folder = None,
                 histfactory_input_file = None):
        """
        Instantiate the ``histfactory`` object. 
        If ``histfactory_input_file`` has the default value ``None``, the other arguments are parsed, otherwise the object is entirely
        reconstructed from the input file. The input file should be a .pickle file exported through the ``histfactory.save_histfactory()`` method.
        Method arguments: see Class arguments
        """
        self.histfactory_input_file = histfactory_input_file
        """Docstring for instance attribute spam."""
        if self.histfactory_input_file is None:
            self.workspace_folder = path.abspath(workspace_folder)
            if name is None:
                self.name = "histfactory_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self.name = name
            self.regions_folders_base_name = regions_folders_base_name
            self.bkg_files_base_name = bkg_files_base_name
            self.patch_files_base_name = patch_files_base_name
            if out_folder is None:
                out_folder = ""
            self.out_folder = path.abspath(out_folder)
            self.output_file_base_name = name+"_histfactory"
            subfolders = [path.join(self.workspace_folder,f) for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder,f))]
            regions = [f.replace(regions_folders_base_name, "") for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder, f))]
            self.regions = dict(zip(regions,subfolders))
            self.__import_histfactory()
        else:
            self.histfactory_input_file = self.histfactory_input_file.replace(".pickle","")+".pickle"
            self.__load_histfactory()

    def __import_histfactory(self):
        """
        Private method used by the ``__init__`` one to import all likelihoods in ``load_model=False`` mode.
        It scans through the regions folders and build the ``histfactory.likelihoods_dict`` dictionary adding items 
        corresponding to the keys *"signal_region"*, *"bg_only_file"*, *"patch_file"*, *"name"*, 
        and *"model_loaded = False"*.
        The method takes no arguments.
        """
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
                                                       "name": "region_" + region + "_patch_" + path.split(x)[1].replace(self.patch_files_base_name+".",""),
                                                       "model_loaded": False} for x in signal_patch_files]))
                likelihoods_dict = {**likelihoods_dict, **dict_region}
            else:
                print("Likelihoods import from folder",self.regions[region]," failed. Please check background and patch files base name.")
        for n in list(likelihoods_dict.keys()):
            likelihoods_dict[n]["name"] = self.name+"_" + str(n)+"_"+likelihoods_dict[n]["name"]+"_likelihood"
        self.likelihoods_dict = likelihoods_dict
        print("Successfully imported", len(list(self.likelihoods_dict.keys())),"likelihoods from", len(list(self.regions.keys())),"regions.")

    def __load_histfactory(self):
        """
        Private method used by the ``__init__`` one to load the ``histfactory`` object from the file ``histfactory.histfactory_input_file``.
        The method takes no arguments.
        """
        start = timer()
        in_file = self.histfactory_input_file
        pickle_in = open(in_file, 'rb')
        self.workspace_folder = pickle.load(pickle_in)
        self.name = pickle.load(pickle_in)
        self.regions_folders_base_name = pickle.load(pickle_in)
        self.bkg_files_base_name = pickle.load(pickle_in)
        self.patch_files_base_name = pickle.load(pickle_in)
        self.out_folder = pickle.load(pickle_in)
        self.output_file_base_name = pickle.load(pickle_in)
        self.regions = pickle.load(pickle_in)
        self.likelihoods_dict = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(in_file)
        end = timer()
        print('Likelihoods loaded in', str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')


    def import_histfactory(self,lik_numbers_list=None,verbose=True):
        """
        Imports the likelihoods ``lik_numbers_list`` adding to the corresponding item in the ``histfactory.likelihoods_dict`` 
        dictionary the items corresponding to the keys *"model"*, *"obs_data"*, *"pars_init"*, *"pars_bounds"*, 
        *"pars_labels"*, *"pars_pos_poi"*, *"pars_pos_nuis"*.
        When using interactive python in Jupyter notebooks the import process shows a progress bar through the widgets module.
        Args:

            - **lik_numbers_list** (list): list of likelihoods numbers (keys of the ``histfactory.likelihood_dict`` dictionary) to
                import in ``model_loaded=True`` mode. The dictionary items corresponding to the keys ``lik_numbers_list`` are filled
                while all other items are unchanged and remain in ``model_loaded=False`` mode. This allows to only quickly import some
                likelihoods corresponding to interesting regions of the parameter space without having to import all the HistFactory 
                Workspace. If ``lik_numbers_list=None`` all available likelihoods are imported in ``model_loaded=True``.
                default: ``None``
            - **verbose (bool or int): verbose mode. See :ref:`_verbose_implementation`.
                default: ``True``
        """
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        if lik_numbers_list is None:
            lik_numbers_list = list(self.likelihoods_dict.keys()) 
        overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 'width': '500px', 'height': '14px', 
                                                 'padding': '0px', 'margin': '-5px 0px -20px 0px'})
        display(overall_progress)
        iterator = 0
        for n in lik_numbers_list:
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
                schema = requests.get('https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json').json()
                jsonschema.validate(instance=spec, schema=schema)
                end_patch = timer()
                print(self.likelihoods_dict[n]["patch_file"], "processed in", str(end_patch-start_patch), "s.")
            iterator = iterator + 1
            overall_progress.value = float(iterator)/(len(lik_numbers_list))
        end = timer()
        print("Imported",len(lik_numbers_list),"likelihoods in ", str(end-start), "s.")

    def save_histfactory(self, lik_numbers_list=None, overwrite=False, verbose=True):
        """
        Saves the ``histfactory`` object to the file ``histfactory.out_folder,histfactory.output_file_base_name+".pickle"`` using pickle.
        In particular it does a picle.dump of each of the attribuses ``workspace_folder``, name regions_folders_base_name bkg_files_base_name, patch_files_base_name,
        out_folder, output_file_base_name, regions, likelihood_dict in sequence. To save space, in the likelihood_dict the members corresponding
        to the keys ``lik_numbers_list`` are saved in the ``model_loaded=True`` mode (so with full model included), while the others
        are saved in ``model_loaded=False`` mode.
        Args:

            - **lik_numbers_list** (list): list of likelihoods numbers (keys of the ``histfactory.likelihood_dict`` dictionary) that
                are saved in ``model_loaded=True`` mode. The default value ``None`` implies that all members are saved in 
                ``model_loaded=True`` mode.
                default: ``None``
            - **overwrite** (bool): flag that determines whether an existing file gets overwritten or if a new file is created. 
                If ``overwrite=True`` the ``utils.check_rename_file()`` function (see :ref:`_utils_check_rename_file`) is used  
                to append a time-stamp to the file name.
                default: ``False``
            - **verbose** (bool or int): verbose mode. See :ref:`_verbose_implementation`.
                default: ``True``
        """
        global ShowPrints
        ShowPrints = verbose
        start = timer()
        if lik_numbers_list is None:
            sub_dict = dict(self.likelihoods_dict)
        else:
            tmp1 = {i: self.likelihoods_dict[i] for i in lik_numbers_list}
            tmp2 = utils.dic_minus_keys(self.likelihoods_dict,lik_numbers_list)
            for key in tmp2.keys():
                tmp2[key] = utils.dic_minus_keys(tmp2[key], ["model", "obs_data", "pars_init", 
                                                           "pars_bounds", "pars_labels", "pars_pos_poi",
                                                           "pars_pos_poi","pars_pos_nuis"])
                tmp2[key]["model_loaded"] = False
            sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if overwrite:
            out_file = path.join(self.out_folder,self.output_file_base_name+".pickle")
        else:
            out_file = utils.check_rename_file(path.join(self.out_folder, self.output_file_base_name+".pickle"))
        pickle_out = open(out_file, 'wb')
        pickle.dump(self.workspace_folder, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.regions_folders_base_name, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.bkg_files_base_name, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.patch_files_base_name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.out_folder, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.output_file_base_name, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.regions, pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sub_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        statinfo = stat(out_file)
        end = timer()
        print('Likelihoods saved in file', out_file,"in", str(end-start),'seconds.\nFile size is ', statinfo.st_size, '.')

    def get_lik_object(self, lik_number=0):
        """
        Generates a ``likelihood`` object containing all properties needed for further processing. The logpdf method is built from
        the ``pyhf.Workspace.model.logpdf()`` pyhf method, and it takes two arguments: the array of parameters values ``x`` and
        the array of observed data ``obs_data``. With respect to the pyhf ``logpdf`` method, the logpdf in the ``likelihood`` object
        is flattened to output a float (instead of a numpy.ndarray containing a float).
        Args:

            - **lik_number** (int): the number of the likelihood for which the ``likelihood`` object is constructed.
                default: ``0``
            - **verbose (bool or int): verbose mode. See :ref:`_verbose_implementation`.
                default: ``True``

        Returns:

            - ``likelihood`` class object.
        """
        start = timer()
        lik = dict(self.likelihoods_dict[lik_number])
        if not lik["model_loaded"]:
            print("Model for likelihood",lik_number,"not loaded. Attempting to load it.")
            self.import_histfactory(lik_numbers_list=[lik_number], verbose=True)
        name = lik["name"]
        def logpdf(x,obs_data):
            return lik["model"].logpdf(x, obs_data)[0]
        logpdf_args = [lik["obs_data"]]
        pars_pos_poi = lik["pars_pos_poi"]
        pars_pos_nuis = lik["pars_pos_nuis"]
        pars_init = lik["pars_init"]
        pars_labels = lik["pars_labels"]
        pars_bounds = lik["pars_bounds"]
        out_folder = self.out_folder
        lik_obj = Likelihood(name=name,
                      logpdf=logpdf,
                      logpdf_args=logpdf_args,
                      pars_pos_poi=pars_pos_poi,
                      pars_pos_nuis=pars_pos_nuis,
                      pars_init=pars_init,
                      pars_labels=pars_labels,
                      pars_bounds=pars_bounds,
                      out_folder=out_folder,
                      lik_input_file=None)
        end = timer()
        print("likelihood object created for likelihood",lik_number,"in",str(end-start),"s.")
        return lik_obj



