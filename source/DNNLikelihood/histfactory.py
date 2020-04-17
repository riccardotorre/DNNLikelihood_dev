__all__ = ["Histfactory"]

import builtins
import codecs
import json
import pickle
import sys
from datetime import datetime
from os import listdir, path, stat
from timeit import default_timer as timer

import jsonpatch
import jsonschema
import numpy as np
import pyhf
import requests
from jsonpatch import JsonPatch
from IPython.core.display import display

from . import utils
from .likelihood import Likelihood
from .show_prints import Verbosity, print


class Histfactory(Verbosity):
    """
    This class is a container for the ``Histfactory`` object created from an ATLAS histfactory workspace. It allows one to import histfactory workspaces, 
    read parameters and logpdf using the |pyhf_link| package, create :class:`Likelihood <DNNLikelihood.Likelihood>` objects and save them for later use
    (see :ref:`the Likelihood object <likelihood_object>`).

.. |pyhf_link| raw:: html
    
    <a href="https://scikit-hep.org/pyhf/"  target="_blank"> pyhf</a>
    """
#    __slots__ = "workspace_folder"
    def __init__(self,
                 workspace_folder = None,
                 name = None,
                 regions_folders_base_name = "Region",
                 bkg_files_base_name="BkgOnly",
                 patch_files_base_name ="patch",
                 output_folder = None,
                 histfactory_input_file = None,
                 verbose = True):
        """
        The :class:`Histfactory <DNNLikelihood.Histfactory>` object can be initialized in two different ways, depending on the value of 
        the :option:`histfactory_input_file` argument.

        - :option:`histfactory_input_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no name is available, then one is created. The private method
            :meth:`Histfactory.__import_histfactory <DNNLikelihood.Histfactory._Histfactory__import_histfactory>` is called to import the workspace
            at folder :option:`workspace_folder`. This method also saves the object through the
            :meth:`Histfactory.save_histfactory <DNNLikelihood.Histfactory.save_histfactory>` method. 
            See the documentation of the aforementioned methods for more details.
        
        - :option:`histfactory_input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Histfactory.__load_histfactory <DNNLikelihood.Histfactory._Histfactory__load_histfactory>`
            If the input argument :option:`output_folder` is ``None`` (default), the attribute :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
            is set from the input file, otherwise it is set to the input argument.
        
        - **Arguments**

            See class :ref:`Arguments documentation <histfactory_arguments>`.

        - **Produces file**

            - :attr:`Histfactory.histfactory_output_log_file <DNNLikelihood.Histfactory.histfactory_output_log_file>`
            - :attr:`Histfactory.histfactory_output_log_file <DNNLikelihood.Histfactory.histfactory_output_json_file>`
            - :attr:`Histfactory.histfactory_output_log_file <DNNLikelihood.Histfactory.histfactory_output_pickle_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.histfactory_input_file = histfactory_input_file
        if self.histfactory_input_file is None:
            self.histfactory_input_json_file = self.histfactory_input_file
            self.histfactory_input_pickle_file = self.histfactory_input_file
            self.histfactory_input_log_file = self.histfactory_input_file
            self.log = {timestamp: {"action": "created"}}
            self.workspace_folder = path.abspath(workspace_folder)
            self.name = name
            self.__check_define_name()
            self.regions_folders_base_name = regions_folders_base_name
            self.bkg_files_base_name = path.splitext(bkg_files_base_name)[0]
            self.patch_files_base_name = patch_files_base_name
            if output_folder is None:
                output_folder = ""
            self.output_folder = utils.check_create_folder(path.abspath(output_folder))
            self.histfactory_output_json_file = path.join(self.output_folder, self.name+".json")
            self.histfactory_output_log_file = path.join(self.output_folder, self.name+".log")
            self.histfactory_output_pickle_file = path.join(self.output_folder, self.name+".pickle")
            subfolders = [path.join(self.workspace_folder,f) for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder,f))]
            regions = [f.replace(regions_folders_base_name, "") for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder, f))]
            self.regions = dict(zip(regions,subfolders))
            self.__import_histfactory(verbose=verbose_sub)
        else:
            self.histfactory_input_file = path.abspath(path.splitext(histfactory_input_file)[0])
            self.histfactory_input_json_file = self.histfactory_input_file+".json"
            self.histfactory_input_log_file = self.histfactory_input_file+".log"
            self.histfactory_input_pickle_file = self.histfactory_input_file+".pickle"
            self.__load_histfactory(verbose=verbose_sub)
            if output_folder is not None:
                self.output_folder = path.abspath(output_folder)
                self.histfactory_output_json_file = path.join(self.output_folder, self.name+".json")
                self.histfactory_output_log_file = path.join(self.output_folder, self.name+".log")
                self.histfactory_output_pickle_file = path.join(self.output_folder, self.name+".pickle")

    def __check_define_name(self):
        """
        If :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_histfactory"``,
        otherwise it appends the suffix "_histfactory" (preventing duplication if it is already present).
        """
        if self.name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
            self.name = "model_"+timestamp+"_histfactory"
        else:
            self.name = utils.check_add_suffix(self.name, "_histfactory")

    def __import_histfactory(self, verbose=None):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one to import 
        an histfactory workspace. The method scans through the regions folders in the 
        :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`,
        determines all background and signal (patch) files and generates the attribute
        :attr:`Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`. After creation, all items in the
        dictionary, corresponding to all available likelihoods, have the flag ``load_model=False``.
        
        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
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
                                                       "name": "region_" + region + "_patch_" + path.split(x)[1].replace(self.patch_files_base_name+".","").split(".")[0],
                                                       "model_loaded": False} for x in signal_patch_files]))
                likelihoods_dict = {**likelihoods_dict, **dict_region}
            else:
                print("Likelihoods import from folder",self.regions[region]," failed. Please check background and patch files base name.", show=verbose)
        for n in list(likelihoods_dict.keys()):
            likelihoods_dict[n]["name"] = self.name+"_" + str(n)+"_"+likelihoods_dict[n]["name"]+"_likelihood"
        self.likelihoods_dict = likelihoods_dict
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "import histfactory","folder": self.workspace_folder}
        print("Successfully imported", len(list(self.likelihoods_dict.keys())),"likelihoods from", len(list(self.regions.keys())), "regions.",show=verbose)
        self.save_histfactory(overwrite=False, verbose=verbose_sub)

    def __load_histfactory(self,verbose=None):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one to import a previously saved
        :class:`Histfactory <DNNLikelihood.Histfactory>` object from the files 
        :attr:`Histfactory.histfactory_input_json_file <DNNLikelihood.Histfactory.histfactory_input_json_file>`,
        :attr:`Histfactory.histfactory_input_json_file <DNNLikelihood.Histfactory.histfactory_input_log_file>`
        and :attr:`Histfactory.histfactory_input_pickle_file <DNNLikelihood.Histfactory.histfactory_input_pickle_file>`.

        - **Arguments**

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        with open(self.histfactory_input_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.histfactory_input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        pickle_in = open(self.histfactory_input_pickle_file, "rb")
        self.likelihoods_dict = pickle.load(pickle_in)
        pickle_in.close()
        statinfo = stat(self.histfactory_input_pickle_file)
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded","file name": path.split(self.histfactory_input_json_file)[-1],"file path": self.histfactory_input_json_file}
        print("Likelihoods loaded in", str(end-start),"seconds.\nFile size is ", statinfo.st_size, ".",show=verbose)
        self.save_histfactory_log(overwrite=True, verbose=verbose_sub)

    def import_histfactory(self,lik_numbers_list=None, verbose=None):
        """
        Imports the likelihoods ``lik_numbers_list`` (if the argument is ``None`` it imports all available likelihoods) 
        adding to the corresponding item in the :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`
        dictionary the items corresponding to the keys *"model"*, *"obs_data"*, *"pars_init"*, *"pars_bounds"*, 
        *"pars_labels"*, *"pars_pos_poi"*, *"pars_pos_nuis"*, and changing the value of the item corresponding to the key
        *"model_loaded"* to ``True``. If this value was already ``True`` the likelihood is not re-imported.
        When using interactive python in Jupyter notebooks if ``verbose=2`` the import process shows a progress bar through 
        the |ipywidgets_link| module.
        
        - **Arguments**

            - **lik_numbers_list**
            
                List of likelihoods numbers (keys of the ``Histfactory.likelihood_dict`` dictionary) to
                import. The dictionary items corresponding to the keys ``lik_numbers_list`` are changed
                while all others are left unchanged. This allows to only quickly import some
                likelihoods corresponding to interesting regions of the parameter space without having to import the full workspace. 
                If ``lik_numbers_list=None`` all available likelihoods are imported.
                    
                    - **type**: ``list`` or ``None``
                    - **default**: ``None``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                If ``verbose=2`` a progress bar is shown.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

.. |ipywidgets_link| raw:: html
    
    <a href="https://ipywidgets.readthedocs.io/en/latest/"  target="_blank"> ipywidgets</a>
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if verbose == 2:
            progressbar = True
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                print("If you want to show a progress bar please install the ipywidgets package.",show=verbose)
                self.set_verbosity(verbose)
        else:
            progressbar = False
        start = timer()
        if lik_numbers_list is None:
            lik_numbers_list = list(self.likelihoods_dict.keys())
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 "width": "500px", "height": "14px", 
                                                 "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
        for n in lik_numbers_list:
            if self.likelihoods_dict[n]["model_loaded"]:
                print(self.likelihoods_dict[n]["patch_file"], "already loaded.",show=verbose)
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
                schema = requests.get("https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json").json()
                jsonschema.validate(instance=spec, schema=schema)
                end_patch = timer()
                print(self.likelihoods_dict[n]["patch_file"], "processed in", str(end_patch-start_patch), "s.",show=verbose)
            if progressbar:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(lik_numbers_list))
        end = timer()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "imported likelihoods","likelihoods numbers": lik_numbers_list}
        self.save_histfactory_log(overwrite=True, verbose=verbose_sub)
        self.set_verbosity(verbose)
        print("Imported",len(lik_numbers_list),"likelihoods in ", str(end-start), "s.",show=verbose)

    def save_histfactory_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` attribute in the file
        :attr:`Histfactory.histfactory_input_log_file <DNNLikelihood.Histfactory.histfactory_input_log_file>`

        This method is called by the methods
        
        - :meth:`Histfactory.__load_histfactory <DNNLikelihood.Histfactory._Histfactory__load_histfactory>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Histfactory.import_histfactory <DNNLikelihood.Histfactory.import_histfactory>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Histfactory.save_histfactory <DNNLikelihood.Histfactory.save_histfactory>` with ``overwrite=overwrite`` and ``verbose=verbose``

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``


            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Histfactory.histfactory_output_log_file <DNNLikelihood.Histfactory.histfactory_output_log_file>`
        """
        verbose,_=self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.histfactory_output_log_file)
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #self.log[timestamp] = {"action": "saved", "file name": path.split(self.histfactory_output_log_file)[-1], "file path": self.histfactory_output_log_file}
        dictionary = self.log
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.histfactory_output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Histfactory log file", self.histfactory_output_log_file, "saved in", str(end-start), "s.",show=verbose)

    def save_histfactory_json(self, overwrite=False, verbose=None):
        """
        ``Histfactory`` objects are saved to three files: a .json, a .log, and a .pickle, corresponding to the three attributes
        :attr:`Histfactory.histfactory_input_json_file <DNNLikelihood.Histfactory.histfactory_input_json_file>`,
        :attr:`Histfactory.histfactory_input_log_file <DNNLikelihood.Histfactory.histfactory_input_log_file>`
        and :attr:`Histfactory.histfactory_input_pickle_file <DNNLikelihood.Histfactory.histfactory_input_pickle_file>`.
        This method saves the .json file containing a dictionary with all class attributes but  
        :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`,
        :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>`,
        :attr:`Histfactory.histfactory_input_file <DNNLikelihood.Histfactory.histfactory_input_file>`,
        :attr:`Histfactory.histfactory_input_json_file <DNNLikelihood.Histfactory.histfactory_input_json_file>`
        :attr:`Histfactory.histfactory_input_log_file <DNNLikelihood.Histfactory.histfactory_input_log_file>`, and 
        :attr:`Histfactory.histfactory_input_pickle_file <DNNLikelihood.Histfactory.histfactory_input_pickle_file>`.

        - **Arguments**

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Histfactory.histfactory_output_json_file <DNNLikelihood.Histfactory.histfactory_output_json_file>`
        """
        verbose, verbose_sub=self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.histfactory_output_json_file)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved", "file name": path.split(self.histfactory_output_json_file)[-1], "file path": self.histfactory_output_json_file}
        dictionary = utils.dic_minus_keys(self.__dict__, ["log",
                                                          "likelihoods_dict",
                                                          "histfactory_input_file",
                                                          "histfactory_input_json_file",
                                                          "histfactory_input_log_file",
                                                          "histfactory_input_pickle_file"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(self.histfactory_output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        print("Histfactory json file", self.histfactory_output_json_file, "saved in", str(end-start), "s.",show=verbose)

    def save_histfactory_pickle(self, lik_numbers_list=None, overwrite=False, verbose=None):
        """
        ``Histfactory`` objects are saved to three files: a .json, a .log, and a .pickle, corresponding to the three attributes
        :attr:`Histfactory.histfactory_input_json_file <DNNLikelihood.Histfactory.histfactory_input_json_file>`,
        :attr:`Histfactory.histfactory_input_log_file <DNNLikelihood.Histfactory.histfactory_input_log_file>`
        and :attr:`Histfactory.histfactory_input_pickle_file <DNNLikelihood.Histfactory.histfactory_input_pickle_file>`.
        This method saves the dictionary 
        :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` through a dump in the .pickle file.
        
        In order to save space, in the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` 
        the members corresponding to the keys ``lik_numbers_list`` and that have been imported are saved in the 
        ``dic["model_loaded"]=True`` "mode", while the others are saved in the ``dic["model_loaded"]=False``"mode". 
        See the documentation of :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` for more details.

        - **Arguments**

            - **lik_numbers_list**
            
                List of likelihoods numbers (keys of the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>``
                dictionary) that are saved in the ``dic["model_loaded"]=True`` "mode". The default value ``None`` implies that all members 
                are saved with all available information.
                    
                    - **type**: ``list`` or ``None``
                    - **default**: ``None``

            - **overwrite**
            
                If ``True`` if a file with the same name already exists, then it gets overwritten. If ``False`` is a file with the same name
                already exists, then the old file gets renamed with the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` 
                function.
                    
                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Produces file**

            - :attr:`Histfactory.histfactory_output_pickle_file <DNNLikelihood.Histfactory.histfactory_output_pickle_file>`
        """
        verbose, verbose_sub =self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.histfactory_output_pickle_file)
        if lik_numbers_list is None:
            lik_numbers_list = list(self.likelihoods_dict.keys())
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved", "likelihoods numbers": lik_numbers_list, "file name": path.split(
            self.histfactory_output_pickle_file)[-1], "file path": self.histfactory_output_pickle_file}
        #if lik_numbers_list is None:
        #    sub_dict = dict(self.likelihoods_dict)
        #else:
        tmp1 = {i: self.likelihoods_dict[i] for i in lik_numbers_list}
        tmp2 = utils.dic_minus_keys(self.likelihoods_dict,lik_numbers_list)
        for key in tmp2.keys():
            tmp2[key] = utils.dic_minus_keys(tmp2[key], ["model", "obs_data", "pars_init", 
                                                       "pars_bounds", "pars_labels", "pars_pos_poi",
                                                       "pars_pos_poi","pars_pos_nuis"])
            tmp2[key]["model_loaded"] = False
        sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        pickle_out = open(self.histfactory_output_pickle_file, "wb")
        pickle.dump(sub_dict, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        end = timer()
        print("Histfactory pickle file", self.histfactory_output_pickle_file, "saved in", str(end-start), "s.",show=verbose)

    def save_histfactory(self, lik_numbers_list=None, overwrite=True, verbose=None):
        """
        This methos calls in order the :meth:`Histfactory.save_histfactory_pickle <DNNLikelihood.Histfactory.save_histfactory_pickle>`,
        :meth:`Histfactory.save_histfactory_json <DNNLikelihood.Histfactory.save_histfactory_json>`, and
        :meth:`Histfactory.save_histfactory_log <DNNLikelihood.Histfactory.save_histfactory_log>` methods to save the entire object.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Histfactory.histfactory_output_pickle_file <DNNLikelihood.Histfactory.histfactory_output_pickle_file>`
            - :attr:`Histfactory.histfactory_output_json_file <DNNLikelihood.Histfactory.histfactory_output_json_file>`
            - :attr:`Histfactory.histfactory_output_log_file <DNNLikelihood.Histfactory.histfactory_output_log_file>`
        """
        verbose, _ =self.set_verbosity(verbose)
        self.save_histfactory_pickle(overwrite=overwrite, lik_numbers_list=lik_numbers_list, verbose=verbose)
        self.save_histfactory_json(overwrite=overwrite, verbose=verbose)
        self.save_histfactory_log(overwrite=overwrite, verbose=verbose)

    def get_likelihood_object(self, lik_number=0, verbose=None):
        """
        Generates a :class:`Likelihood <DNNLikelihood.Likelihood>` object containing all properties needed for further processing. 
        The :meth:`Likelihood.logpdf <DNNLikelihood.Likelihood.logpdf>` method is built from the |pyhf_model_logpdf_link| method, 
        and it takes two arguments: the array of parameters values ``x`` and the array of observed data ``obs_data``. 
        With respect to the corresponding |pyhf_model_logpdf_link| method, the logpdf in the ``Likelihood`` object
        is flattened to output a ``float`` (instead of a ``numpy.ndarray`` containing a ``float``).

        - **Arguments**

            - **lik_number**
            
                Number of the likelihood for which the :class:`Likelihood <DNNLikelihood.Likelihood>` object
                is created.
                    
                    - **type**: ``int``
                    - **default**: ``0``

            - **verbose**
            
                Verbosity mode. 
                See the :ref:`Verbosity mode <verbosity_mode>` documentation for the general behavior.
                    
                    - **type**: ``bool``
                    - **default**: ``None`` 

        - **Returns**

            :class:`Likelihood <DNNLikelihood.Likelihood>` object.

        - **Produces files**

            - :attr:`Likelihood.likelihood_output_json_file <DNNLikelihood.Likelihood.likelihood_output_json_file>`
            - :attr:`Likelihood.likelihood_output_pickle_file <DNNLikelihood.Likelihood.likelihood_output_pickle_file>`
            - :attr:`Likelihood.likelihood_output_log_file <DNNLikelihood.Likelihood.likelihood_output_log_file>`

.. |pyhf_model_logpdf_link| raw:: html
    
    <a href="https://scikit-hep.org/pyhf/_generated/pyhf.pdf.Model.html?highlight=logpdf#pyhf.pdf.Model.logpdf"  target="_blank"> pyhf.Model.logpdf</a>
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        lik = dict(self.likelihoods_dict[lik_number])
        if not lik["model_loaded"]:
            print("Model for likelihood",lik_number,"not loaded. Attempting to load it.")
            self.import_histfactory(lik_numbers_list=[lik_number], verbose=True)
        name = lik["name"].replace("_histfactory","")
        def logpdf(x,obs_data):
            return lik["model"].logpdf(x, obs_data)[0]
        logpdf_args = [lik["obs_data"]]
        pars_pos_poi = lik["pars_pos_poi"]
        pars_pos_nuis = lik["pars_pos_nuis"]
        pars_init = lik["pars_init"]
        pars_labels = lik["pars_labels"]
        pars_bounds = lik["pars_bounds"]
        output_folder = self.output_folder
        lik_obj = Likelihood(name=name,
                             logpdf=logpdf,
                             logpdf_args=logpdf_args,
                             pars_pos_poi=pars_pos_poi,
                             pars_pos_nuis=pars_pos_nuis,
                             pars_init=pars_init,
                             pars_labels=pars_labels,
                             pars_bounds=pars_bounds,
                             output_folder=output_folder,
                             likelihood_input_file=None,
                             verbose = self.verbose)
        end = timer()
        #if save:
        #    lik_obj.save_likelihood(overwrite=True, verbose=verbose_sub)
        #    self.set_verbosity(verbose)
        #    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        #    self.log[timestamp] = {"action": "saved likelihood object","likelihood number": lik_number, "files": [lik_obj.likelihood_output_json_file, lik_obj.likelihood_output_pickle_file]}
        #    print("Likelihood object for likelihood",lik_number,"created and saved to files",lik_obj.likelihood_output_json_file,"and", lik_obj.likelihood_output_pickle_file, "in", str(end-start), "s.",show=verbose)
        #else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]
        self.log[timestamp] = {"action": "saved likelihood object", "likelihood number": lik_number, "file names": [path.split(
            lik_obj.likelihood_output_pickle_file)[-1], path.split(
            lik_obj.likelihood_output_json_file)[-1], path.split(
            lik_obj.likelihood_output_log_file)[-1]], "files paths": [
            lik_obj.likelihood_output_pickle_file, lik_obj.likelihood_output_json_file, lik_obj.likelihood_output_log_file]}
        print("Likelihood object for likelihood",lik_number,"created and saved to files",lik_obj.likelihood_output_json_file,"and", lik_obj.likelihood_output_pickle_file, "in", str(end-start), "s.",show=verbose)
        self.save_histfactory_log(overwrite=True, verbose=verbose_sub)
        return lik_obj
