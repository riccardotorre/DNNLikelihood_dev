__all__ = ["Histfactory"]

import builtins
import codecs
import json
import shutil
import sys
from datetime import datetime
from os import listdir, path, stat
from timeit import default_timer as timer

import deepdish as dd
import jsonpatch
import jsonschema
import numpy as np
import pyhf
import requests
from IPython.core.display import display
from jsonpatch import JsonPatch

from . import utils
from .likelihood import Lik
from .show_prints import Verbosity, print

header_string = "=============================="
footer_string = "------------------------------"

class Histfactory(Verbosity):
    """
    This class is a container for the the :mod:`Histfactory <histfactory>` object created from an ATLAS histfactory workspace. It allows one to import histfactory workspaces, 
    read parameters and logpdf using the |pyhf_link| package, create :class:`Lik <DNNLikelihood.Lik>` objects and save them for later use
    (see the :mod:`Likelihood <likelihood>` object documentation).
    """
#    __slots__ = "workspace_folder"
    def __init__(self,
                 workspace_folder = None,
                 name = None,
                 regions_folders_base_name = "Region",
                 bkg_files_base_name="BkgOnly",
                 patch_files_base_name ="patch",
                 output_folder = None,
                 input_file = None,
                 verbose = True):
        """
        The :class:`Histfactory <DNNLikelihood.Histfactory>` object can be initialized in two different ways, depending on the value of 
        the :argument:`input_file <Histfactory.input_file>` argument.

        - :argument:`input_file` is ``None`` (default)

            All arguments are parsed and saved in corresponding attributes. If :argument:`name` is ``None`` (default), then a name is created
            and saved in the corresponding attribute. The private method
            :meth:`Histfactory.__import <DNNLikelihood.Histfactory._Histfactory__import>` is called to import the workspace
            specified by the argument :argument:`workspace_folder`. Finally, the 
            :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>`
            method is called to save the object. 
            See the documentation of the aforementioned methods for more details.
        
        - :argument:`input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Histfactory.__load <DNNLikelihood.Histfactory._Histfactory__load>`
            Depending on the value of the argument :argument:`output_folder` the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` method behaves as follows:

                - If :argument:`output_folder` is ``None`` (default) or is equal to :argument:`input_folder`
                    
                    The attribute :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
                    is set equal to the :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>` one.

                - If :argument:`output_folder` is not ``None`` and is different than :argument:`input_folder`

                    The new :argument:`output_folder` is saved in the 
                    :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attribute and files present in the input folder are copied to the new 
                    output folder, so that all previous results are preserved in the new path.
        
        - **Arguments**

            See class :ref:`Arguments documentation <histfactory_arguments>`.

        - **Creates/updates files**

            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>` (only if :argument:`input_file` is ``None``)
            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>` (only if :argument:`input_file` is ``None``)
            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nInitialize Histfactory object.\n",show=verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.output_folder = output_folder
        self.input_file = input_file
        self.__check_define_input_files()
        if self.input_file == None:
            self.log = {timestamp: {"action": "created"}}
            self.workspace_folder = path.abspath(workspace_folder)
            self.name = name
            self.__check_define_name()
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            self.regions_folders_base_name = regions_folders_base_name
            self.bkg_files_base_name = path.splitext(bkg_files_base_name)[0]
            self.patch_files_base_name = patch_files_base_name
            subfolders = [path.join(self.workspace_folder,f) for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder,f))]
            regions = [f.replace(regions_folders_base_name, "") for f in listdir(self.workspace_folder) if path.isdir(path.join(self.workspace_folder, f))]
            self.regions = dict(zip(regions,subfolders))
            self.__import(verbose=verbose_sub)
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.__load(verbose=verbose_sub)
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    def __check_define_input_files(self,verbose=None):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one
        to set the attributes corresponding to input files and folders
        
            - :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`
            - :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>`
            - :attr:`Histfactory.input_h5_file <DNNLikelihood.Histfactory.input_h5_file>`
            - :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`
        
        depending on the initial value of the 
        :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_file == None:
            self.input_h5_file = None
            self.input_log_file = None
            self.input_folder = None
            print(header_string,"\nNo Histfactory input files and folders specified.\n", show=verbose)
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_h5_file = self.input_file+".h5"
            self.input_log_file = self.input_file+".log"
            self.input_folder = path.split(self.input_file)[0]
            print(header_string,"\nHistfactory input folder set to\n\t", self.input_folder,".\n",show=verbose)

    def __check_define_output_files(self,timestamp=None,verbose=False):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one
        to set the attributes corresponding to output files and folders

            - :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`
            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`
            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`

        depending on the initial values of the 
        :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>` and
        :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` attributes.
        It also creates the output folder if it does not exist.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.output_folder is not None:
            self.output_folder = path.abspath(self.output_folder)
            if self.input_folder is not None and self.output_folder != self.input_folder:
                utils.copy_and_save_folder(self.input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if self.input_folder is not None:
                self.output_folder = self.input_folder
            else:
                self.output_folder = path.abspath("")
        self.output_folder = utils.check_create_folder(self.output_folder)
        print(header_string,"\nHistFactory output folder set to\n\t", self.output_folder,".\n",show=verbose)
        if path.exists(path.join(self.output_folder, "histfactory_workspace")):
            self.workspace_folder = path.join(self.output_folder, "histfactory_workspace")
            print(header_string,"\nHistfactory Workspace folder\n\t", self.workspace_folder,"\nalready present in the output folder.\n",show=verbose)
        else:
            workspace_folder_old = self.workspace_folder
            shutil.copytree(self.workspace_folder, path.join(self.output_folder, "histfactory_workspace"))
            self.workspace_folder = path.join(self.output_folder, "histfactory_workspace")
            print(header_string, "\nHistfactory Workspace folder\n\t",workspace_folder_old,"\ncopied into the folder\n\t",self.workspace_folder,".\n", show=verbose)
        self.output_h5_file = path.join(self.output_folder, self.name+".h5")
        self.output_json_file = path.join(self.output_folder, self.name+".json")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        
    def __check_define_name(self):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one
        to define the :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` attribute.
        If the latter attribute is ``None``, then it is replaced with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_histfactory"``, otherwise
        the suffix "_histfactory" is appended (preventing duplication if it is already present).
        """
        if self.name == None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_histfactory"
        else:
            self.name = utils.check_add_suffix(self.name, "_histfactory")

    def __import(self, verbose=None):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one to import 
        an histfactory workspace. The method scans through the regions folders in the 
        :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`,
        determines all background and signal (patch) files and generates the dictionary attribute
        :attr:`Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`. Upon creation, all items in this
        dictionary, corresponding to all available likelihoods, have the flag ``load_model = False``.
        
        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        likelihoods_dict = {}
        for region in self.regions.keys():
            region_path = self.regions[region]
            regionfiles = [f for f in listdir(region_path) if path.isfile(path.join(region_path, f))]
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
                print(header_string,"\nLikelihoods import from folder",self.regions[region]," failed. Please check background and patch files base name.\n", show=verbose)
        for n in list(likelihoods_dict.keys()):
            likelihoods_dict[n]["name"] = self.name+"_" + str(n)+"_"+likelihoods_dict[n]["name"]+"_likelihood"
        self.likelihoods_dict = likelihoods_dict
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "imported histfactory workspace",
                               "folder": path.split(self.workspace_folder)[-1]}
        print(header_string,"\nSuccessfully imported", len(list(self.likelihoods_dict.keys())),"likelihoods from", len(list(self.regions.keys())), "regions.\n",show=verbose)

    def __load(self,verbose=None):
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one 
        to load a previously saved :class:`Histfactory <DNNLikelihood.Histfactory>` object from the files
        
            - :attr:`Histfactory.input_h5_file <DNNLikelihood.Histfactory.input_h5_file>`
            - :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`

        The method loads, with the |deepdish_link| package, the content od the 
        :attr:`Histfactory.input_h5_file <DNNLikelihood.Histfactory.input_h5_file>`
        file into a temporary dictionary, subsequently used to update the 
        :attr:`Histfactory.__dict__ <DNNLikelihood.Histfactory.__dict__>` attribute.
        The method also loads the content of the :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`
        file, assigning it to the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` attribute.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        dictionary = dd.io.load(self.input_h5_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_h5_file)[-1],
                                               path.split(self.input_log_file)[-1]]}
        print(header_string,"\nHistfactory object loaded in", str(end-start), ".\n", show=verbose)

    def import_likelihoods(self,lik_list=None, progressbar=True, verbose=None):
        """
        Imports the likelihoods ``lik_list`` (if the argument is ``None`` it imports all available likelihoods) 
        adding to the corresponding item in the :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`
        dictionary the items corresponding to the keys 
        
            - *"model"*
            - *"obs_data"*
            - *"pars_central"*
            - *"pars_bounds"*
            - *"pars_labels"*
            - *"pars_pos_poi"*
            - *"pars_pos_nuis"*
        
        and changing the value of the item corresponding to the key *"model_loaded"* to ``True``. 
        If this value was already ``True`` the likelihood is not re-imported.
        When using interactive python in Jupyter notebooks if ``progressbar = True`` the import process 
        shows a progress bar through the |ipywidgets_link| module.
        
        - **Arguments**

            - **lik_list**
            
                List of likelihoods numbers (keys of the ``Histfactory.likelihood_dict`` dictionary) to
                import. The dictionary items corresponding to the keys ``lik_list`` are changed
                while all others are left unchanged. This allows to only quickly import some
                likelihoods corresponding to interesting regions of the parameter space without having 
                to import all likelihoods in the workspace. 
                If ``lik_list = None`` all available likelihoods are imported.
                    
                    - **type**: ``list`` or ``None``
                    - **default**: ``None``

            - **progressbar**
            
                If ``True`` and if the length of ``lik_list`` is bigger than one, 
                then  a progress bar is shown.
                    
                    - **type**: ``bool``
                    - **default**: ``True`` 

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if progressbar:
            try:
                import ipywidgets as widgets
            except:
                progressbar = False
                print(header_string,"\nIf you want to show a progress bar please install the ipywidgets package.\n",show=verbose)
        if len(lik_list) == 1:
            progressbar = False
        start = timer()
        if lik_list == None:
            lik_list = list(self.likelihoods_dict.keys())
        if progressbar:
            overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={
                                                 "width": "500px", "height": "14px", 
                                                 "padding": "0px", "margin": "-5px 0px -20px 0px"})
            display(overall_progress)
            iterator = 0
        for n in lik_list:
            patch_file = path.join(self.workspace_folder,self.regions[self.likelihoods_dict[n]["signal_region"]], self.likelihoods_dict[n]["patch_file"])
            bg_only_file = path.join(self.workspace_folder,self.regions[self.likelihoods_dict[n]["signal_region"]], self.likelihoods_dict[n]["bg_only_file"])
            if self.likelihoods_dict[n]["model_loaded"]:
                print(header_string,"\nFile\n\t",patch_file, "\nis already loaded.\n",show=verbose)
            else:
                print(header_string,"\nLoading\n\t",patch_file, "\npatch file.\n",show=verbose)
                start_patch = timer()
                with open(bg_only_file) as json_file:
                    bgonly = json.load(json_file)
                with open(patch_file) as json_file:
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
                self.likelihoods_dict[n]["pars_central"] = np.array(model.config.suggested_init())
                self.likelihoods_dict[n]["pars_bounds"] = np.array(list(pars_settings.values()))
                self.likelihoods_dict[n]["pars_labels"] = list(pars_mapping.values())
                self.likelihoods_dict[n]["pars_pos_poi"] = np.array([model.config.poi_index]).flatten()
                self.likelihoods_dict[n]["pars_pos_nuis"] = np.array([i for i in range(len(self.likelihoods_dict[n]["pars_central"])) if i not in np.array([model.config.poi_index]).flatten().tolist()])
                self.likelihoods_dict[n]["model_loaded"] = True
                schema = requests.get("https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json").json()
                jsonschema.validate(instance=spec, schema=schema)
                end_patch = timer()
                print(header_string,"\nFile\n\t",patch_file, "\nprocessed in", str(end_patch-start_patch), "s.\n",show=verbose)
            if progressbar:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(lik_list))
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "imported likelihoods",
                               "likelihoods numbers": lik_list}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nImported",len(lik_list),"likelihoods in ", str(end-start), "s.\n",show=verbose)

    def save_log(self, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` attribute in the file
        :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`

        This method is called by the methods
        
        - :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` (with ``overwrite=True``)
        - :meth:`Histfactory.get_likelihood_object <DNNLikelihood.Histfactory.get_likelihood_object>` (with ``overwrite=True``)
        - :meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` (with ``overwrite=True``)
        - :meth:`Histfactory.save <DNNLikelihood.Histfactory.save>`

        - **Arguments**

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates file**

            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
        """
        verbose,verbose_sub=self.set_verbosity(verbose)
        start = timer()
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            if timestamp is None:
                timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            output_log_file = utils.generate_dump_file_name(self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nHistfactory log file\n\t", output_log_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nHistfactory log file\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nHistfactory log file dump\n\t", output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_h5(self, lik_list=None, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the :class:`Histfactory <DNNLikelihood.Histfactory>` object to the HDF5 file
        :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`.
        The object is saved by storing the content of the 
        :attr:``Histfactory.__dict__ <DNNLikelihood.Histfactory.__dict__>`` 
        attribute in a .h5 file using the |deepdish_link| package. 
        The following attributes are excluded from the saved dictionary:

            - :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`
            - :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>`
            - :attr:`Histfactory.input_h5_file <DNNLikelihood.Histfactory.input_h5_file>`
            - :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`
            - :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`
            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`
            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
            - :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`
            - :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` (saved to the file :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`)
            - :attr:`Histfactory.verbose <DNNLikelihood.Histfactory.verbose>`
        
        In order to save space, in the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` 
        the members corresponding to the keys ``lik_list`` and that have been imported are saved in the 
        ``dic["model_loaded"]=True`` mode, while the others are saved in the ``dic["model_loaded"]=False``mode. 
        See the documentation of :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` for more details.

        - **Arguments**

            - **lik_list**
            
                List of likelihoods numbers (keys of the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>``
                dictionary) that are saved in the ``dic["model_loaded"]=True`` "mode". The default value ``None`` implies that all members 
                are saved with all available information.
                    
                    - **type**: ``list`` or ``None``
                    - **default**: ``None``

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates files**

            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`
        """
        verbose, verbose_sub =self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_h5_file = self.output_h5_file
            if not overwrite:
                utils.check_rename_file(output_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_h5_file = utils.generate_dump_file_name(self.output_h5_file, timestamp=timestamp)
        if lik_list == None:
            lik_list = list(self.likelihoods_dict.keys())
        dictionary = utils.dic_minus_keys(self.__dict__, ["input_file", 
                                                          "input_folder",
                                                          "input_h5_file",
                                                          "input_log_file",
                                                          "output_folder",
                                                          "output_h5_file",
                                                          "output_log_file",
                                                          "workspace_folder",
                                                          "likelihoods_dict", 
                                                          "log", 
                                                          "verbose"])
        tmp1 = {i: self.likelihoods_dict[i] for i in lik_list}
        tmp2 = utils.dic_minus_keys(self.likelihoods_dict,lik_list)
        for key in tmp2.keys():
            tmp2[key] = utils.dic_minus_keys(tmp2[key], ["model","obs_data","pars_bounds",
                                                         "pars_central","pars_labels",
                                                         "pars_pos_nuis","pars_pos_poi",
                                                         "pars_pos_poi"])
            tmp2[key]["model_loaded"] = False
        sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        dictionary = {**dictionary, **{"likelihoods_dict": sub_dict}}
        dd.io.save(output_h5_file, dictionary)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_h5_file)[-1]}
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nHistfactory h5 file\n\t", output_h5_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nHistfactory h5 file\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nHistfactory h5 file dump\n\t", output_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_json(self, lik_list=None, timestamp=None, overwrite=False, verbose=None):
        """
        Part of the :class:`Histfactory <DNNLikelihood.Histfactory>` object is also saved to the human
        readable json file :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`.

        The object is saved by storing all json serializable attributes obtained from the
        :attr:``Histfactory.__dict__ <DNNLikelihood.Histfactory.__dict__>`` 
        attribute. The following attributes are excluded from the saved dictionary:

            - :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>`
            - :attr:`Histfactory.input_folder <DNNLikelihood.Histfactory.input_folder>`
            - :attr:`Histfactory.input_h5_file <DNNLikelihood.Histfactory.input_h5_file>`
            - :attr:`Histfactory.input_log_file <DNNLikelihood.Histfactory.input_log_file>`
            - :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>`
            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`
            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`
            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
            - :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`
            - :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>` (saved to the file :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`)
            - :attr:`Histfactory.verbose <DNNLikelihood.Histfactory.verbose>`
        
        In order to save space, in the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` 
        the members corresponding to the keys ``lik_list`` and that have been imported are saved in the 
        ``dic["model_loaded"]=True`` "mode", while the others are saved in the ``dic["model_loaded"]=False``"mode". 
        The ``likelihoods_dict["model"]`` item of the :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>``
        attribute is not saved, since it is not a json serializable object.
        See the documentation of :attr:``Histfactory.likelihood_dict <DNNLikelihood.Histfactory.likelihood_dict>`` for more details.

        - **Arguments**

            Same arguments of the :meth:`Histfactory.save_h5 <DNNLikelihood.Histfactory.save_h5>` method.

        - **Creates/updates files**

            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_json_file = self.output_json_file
            if not overwrite:
                utils.check_rename_file(output_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_json_file = utils.generate_dump_file_name(self.output_json_file, timestamp=timestamp)
        if lik_list == None:
            lik_list = list(self.likelihoods_dict.keys())
        dictionary = utils.dic_minus_keys(self.__dict__, ["input_file", 
                                                          "input_folder",
                                                          "input_h5_file",
                                                          "input_log_file",
                                                          "output_folder",
                                                          "output_h5_file",
                                                          "output_log_file",
                                                          "workspace_folder",
                                                          "likelihoods_dict", 
                                                          "log", 
                                                          "verbose"])
        for key in dictionary["regions"].keys():
            dictionary["regions"][key] = path.split(dictionary["regions"][key])[-1]
        dictionary["output_json_file"] = path.split(dictionary["output_json_file"])[-1]
        tmp1 = {i: dict(self.likelihoods_dict[i]) for i in lik_list}
        for key in tmp1.keys():
            tmp1[key]["model"] = "pyhf model not saved to json"
        tmp2 = utils.dic_minus_keys(self.likelihoods_dict,lik_list)
        for key in tmp2.keys():
            tmp2[key] = utils.dic_minus_keys(tmp2[key], ["model","obs_data","pars_bounds",
                                                         "pars_central","pars_labels",
                                                         "pars_pos_nuis","pars_pos_poi",
                                                         "pars_pos_poi"])
            tmp2[key]["model_loaded"] = False
        sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        dictionary = utils.convert_types_dict({**dictionary, **{"likelihoods_dict": sub_dict}})
        with codecs.open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved object json",
                               "file name": path.split(output_json_file)[-1]}
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nHistfactory json file\n\t", output_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nHistfactory json file\n\t", output_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\n\nHistfactory json file\n\t dump\n\t", output_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
    
    def save(self, lik_list=None, timestamp=None, overwrite=False, verbose=None):
        """
        Saves the :class:`Histfactory <DNNLikelihood.Histfactory>` object by calling the following 
        three methods:
        
            - :meth:`Histfactory.save_json <DNNLikelihood.Histfactory.save_json>`
            - :meth:`Histfactory.save_h5 <DNNLikelihood.Histfactory.save_h5>`
            - :meth:`Histfactory.save_log <DNNLikelihood.Histfactory.save_log>`

        The :class:`Histfactory <DNNLikelihood.Histfactory>` object is saved to three files: an HDF5 compressed file
        used to import back the object, a human-readable json file including json serializable
        attributes, and a log file including the content of the :attr:`Histfactory.log <DNNLikelihood.Histfactory.log>`
        attribute.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Creates/updates files**

            - :attr:`Histfactory.output_h5_file <DNNLikelihood.Histfactory.output_h5_file>`
            - :attr:`Histfactory.output_json_file <DNNLikelihood.Histfactory.output_json_file>`
            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_json(lik_list=lik_list, timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_h5(lik_list=lik_list, timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_log(timestamp=timestamp, overwrite=overwrite, verbose=verbose)

    def get_likelihood_object(self, lik_number=None, output_folder=None, verbose=None):
        """
        Generates a :class:`Lik <DNNLikelihood.Lik>` object (see its documentation for details) 
        containing all properties needed for further processing.
        If the ``lik_number`` argument corresponds to a likelihood that has not jet been imported with the
        :meth:`Histfactory.import_likelihoods <DNNLikelihood.Histfactory.import_likelihoods>` method,
        then this method is called and the likelihood is automatically imported.

        - **Arguments**

            - **lik_number**
            
                Number of the likelihood for which the :class:`Lik <DNNLikelihood.Lik>` object
                is created.
                    
                    - **type**: ``int``
                    - **default**: ``0``

            - **output_folder**
            
                Folder where the output :class:`Lik <DNNLikelihood.Lik>` object is saved.
                If specified is passed as input to the :class:`Lik <DNNLikelihood.Lik>` object, otherwise
                the :attr:`Histfactory.output_folder <DNNLikelihood.Histfactory.output_folder>` is passed
                and the :class:`Lik <DNNLikelihood.Lik>` object is saved in the same folder as the 
                :class:`Histfactory <DNNLikelihood.Histfactory>` object.
                    
                    - **type**: ``str``
                    - **default**: ``None``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            :class:`Lik <DNNLikelihood.Lik>` object.

        - **Creates files**

            - :attr:`Lik.output_h5_file <DNNLikelihood.Lik.output_h5_file>`
            - :attr:`Lik.output_json_file <DNNLikelihood.Lik.output_json_file>`
            - :attr:`Lik.output_log_file <DNNLikelihood.Lik.output_log_file>`

        - **Updates file**

            - :attr:`Histfactory.output_log_file <DNNLikelihood.Histfactory.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nCreating 'Lik' object\n", show=verbose)
        start = timer()
        if lik_number == None:
            print(header_string, "\nPlease specify the ``lik_number`` argument to create a Likelihood object.\nNo Likelihood has been created.")
            return None
        if output_folder == None:
            output_folder = self.output_folder
        lik = dict(self.likelihoods_dict[lik_number])
        if not lik["model_loaded"]:
            print(header_string,"\nModel for likelihood",lik_number,"not loaded. Attempting to load it.\n")
            self.import_likelihoods(lik_list=[lik_number], verbose=True)
            lik = dict(self.likelihoods_dict[lik_number])
        lik_obj = Lik(name=lik["name"].replace("_histfactory", ""),
                      logpdf=lik["model"].logpdf,
                      logpdf_args=[lik["obs_data"]],
                      logpdf_kwargs = None,
                      pars_central=lik["pars_central"],
                      pars_pos_poi=lik["pars_pos_poi"],
                      pars_pos_nuis=lik["pars_pos_nuis"],
                      pars_labels=lik["pars_labels"],
                      pars_bounds=lik["pars_bounds"],
                      output_folder=output_folder,
                      input_file=None,
                      verbose = self.verbose)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved likelihood object", 
                               "likelihood number": lik_number, 
                               "file names": [path.split(lik_obj.output_h5_file)[-1],
                                              path.split(lik_obj.output_log_file)[-1]]}
        print(header_string,"\nLik object for likelihood", lik_number, "created and saved in", str(end-start), "s.\n", show=verbose)
        self.save_log(overwrite=True, verbose=verbose_sub)
        return lik_obj
