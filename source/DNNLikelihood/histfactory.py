__all__ = ["HistFactoryFileManager",
           "HistfactoryPredictions",
           "Histfactory"]

import codecs
import json
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime
from timeit import default_timer as timer

import deepdish as dd # type: ignore
import jsonpatch # type: ignore
import jsonschema # type: ignore
import numpy as np
from numpy import typing as npt
import pyhf # type: ignore
import requests # type: ignore
from IPython.core.display import display # type: ignore
from jsonpatch import JsonPatch

from DNNLikelihood import utils
from DNNLikelihood import FileManager, Predictions, LogPDF, Figures
from DNNLikelihood import Lik, LikFileManager, LikParsManager
from DNNLikelihood import Verbosity, print

from .base import FileManager, Predictions, LogPDF, Figures

Array = Union[List, npt.NDArray]
IntBool = Union[int, bool]
StrPath = Union[str, Path]
StrBool = Union[str, bool]
LogPredDict = Dict[str,Dict[str,Any]]

header_string = "=============================="
footer_string = "------------------------------"


class HistFactoryFileManager(FileManager):
    obj_name: str = "Histfactory"

    def __init__(self,
                 name: Optional[str] = None,
                 workspace_folder: Optional[str] = None,
                 regions_folders_base_name: str = "Region",
                 bkg_files_base_name: str = "BkgOnly",
                 patch_files_base_name: str = "patch",
                 input_file: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 verbose: Optional[Union[int, bool]] = None
                 ) -> None:
        # Define self.input_file, self.output_folder, self.verbose
        super().__init__(name=name,
                         input_file=input_file,
                         output_folder=output_folder,
                         verbose=verbose)
        verbose, verbose_sub = self.set_verbosity(verbose)
        if workspace_folder is None:
            self._workspace_folder = None
        else:
            self._workspace_folder = Path(workspace_folder)
        self.regions_folders_base_name = regions_folders_base_name
        self.bkg_files_base_name = bkg_files_base_name.split(".")[0]
        self.patch_files_base_name = patch_files_base_name
        # Check workspace
        self.__check_workspace_folder(verbose=verbose_sub)

    def __check_workspace_folder(self,
                                 verbose: Optional[Union[int, bool]] = None
                                 ) -> None:
        verbose, _ = self.set_verbosity(verbose)
        self.workspace_folder = self.output_folder.joinpath("histfactory_workspace")
        if self.workspace_folder.exists():
            print(header_string, "\nHistfactory Workspace folder\n\t", str(self.workspace_folder), "\nalready present in the output folder.\n", show=verbose)
        elif self._workspace_folder is not None:
            shutil.copytree(self._workspace_folder, self.workspace_folder)
            print(header_string, "\nHistfactory Workspace folder\n\t", str(self._workspace_folder), "\ncopied into the folder\n\t", str(self.workspace_folder), ".\n", show=verbose)
        else:
            raise Exception("No workspace folder specified.")

    def __load(self,
               obj: "Histfactory",
               verbose: Optional[IntBool] = None
               ) -> None:
        verbose, _ = self.set_verbosity(verbose)
        start = timer()
        obj_dict = dd.io.load(self.input_h5_file)
        self.regions_folders_base_name = obj_dict["regions_folders_base_name"]
        self.bkg_files_base_name = obj_dict["bkg_files_base_name"]
        self.patch_files_base_name = obj_dict["patch_files_base_name"]
        with self.input_log_file.open() as json_file:
            obj.log = json.load(json_file)
        obj.__dict__.update(obj_dict)
        end = timer()
        timestamp = utils.generate_timestamp()
        obj.log[timestamp] = {"action": "loaded",
                              "files names": [self.input_h5_file.name,
                                              self.input_log_file.name]}
        print(header_string, "\nHistfactory object loaded in", str(end-start), ".\n", show=verbose)
        
    def __import_patch(self,
                       bg_only_file: Path,
                       patch_file: Path
                      ) -> List[Dict]:
        with bg_only_file.open() as json_file:
            bgonly = json.load(json_file)
        with patch_file.open() as json_file:
            patch = JsonPatch(json.load(json_file))
        return [bgonly, patch]

class HistfactoryPredictions(Predictions):
    """
    """
    def __init__(self,
                 obj_name: str,
                 figures = None,
                 verbose = None) -> None:
        super().__init__(obj_name = obj_name,
                         verbose=verbose)

class Histfactory(Verbosity):
    """
    This class is a container for the the :mod:`Histfactory <histfactory>` object created from an ATLAS histfactory workspace. It allows one to import histfactory workspaces, 
    read parameters and logpdf using the |pyhf_link| package, create :class:`Lik <DNNLikelihood.Lik>` objects and save them for later use
    (see the :mod:`Likelihood <likelihood>` object documentation).
    """

    def __init__(self,
                 file_manager: HistFactoryFileManager,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Declaration of needed types for attributes
        self.likelihoods_dict: Dict[int,Dict[str,Any]] = {}
        self.regions: Dict[str,Path]
        self.log: LogPredDict
        self.name: str
        # Initialization of parent class
        super().__init__(verbose)
        # Initialization of verbosity mode
        verbose, verbose_sub = self.set_verbosity(self.verbose)
        # Initialization of object
        timestamp = utils.generate_timestamp()
        print(header_string, "\nInitialize Histfactory object.\n", show=verbose)
        self.file_manager = file_manager
        self.predictions = HistfactoryPredictions(self.file_manager.obj_name, verbose=verbose_sub)
        self.figures = Figures(verbose=verbose_sub)
        self.regions_folders_base_name = self.file_manager.regions_folders_base_name
        self.bkg_files_base_name = self.file_manager.bkg_files_base_name
        self.patch_files_base_name = self.file_manager.patch_files_base_name
        if self.file_manager.input_file is None:
            self.log = {timestamp: {"action": "created"}}
            self.name = self.file_manager.name.name
            subfolders = [self.file_manager.workspace_folder.joinpath(f) for f in self.file_manager.workspace_folder.iterdir() if self.file_manager.workspace_folder.joinpath(f).is_dir()]
            regions = [str(f).replace(str(self.regions_folders_base_name), "") for f in self.file_manager.workspace_folder.iterdir() if self.file_manager.workspace_folder.joinpath(f).is_dir()]
            self.regions = dict(zip(regions, subfolders))
            self.likelihoods_dict = {}
            self.__build_likelihoods_dict(verbose=verbose_sub) # fills self.likelihoods_dict dictionary
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.__load(verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    @property
    def exclude_attrs(self) -> list:
        tmp = ["exclude_attrs",
               "file_manager",
               "likelihoods_dict",
               "log",
               "verbose"]
        return tmp

    def __build_likelihoods_dict(self,
                                 verbose: Optional[IntBool] = None
                                 ) -> None:
        """
        Private method used by the :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` one to import 
        an histfactory workspace. The method scans through the regions folders in the 
        :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>`,
        determines all background and signal (patch) files and generates the dictionary attribute
        :attr:`Histfactory.likelihoods_dict <DNNLikelihood.Histfactory.likelihoods_dict>`. Upon creation, all items in this
        dictionary, corresponding to all available likelihoods, have the flag ``load_model = False``.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        for region in self.regions.keys():
            region_path = self.regions[region]
            regionfiles = [f for f in region_path.iterdir() if region_path.joinpath(f).is_file()]
            bgonly_files = [x for x in regionfiles if self.bkg_files_base_name in str(x)]
            signal_patch_files = [x for x in regionfiles if self.patch_files_base_name in str(x)]
            if len(bgonly_files) > 0 and len(signal_patch_files) > 0:
                bgonly_file = bgonly_files[0]
                range_region = list(range(len(self.likelihoods_dict.keys()), len(self.likelihoods_dict.keys())+len(signal_patch_files)))
                dict_region = dict(zip(range_region, [{"signal_region": region,
                                                       "bg_only_file": bgonly_file,
                                                       "patch_file": signal_patch_file,
                                                       "name": "region_" + region + "_patch_" + signal_patch_file.stem.replace(self.patch_files_base_name+".", "").split(".")[0],
                                                       "model_loaded": False} 
                                                      for signal_patch_file in signal_patch_files]))
                self.likelihoods_dict.update(dict_region)
            else:
                print(header_string, "\nLikelihoods import from folder", self.regions[region], " failed. Please check background and patch files base name.\n", show=verbose)
        for n in list(self.likelihoods_dict.keys()):
            self.likelihoods_dict[n]["name"] = self.name+"_" + str(n)+"_"+self.likelihoods_dict[n]["name"]+"_likelihood"
        self.log[timestamp] = {"action": "imported histfactory workspace",
                               "folder": self.file_manager.workspace_folder.name}
        print(header_string, "\nSuccessfully imported", len(list(self.likelihoods_dict.keys())), "likelihoods from", len(list(self.regions.keys())), "regions.\n", show=verbose)

    def __load(self,
               verbose: Optional[IntBool] = None
               ) -> None:
        self.file_manager.__load(obj = self, verbose = verbose)

    def import_likelihoods(self,
                           lik_list: Optional[List] = None, 
                           progressbar: bool = True, 
                           verbose: Optional[Union[int, bool]] = None
                          ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        start = timer()
        iterator = 0
        overall_progress = None
        if lik_list is None:
            lik_list = list(self.likelihoods_dict.keys())
        else:
            lik_list = list(lik_list)
        if len(lik_list) == 1:
            progressbar = False
        if progressbar:
            try:
                import ipywidgets as widgets # type: ignore
                overall_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, layout={"width": "500px", "height": "14px",
                                                                                              "padding": "0px", "margin": "-5px 0px -20px 0px"})
                display(overall_progress)
            except:
                progressbar = False
                print(header_string, "\nIf you want to show a progress bar please install the ipywidgets package.\n", show=verbose)            
        for n in lik_list:
            patch_file = self.file_manager.workspace_folder.joinpath(self.regions[self.likelihoods_dict[n]["signal_region"]], self.likelihoods_dict[n]["patch_file"])
            bg_only_file = self.file_manager.workspace_folder.joinpath(self.regions[self.likelihoods_dict[n]["signal_region"]], self.likelihoods_dict[n]["bg_only_file"])
            if self.likelihoods_dict[n]["model_loaded"]:
                print(header_string, "\nFile\n\t", patch_file, "\nis already loaded.\n", show=verbose)
            else:
                print(header_string, "\nLoading\n\t", patch_file, "\npatch file.\n", show=verbose)
                start_patch = timer()
                [bgonly, patch] = self.file_manager.__import_patch(bg_only_file=bg_only_file,
                                                                   patch_file=patch_file)
                spec = jsonpatch.apply_patch(bgonly, patch)
                ws = pyhf.Workspace(spec)
                model = ws.model()
                self.likelihoods_dict[n]["model"] = model
                pars_mapping: Dict[int,str]= {}
                pars_settings: Dict[int, str] = {}
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
                self.likelihoods_dict[n]["pars_pos_nuis"] = np.array([i for i in range(len(self.likelihoods_dict[n]["pars_central"]))
                                                                     if i not in np.array([model.config.poi_index]).flatten().tolist()])
                self.likelihoods_dict[n]["model_loaded"] = True
                schema = requests.get("https://scikit-hep.org/pyhf/schemas/1.0.0/workspace.json").json()
                jsonschema.validate(instance=spec, schema=schema)
                end_patch = timer()
                print(header_string, "\nFile\n\t", patch_file, "\nprocessed in", str(end_patch-start_patch), "s.\n", show=verbose)
            if progressbar and overall_progress is not None:
                iterator = iterator + 1
                overall_progress.value = float(iterator)/(len(lik_list))
        end = timer()
        self.log[timestamp] = {"action": "imported likelihoods",
                               "likelihoods numbers": lik_list}
        self.save_log(overwrite=True, 
                      verbose=verbose_sub)
        print(header_string, "\nImported", len(lik_list), "likelihoods in ", str(end-start), "s.\n", show=verbose)

    def save_log(self,
                 overwrite: StrBool = False,
                 verbose: Optional[Union[int, bool]] = None
                ) -> None:
        """
        See HistFactoryFileManager.save_log
        """
        self.file_manager.save_log(log=self.log, 
                                   overwrite = overwrite,
                                   verbose = verbose)

    def save_object(self, 
                    lik_list: Optional[list] = None,
                    overwrite: StrBool = False,
                    verbose: Optional[Union[int, bool]] = None
                   ) -> None:
        """
        """
        if lik_list is None:
            lik_list = list(self.likelihoods_dict.keys())
        dictionary = utils.dic_minus_keys(self.__dict__, self.exclude_attrs)
        for key in dictionary["regions"].keys():
            dictionary["regions"][key] = dictionary["regions"][key].name
        tmp1 = {i: self.likelihoods_dict[i] for i in lik_list}
        tmp2 = utils.dic_minus_keys(self.likelihoods_dict, lik_list)
        for key in tmp2.keys():
            tmp2[key] = utils.dic_minus_keys(tmp2[key], ["model", "obs_data", "pars_bounds",
                                                         "pars_central", "pars_labels",
                                                         "pars_pos_nuis", "pars_pos_poi",
                                                         "pars_pos_poi"])
            tmp2[key]["model_loaded"] = False
        sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        dictionary_h5 = {**dictionary, **{"likelihoods_dict": sub_dict}}
        for key in tmp1.keys():
            tmp1[key]["model"] = "pyhf model not saved to json"
        sub_dict = {**tmp1, **tmp2}
        sub_dict = dict(sorted(sub_dict.items()))
        dictionary_json = utils.convert_types_dict({**dictionary, **{"likelihoods_dict": sub_dict}})
        kwargs: Dict[str, Any] = {"log": self.log,
                                  "overwrite": overwrite,
                                  "verbose": verbose}
        self.file_manager.save_h5(dict_to_save=dictionary_h5, **kwargs)
        self.file_manager.save_json(dict_to_save=dictionary_json, **kwargs)

    def save_predictions_json(self, 
                              overwrite: StrBool = False,
                              verbose: Optional[Union[int, bool]] = None
                             ) -> None:
        """ 
        Save predictions json
        """
        dic_pred = utils.convert_types_dict(self.predictions.__dict__)
        dic_figs = {"figures": self.figures.figures}
        dictionary = {**dic_pred,  **dic_figs}
        self.file_manager.save_json(dict_to_save=dictionary,
                                    log=self.log,
                                    overwrite = overwrite,
                                    verbose = verbose)

    def save(self,
             lik_list: Optional[List] = None,
             overwrite: StrBool = False,
             verbose: Optional[Union[int, bool]] = None
             ) -> None:
        """
        """
        verbose, _ = self.set_verbosity(verbose)
        kwargs: Dict[str, Any] = {"overwrite": overwrite,
                                  "verbose": verbose}
        self.save_object(lik_list=lik_list, **kwargs)
        self.save_predictions_json(**kwargs)
        self.save_log(**kwargs)

    def get_likelihood_object(self,
                              lik_number: Optional[int] = None,
                              output_folder: Optional[StrPath] = None,
                              verbose: Optional[Union[int, bool]] = None
                              ) -> Lik:
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
        if lik_number is None:
            raise Exception("Please specify the ``lik_number`` argument to create a Likelihood object.")
        if output_folder is None:
            output_folder = self.file_manager.output_folder
        lik = dict(self.likelihoods_dict[lik_number])
        if not lik["model_loaded"]:
            print(header_string, "\nModel for likelihood", lik_number, "not loaded. Attempting to load it.\n")
            self.import_likelihoods(lik_list=[lik_number], verbose=True)
            lik = dict(self.likelihoods_dict[lik_number])
        lik_file_manager = LikFileManager(name = lik["name"].replace("_histfactory", ""),
                                          input_file = None,
                                          output_folder = output_folder,
                                          verbose = self.verbose)
        lik_logpdf = LogPDF(logpdf=lik["model"].logpdf,
                            logpdf_args=[lik["obs_data"]],
                            logpdf_kwargs=None)
        lik_pars_manager = LikParsManager(pars_central=lik["pars_central"],
                                          pars_pos_poi = lik["pars_pos_poi"],
                                          pars_pos_nuis = lik["pars_pos_nuis"],
                                          pars_labels = lik["pars_labels"],
                                          pars_bounds = lik["pars_bounds"],
                                          logpdf = lik_logpdf,
                                          verbose = self.verbose)
        lik_obj = Lik(file_manager = lik_file_manager,
                      logpdf = lik_logpdf,
                      parameters = lik_pars_manager,
                      verbose = self.verbose)
        end = timer()
        timestamp = utils.generate_timestamp()
        self.log[timestamp] = {"action": "saved likelihood object",
                               "likelihood number": lik_number,
                               "file names": [lik_obj.file_manager.output_h5_file.name,
                                              lik_obj.file_manager.output_log_file.name]}
        print(header_string, "\nLik object for likelihood", lik_number, "created and saved in", str(end-start), "s.\n", show=verbose)
        self.save_log(overwrite=True, verbose=verbose_sub)
        return lik_obj