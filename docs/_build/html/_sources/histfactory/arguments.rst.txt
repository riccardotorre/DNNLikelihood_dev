.. _histfactory_arguments:

Arguments
"""""""""

.. currentmodule:: Histfactory

.. argument:: workspace_folder

   Path (either relative to the code execution folder or absolute)
   containing the ATLAS histfactory workspace (the one containing the "Regions" subfolders).
   Unless it is already there, the folder is copied into the 
   :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>` folder
   and renamed "hitfactory_workspace".

      - **type**: ``str`` or ``None``
      - **default**: ``None``   
    
.. argument:: name
   
   See :argument:`name <common_classes_arguments.name>`.

.. argument:: regions_folders_base_name

   Common folder name of the "Region" folders contained in the 
   :argument:`workspace_folder` (these folders are usually named "RegionA", "RegionB", etc.).
   It is used to set the 
   :attr:`Histfactory.regions_folders_base_name <DNNLikelihood.Histfactory.regions_folders_base_name>` 
   attribute.
            
      - **type**: ``str``
      - **default**: ``Region``  

.. argument:: bkg_files_base_name

   Name (with or without the .json extension) of the "background" json files 
   in the "Region" folders (e.g. "BkgOnly").
   It is used to set the 
   :attr:`Histfactory.bkg_files_base_name <DNNLikelihood.Histfactory.bkg_files_base_name>` 
   attribute.
            
      - **type**: ``str``
      - **default**: ``BkgOnly``

.. argument:: patch_files_base_name
      
   Base name (without the .json extension) of the "signal" patch
   json files in the "Region" folders (e.g. "patch").
   It is used to set the 
   :attr:`Histfactory.patch_files_base_name <DNNLikelihood.Histfactory.patch_files_base_name>` 
   attribute.
            
      - **type**: ``str``
      - **default**: ``patch`` 

.. argument:: output_folder
     
   See :argument:`output_folder <common_classes_arguments.output_folder>`.

.. argument:: input_file

    See :argument:`input_file <common_classes_arguments.input_file>`.
         
.. argument:: verbose

    See :argument:`verbose <common_classes_arguments.verbose>`.

.. include:: ../external_links.rst