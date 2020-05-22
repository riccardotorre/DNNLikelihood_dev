.. _histfactory_arguments:

Arguments
"""""""""

.. currentmodule:: Histfactory

.. argument:: workspace_folder
   :type: str

   Path (either relative to the code execution folder or absolute)
   containing the ATLAS histfactory workspace (containing the "Regions" subfolders).
   It is saved in the :attr:`Histfactory.workspace_folder <DNNLikelihood.Histfactory.workspace_folder>` attribute.

      - **type**: ``str`` or ``None``
      - **default**: ``None``   
    
.. argument:: name
   
   Name of the :class:`Histfactory <DNNLikelihood.Histfactory>` object.
   It is used to build the :attr:`Histfactory.name <DNNLikelihood.Histfactory.name>` attribute.
         
      - **type**: ``str`` or ``None``
      - **default**: ``None``   

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
     
    Path (either relative to the code execution folder or absolute) where output files are saved.
    It is used to set the :attr:`Lik.output_folder <DNNLikelihood.Sampler.output_folder>` attribute.
        
        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: input_file
         
   File name (either relative to the code execution folder or absolute, with or without extension) 
   of a saved :class:`Histfactory <DNNLikelihood.Histfactory>` object. 
   It is used to set the 
   :attr:`Histfactory.input_file <DNNLikelihood.Histfactory.input_file>` 
   attribute.
            
      - **type**: ``str`` or ``None``
      - **default**: ``None``

.. argument:: verbose

   Argument used to set the verbosity mode of the 
   :meth:`Histfactory.__init__ <DNNLikelihood.Histfactory.__init__>` 
   method and the default verbosity mode of all class methods that accept a
   ``verbose`` argument.
   See :ref:`Verbosity mode <verbosity_mode>`.

      - **type**: ``bool`` or ``int``
      - **default**: ``True``

.. include:: ../external_links.rst