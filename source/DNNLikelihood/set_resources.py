import os
import numpy as np
import builtins
#from multiprocessing import cpu_count
from tensorflow.python.client import device_lib
import cpuinfo

from . import show_prints
from .show_prints import print

#https://stackoverflow.com/questions/42322698/tensorflow-keras-multi-threaded-model-fitting?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

#import subprocess
#def get_available_gpus():
#    if os.name == 'nt':
#        try:
#            os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA Corporation\NVSMI'
#            available_gpus = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
#        except:
#            print("nvidia-smi.exe not found it its system folder 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'. Please modify the PATH accordingly.")
#            available_gpus = []
#    else:
#        available_gpus = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
#    #available_gpus_current = K.tensorflow_backend._get_available_gpus()
#    print(str(len(available_gpus))+" GPUs available in current environment")
#    if len(available_gpus) >0:
#        print(available_gpus)
#    return available_gpus
#def get_available_cpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def get_available_gpus(verbose=True):
    global ShowPrints
    ShowPrints = verbose
    local_device_protos = device_lib.list_local_devices()
    available_gpus = [[x.name, x.physical_device_desc]
                      for x in local_device_protos if x.device_type == 'GPU']
    print(str(len(available_gpus))+" GPUs available")
    return available_gpus

def get_available_cpu(verbose=True):
    global ShowPrints
    ShowPrints = verbose
    local_device_protos = device_lib.list_local_devices()
    id = [x.name for x in local_device_protos if x.device_type == 'CPU'][0]
    local_device_protos = cpuinfo.get_cpu_info()
    brand = local_device_protos['brand']
    cores_count = local_device_protos['count']
    available_cpu = [id, brand, cores_count]
    print(str(cores_count)+" CPU cores available")
    return available_cpu

def set_gpus(gpus_list="all", verbose=True):
    global ShowPrints
    ShowPrints = verbose
    available_gpus = get_available_gpus(verbose=False)
    ShowPrints = verbose
    if len(available_gpus) == 0:
        print('No available GPUs. Running with CPU support only.')
        return available_gpus
    if gpus_list is None:
        print("No GPUs have been set. Running with CPU support only.")
        return []
    elif gpus_list is "all":
        gpus_list = list(range(len(available_gpus)))
    else:
        if np.amax(np.array(gpus_list)) > len(available_gpus)-1:
            print('Not all selected GPU are available.')
            print('Available GPUs are:\n', available_gpus, ".")
            print('Proceeding with all available GPUs.')
            gpus_list = list(range(len(available_gpus)))
    if len(gpus_list) > 1:
        selected_gpus = np.array(available_gpus)[gpus_list].tolist()
        print(len(gpus_list), "GPUs have been set:\n" +
              "\n".join([str(x) for x in selected_gpus]), '.')
        return selected_gpus
    else:
        print("1 GPU hase been set:\n"+str(available_gpus[gpus_list[0]]), '.')
        return [available_gpus[gpus_list[0]]]

def set_gpus_env(gpus_list="all", verbose=True):
    global ShowPrints
    ShowPrints = verbose
    available_gpus = get_available_gpus(verbose=False)
    ShowPrints = verbose
    if len(available_gpus) == 0:
        print('No available GPUs. Running with CPU support only.')
        return available_gpus
    if gpus_list is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("No GPUs have been set. Running with CPU support only.")
        return []
    elif gpus_list is "all":
        gpus_list = list(range(len(available_gpus)))
    else:
        if np.amax(np.array(gpus_list)) > len(available_gpus)-1:
            print('Not all selected GPU are available.')
            print('Available GPUs are:\n', available_gpus,".")
            print('Proceeding with all available GPUs.')
            gpus_list = list(range(len(available_gpus)))
    if len(gpus_list) > 1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus_list).replace('[','').replace(']','')
        selected_gpus = np.array(available_gpus)[gpus_list].tolist()
        print(len(gpus_list), "GPUs have been set:\n" +
              "\n".join([str(x) for x in selected_gpus]), '.')
        return selected_gpus
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_list[0])
        print("1 GPU hase been set:\n"+str(available_gpus[gpus_list[0]]), '.')
        return [available_gpus[gpus_list[0]]]
