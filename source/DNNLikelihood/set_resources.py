import os
import numpy as np
from multiprocessing import cpu_count
from tensorflow.python.client import device_lib

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

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    available_gpus = [[x.name, x.physical_device_desc]
                      for x in local_device_protos if x.device_type == 'GPU']
    print(str(len(available_gpus))+" GPUs available in current environment")
    return available_gpus

def get_available_cpus():
    availableCPUCoresNumber = cpu_count()
    print(str(availableCPUCoresNumber)+" CPU cores available")
    return availableCPUCoresNumber

def setGPUs(n,multi_gpu=False):
    available_gpus = get_available_gpus()
    if len(available_gpus)==0:
        print('No available GPUs in current environment.')
        return list([])
    if np.amax(np.array(n)) > len(available_gpus)-1:
        print('Not all GPU numbers selected are available, please restart the kernel, change your selection and execute again.')
        print('Available GPUs are:\n',available_gpus)
        return list([])
    else:
        if len(n)>1:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=str(n).replace('[','').replace(']','')
            if multi_gpu:
                from keras.utils import multi_gpu_model
                print("Multi GPU mode activated. Imported Keras multi GPU model module.")
            print(len(n), "GPUs have been set:\n"+"\n".join([str(x) for x in available_gpus]), '.')
            return available_gpus
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=str(n[0])
            print("1 GPU hase been set:\n"+str(available_gpus[n[0]]),'.')
            return [available_gpus[n[0]]]
