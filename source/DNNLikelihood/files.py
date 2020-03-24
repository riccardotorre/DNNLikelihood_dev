import sys
import glob
import os
import array

# 
# Base class for the dataset. Datasets for training, validation and testing inherits from this class.
#

class config:
    def __init__(self, name):
        self.name = name
        self.config_dict = {}

        self.f = open(name,"w+")

    def add(self,name_param,param):

        allowed_name_param = ["PARAM_1",
                              "PARAM_2",
                              "PARAMLIST_1",
                              "PARAMLIST_2",
                              "ACTIVATION_FUNCTION"
                              ]
        try:
            index = allowed_name_param.index(name_param)
        except ValueError:
            print("Param name not allowed. Exiting")
        else:
            self.config_dict[name_param] = param 
            self.f.write(name_param + " " + str(self.config_dict[name_param]) + "\n")

class DNNLikelihood:
    def __init__(self, name, param1):
        self.name  = name
        self.param1 = param1

    def print_info(self):
        print("Classe "+self.name+", eventi "+str(self.param1))

class DNNLikelihoodEnsemble:
    def __init__(self, name, n_dim, param1):
        self.name  = name
        self.n_dim = n_dim
        self.param1 = param1
        self.vec = []
        for item in range(n_dim):
            self.vec.append(DNNLikelihood(name+"_"+str(item),param1))

    def print_info(self):
        print("Class "+self.name+" with "+str(len(self.vec))+" likelihoods")

    def get_likelihood(self,i):
        return self.vec[i]


# 
# The training dataset is a fraction of the dataset.
#

#class training_dataset(dataset)
#    def __init__(self, fraction):
#        self.fraction = fraction
#
#    #def get_random():


