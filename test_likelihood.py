from source.files import *

#
# Steering file for running the likelihood program for testing and development purposes.
# The file handles the creation of the dataset, of the model and its training, and of the saving of the results. 
# Everything for documentation and preservation should enter here.
#

#
# Configuration file name. It hosts all configuration variables and outputs its content
# in a configuration file. Designed to allow working in both online and remote modes.
#

conf = config("my_config_test_1")
conf.add("PARAM_1",15)
conf.add("PARAM_2","12")
conf.add("PARAMLIST_1",[1,2,3])
conf.add("PARAMLIST_2",["100","200"])
conf.add("aaACTIVATION_FUNCTION","sigmoid")

# Creation of the dataset

ensemble = DNNLikelihoodEnsemble("Pippo",3,10)
print ensemble.print_info()
ensemble.get_likelihood(1).print_info()
 
# 2. Configuration of the model

#hidden_layer_list = [
#    [[100,'relu'],[200,'relu'],[100,'relu']],
#    ]
#activation_function = ['sigmoid']
#dropout_rate = [0]
#optimizer = 'adam'
#
#my_model = model(hidden_layer_list,activation_function,dropout_rate,optimizer)
#
# 3. Training 

# 4. Saving of the results

