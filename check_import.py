import sys
from timeit import default_timer as timer

print("Import full DNNLikelihood module")
sys.path.append(r"/eos.workaround/home-r/rtorre/Git/GitHub/DNNLikelihood/DNNLikelihood_dev/source")
start = timer()
import DNNLikelihood
end = timer()
print("DNNLikelihood module successfully imported in", end-start,"s.")

#print("Check __init__ sequence")
#
#print("Import 'Verbosity' class from show_prints.py")
#try:
#    from .show_prints import Verbosity
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'print' function from show_prints.py")
#try:
#    from .show_prints import print
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'Resources' class from resources.py")
#try:
#    from .resources import Resources
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import utils.py")
#try:
#    import utils
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import '_FunctionWrapper' class from utils.py")
#try:
#    from utils import _FunctionWrapper
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import inference.py")
#try:
#    import inference
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import corner.py")
#try:
#    import corner
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'HistFactory' class from histfactory.py")
#try:
#    from histfactory import Histfactory
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'Lik' class from likelihood.py")
#try:
#    from likelihood import Lik
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'Sampler' class from sampler.py")
#try:
#    from sampler import Sampler
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'Data' class from data.py")
#try:
#    from data import Data
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'DnnLik' class from dnn_likelihood.py")
#try:
#    from dnn_likelihood import DnnLik
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#
#print("Import 'DnnLikEnsemble' class from dnn_likelihood_ensemble.py")
#try:
#    from dnn_likelihood_ensemble import DnnLikEnsemble
#    print("Succeded.")
#except Exception as e:
#    print(e.message, e.args)
#