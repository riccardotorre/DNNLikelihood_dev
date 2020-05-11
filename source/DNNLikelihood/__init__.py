import sys
sys.dont_write_bytecode = True

from .show_prints import Verbosity
from .show_prints import print
from .resources import Resources
from . import utils
from . import inference
from . import corner
from .histfactory import Histfactory
from .likelihood import Likelihood
from .sampler import Sampler
from .data import Data
from .DNN_likelihood_ensemble import DNN_likelihood_ensemble
from .DNN_likelihood import DNN_likelihood

#from .DNNLik import DNNLik
#from . import files

# Print strategy: most functions have an optional argument verbose. 
#   Warnings and errors always print. When verbose=0 no information is printed.
#   When verbose=-1 only information from the current function is printed, while
#   information from embedded functions is not printed. When verbose > 0 all information are
#   printed. Finally, for functions with different verbose modes (like tf.keras model.fit) the
#   verbose argument is passed to the function and set to zero if verbose = -1.
#   !!!!!!!!!!!!! Should review the ShowPrints strategy: check if it's useful or if we can get rid of it
