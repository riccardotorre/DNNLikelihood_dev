import sys
from os import path
sys.dont_write_bytecode = True

from .show_prints import Verbosity
from .show_prints import print
from .resources import Resources
from . import custom_losses
from . import utils
from .utils import _FunctionWrapper
from . import inference
from . import corner
from .histfactory import Histfactory
from .likelihood import Lik
from .sampler import Sampler
from .data import Data
from .dnn_likelihood_ensemble import DnnLikEnsemble
from .dnn_likelihood import DnnLik
mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

#from .DNNLik import DNNLik
#from . import files

# Print strategy: most functions have an optional argument verbose. 
#   Warnings and errors always print. When verbose=0 no information is printed.
#   When verbose=-1 only information from the current function is printed, while
#   information from embedded functions is not printed. When verbose > 0 all information are
#   printed. Finally, for functions with different verbose modes (like tf.keras model.fit) the
#   verbose argument is passed to the function and set to zero if verbose = -1.
#   !!!!!!!!!!!!! Should review the ShowPrints strategy: check if it's useful or if we can get rid of it
