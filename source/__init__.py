from . import set_resources
from . import utility
from . import inference
from .histfactory import histfactory
from .Lik import Lik
from .mcmc import MCMC
from .data_sample import Data_sample
from .DNNLik_ensemble import DNNLik_ensemble
from .DNNLik import DNNLik

#from .DNNLik import DNNLik
#from . import files

# Print strategy: most functions have an optional argument verbose. 
#   Warnings and errors always print. When verbose=0 no information is printed.
#   When verbose=-1 only information from the current function is printed, while
#   information from embedded functions is not printed. When verbose > 0 all information are
#   printed. Finally, for functions with different verbose modes (like tf.keras model.fit) the
#   verbose argument is passed to the function and set to zero if verbose = -1.
#   !!!!!!!!!!!!! Should review the ShowPrints strategy: check if it's useful or if we can get rid of it
