__all__ = ["Common"]

from datetime import datetime
from . import inference, utils
from .show_prints import Verbosity, print

class Shared(Verbosity):
    """
    This class contains methods inherited by multiple classes
    """
    def __init__(self, verbose = True):
        """
        Bla bla
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        
    def __check_define_name(self,suffix=""):
        """
        Private method used by the ``__init__`` method of various classes
        to define the object name attribute :attr:`name <common_classes_attributes.name>` attribute.
        If it is ``None`` it replaces it with
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_"+suffix``,
        otherwise it appends the suffix ``"_"+suffix`` (preventing duplication if it is already present).
        """
        if self.name == None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_"+suffix
        else:
            self.name = utils.check_add_suffix(self.name, "_"+suffix)
