import builtins

verbose = True
#def print(*args, **kwargs):
#    global verbose
#    if type(verbose) is bool:
#        if verbose:
#            return builtins.print(*args, **kwargs)
#    if type(verbose) is int:
#        if verbose != 0:
#            return builtins.print(*args, **kwargs)


def print(*args, **kwargs):
    """
    Redefinition of the built-in print function.
    """
    global verbose
    try:
        show = kwargs.pop("show")
    except:
        show = None
    if show is None:
        if verbose:
            return builtins.print(*args, **kwargs)
    elif show:
        return builtins.print(*args, **kwargs)

class Verbosity():
    def set_verbosity(self, v):
        global verbose
        if v is None:
            verbose = self.verbose
        else:
            verbose = v
        if verbose < 0:
            verbose_sub = 0
        else:
            verbose_sub = verbose
        return [verbose, verbose_sub]
