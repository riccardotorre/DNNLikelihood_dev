import builtins

verbose = True
def print(*args, **kwargs):
    global verbose
    if type(verbose) is bool:
        if verbose:
            return builtins.print(*args, **kwargs)
    if type(verbose) is int:
        if verbose != 0:
            return builtins.print(*args, **kwargs)
