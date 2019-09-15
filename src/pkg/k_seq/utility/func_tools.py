
def get_args_params(func, exclude_x=True):
    """
    Utility function to get the number of arguments for a function (callable)
    Args:
        func (`callable`): the function
        exclude_x (`bool`): if exclude the first argument (usually `x`)

    Returns: a tuple of arguments name in order

    """
    from inspect import signature
    arg_tuple = tuple(signature(func).parameters.keys())
    if exclude_x:
        return arg_tuple[1:]
    else:
        return arg_tuple


class FunctionWrapper:

    @staticmethod
    def _wrap_function(func, data):
        from functools import partial, update_wrapper
        wrapped = partial(func, data)
        update_wrapper(wrapped, func)
        return wrapped

    def __init__(self, data, functions):
        if callable(functions):
            functions = [functions]

        from functools import partial, update_wrapper

        self.__dict__.update({
            func.__name__: self._wrap_function(func, data) for func in functions
        })


class DotDict(dict):
    """A dict with dot access and autocompletion. The snippet get from git/globor

    The idea and most of the code was taken from
    http://stackoverflow.com/a/23689767,
    http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
    http://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-get-set
    """

    def __init__(self, *a, **kw):
        dict.__init__(self)
        self.update(*a, **kw)
        self.__dict__ = self

    def __setattr__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError('This key is reserved for the dict methods.')
        dict.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if key in dict.__dict__:
            raise AttributeError('This key is reserved for the dict methods.')
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self
