
def param_to_dict(key_list, **kwargs):
    """Assign kwargs to the dictionary with key from key_list
    - if the arg is a single value, it will be assigned to all keys
    - if the arg is a list, it will should have same length as key_list
    - if the arg is a dict, it should contain all members in the key
    """
    import numpy as np
    import pandas as pd

    def parse_args(kwargs, key, ix):
        arg_dict = {}
        for arg_name, arg in kwargs.items():
            if isinstance(arg, (list, np.ndarray, pd.Series)):
                if len(arg) == len(key_list):
                    arg_dict[arg_name] = arg[ix]
                else:
                    raise ValueError(f"{arg_name} is a list, but the length does not match sample files")
            elif isinstance(arg, dict):
                if key in arg.keys():
                    arg_dict[arg_name] = arg[key]
                else:
                    raise KeyError(f'Sample {key} not found in {arg_name}')
            else:
                arg_dict[arg_name] = arg
        return arg_dict

    if isinstance(key_list, dict):
        key_list = list(key_list.keys())

    return {key:parse_args(kwargs, key, ix) for ix,key in enumerate(key_list)}


def get_func_params(func, exclude_x=True):
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


class FuncToMethod(object):
    """Convert a set of functions to a collection of methods on the object"""

    @staticmethod
    def _wrap_function(func, obj=None):
        from functools import partial, update_wrapper
        if obj is None:
            return func
        else:
            wrapped = partial(func, obj)
            update_wrapper(wrapped, func)
            return wrapped

    def __init__(self, functions, obj=None):
        if callable(functions):
            functions = [functions]

        self.__dict__.update({
            func.__name__: self._wrap_function(func, obj=obj) for func in functions
        })


class DictToAttr(object):
    """Convert a dictionary to a group of attributes"""

    def __repr__(self):
        return f"An attribute class with keys: {list(self.__dict__.keys())}"

    def __init__(self, attr_dict):
        if isinstance(attr_dict, dict):
            self.__dict__.update(attr_dict)
        else:
            raise TypeError('attr_dict needs to be dictionary')

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def add(self, attr_dict):
        self.__dict__.update(attr_dict)

