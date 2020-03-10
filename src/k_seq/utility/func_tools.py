import numpy as np
import pandas as pd
from yutility import logging


def is_int(x):
    return isinstance(x, (int, np.int_, np.int0, np.int8, np.int16, np.int32, np.int64))


def is_numeric(x):
    return is_int(x) or isinstance(x, (float, np.float_, np.float16, np.float32, np.float64))


def is_sparse(obj):
    if isinstance(obj, pd.Series):
        return pd.api.types.is_sparse(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.apply(pd.api.types.is_sparse).all()


def update_none(arg, update_by):
    """Update arguments with some default value
    Args:
        arg: variable object
        update_by: variable object
    """
    if arg is None:
        return update_by
    else:
        return arg


def dict_flatten(d, parent_key='', sep='_'):

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items += [(key, value) for key, value in dict_flatten(v, new_key, sep=sep).items()]
        else:
            items.append((new_key, v))
    return dict(items)


def get_func_params(func, required_only=True):
    """Get the name of arguments for a function (callable), or the arguments in __init__ for a Class (seq_data not included)

    Args:
        func (`callable`): the function
        required_only (bool): if exclude arguments with default values

    Returns: a list of arguments name in order
    """
    from inspect import signature

    if not callable(func):
        logging.error('func is not a callable', error_type=TypeError)

    sign = signature(func)
    if required_only:
        return [key for key, param in sign.parameters.items() if param.default is sign.return_annotation]
    else:
        return list(sign.parameters.keys())


class FuncToMethod(object):
    """Convert a set of functions to a collection of methods on the object"""

    @staticmethod
    def _wrap_function(func, *args, **kwargs):
        from functools import partial, update_wrapper
        if kwargs == {} and args == ():
            return func
        else:
            wrapped = partial(func, *args, **kwargs)
            update_wrapper(wrapped, func)
            return wrapped

    def __init__(self, functions, *args, **kwargs):
        if callable(functions):
            functions = [functions]

        for func in functions:
            setattr(self, func.__name__, self._wrap_function(func, *args, **kwargs))


class AttrScope(object):
    """A name scope for a group of attributes"""

    def __repr__(self):
        return f"An attribute class with keys: {list(self.__dict__.keys())}"

    def __init__(self, attr_dict=None, keys=None, **attr_kwargs):
        """Create a name scope for a group of attributes

        Args:
            attr_dict (dict): a dictionary with values to pass
            keys (list of str): a list of attributes to initialize with None
            attr_kwargs: or directly pass some keyword arguments
        """
        if keys is not None:
            for key in keys:
                setattr(self, key, None)
        if attr_dict is None:
            attr_dict = {}
        if attr_kwargs is None:
            attr_kwargs = {}
        self.__dict__.update({**attr_dict, **attr_kwargs})

    def __getitem__(self, item):
        return getattr(self, item)

    def add(self, attr_dict=None, **kwargs):
        if attr_dict is None:
            attr_dict = {}
        if kwargs is None:
            kwargs = {}
        self.__dict__.update({**kwargs, **attr_dict})


def param_to_dict(key_list, **kwargs):
    """Assign kwargs to the dictionary with key from key_list
    - if the arg is a single value, it will be assigned to all keys
    - if the arg is a list, it will should have same length as key_list
    - if the arg is a dict, it should contain all members in the key
    """
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



def check_attr_value(obj, **attr):
    for name, value in attr.items():
        if value is None:
            attr[name] = getattr(obj, name)
    return attr