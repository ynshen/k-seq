
class DocHelper(object):
    """Helper for docstring controls in a module

    Attributes:
        var_lib (pd.DataFrame): contains all documented variables, include columns of name (index), dtype, doc

    Methods:

        add: add keyword arguments to the doc_helper

        get: generate a formatted docstring

    """

    def __init__(self, **kwargs):
        """Add arguments in initialization by keyword arguments
        Accept two values for kwargs:
            - str: the documentation string
            - tuple of (str, str): (variable type, docstring)

        Examples:
            doc_strings = DocHelper(x='the first integer', y=('int', 'the second value'))
        """
        import pandas as pd
        self.var_lib = pd.DataFrame(columns=('name', 'dtype', 'docstring')).set_index('name')
        if kwargs != {}:
            self._add(self.var_lib, **kwargs)

    @staticmethod
    def _add(lib, **kwargs):
        for key, doc in kwargs.items():
            if isinstance(doc, str):
                lib.loc[key, 'docstring'] = doc
            elif isinstance(doc, (list, tuple)):
                lib.loc[key, 'dtype'] = doc[0]
                lib.loc[key, 'docstring'] = doc[1]

    def add(self, **kwargs):
        """Add kwarg arguments to the doc helper var_lib"""
        self._add(self.var_lib, **kwargs)

    @staticmethod
    def _record_to_string(variable):
        if variable.isna()['dtype']:
            return f"{variable.name}: {'' if variable.isna()['docstring'] else variable['docstring']}\n"
        else:
            return f"{variable.name} ({variable['dtype']}): {variable['docstring']}\n"

    def get(self, var_names, indent=4, sep=''):
        """Generate a formatted docstring

        Args:

            var_names (list, tuple, callable): a list or tuple of variable names to retrieve, if the variable name does
                not exist in record, a line with null info will be created

            indent (int): indent for the docstring lines. Default 4

            sep (str): separation symbols between docstring lines (in addition of a natural line break). Default `\n`
        """
        if callable(var_names):
            var_names = get_func_params(func=var_names, exclude_x=False)
            var_names = [name for name in var_names if name != 'self']
        else:
            var_names = list(var_names)
        indent = ' ' * indent
        doc = list(self.var_lib.reindex(var_names).apply(self._record_to_string, axis=1))
        return indent + (sep + indent).join(doc)


def var_to_doc(doc_var):
    """Deprecated.
    Convert a variable (dictionary or list/tuple)of args/attrs/methods documents to a string of documentation

    Args:
        doc_var ('dict` or list-like): variable contains document info

    Returns:
        `str` of docstring
    """

    def parse_dict_value(single_var):

        if isinstance(single_var, str):
            return f': {single_var}\n'
        elif isinstance(single_var, (tuple, list)):
            if len(single_var) == 1:
                return f': {single_var[0]}\n'
            elif len(single_var) >= 2:
                return f"(`{' or '.join(single_var[0:-1])}`): {single_var[-1]}\n"
            else:
                raise TypeError('List should have format (var_doc), or (var_type, var_doc)')
        else:
            raise TypeError('Unknown variable doc string type')

    if isinstance(doc_var, dict):
        doc_string = '\n  '.join([f'{key}{parse_dict_value(value)}' for key, value in doc_var.items()])
    elif isinstance(doc_var, (list, tuple)):
        raise NotImplementedError('list-like docstring not implemented yet')
    else:
        raise TypeError('Unknown docstring type doc_dict')

    return doc_string


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


def get_object_hex(obj):
    return f"<{obj.__class__.__module__}{obj.__class__.__name__} at {hex(id(obj))}>"


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


def check_attr_value(obj, **attr):
    for name, value in attr.items():
        if value is None:
            attr[name] = getattr(obj, name)
    return attr


def dict_flatten(d, parent_key='', sep='_'):

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items += [(key, value) for key, value in dict_flatten(v, new_key, sep=sep).items()]
        else:
            items.append((new_key, v))
    return dict(items)


def get_func_params(func, exclude_x=True):
    """
    Utility function to get the number of arguments for a function (callable)
    Args:
        func (`callable`): the function
        exclude_x (`bool`): if exclude the first argument (usually `x`)

    Returns: a tuple of arguments name in order

    """
    from inspect import signature, isfunction, isclass

    if callable(func):
        arg_tuple = tuple(signature(func).parameters.keys())
    else:
        raise TypeError('Unidentified func passed')
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
        return self.__getattribute__(item)

    def add(self, attr_dict=None, **kwargs):
        if attr_dict is None:
            attr_dict = {}
        if kwargs is None:
            kwargs = {}
        self.__dict__.update({**kwargs, **attr_dict})
