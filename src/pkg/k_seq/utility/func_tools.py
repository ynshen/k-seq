
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

    def __init__(self, attr_dict):
        if isinstance(attr_dict, dict):
            self.__dict__.update(attr_dict)
        else:
            raise TypeError('attr_dict needs to be dictionary')

    def __getitem__(self, item):
        return self.__getattribute__(item)

    def add(self, attr_dict):
        self.__dict__.update(attr_dict)

