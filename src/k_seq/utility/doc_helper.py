"""Helper function to control docstrings for arguments/methods/more with same names at one place
"""


class DocHelper(object):
    """Control docstring for arguments/varaibles/methods/more with same names at one place

    Attributes:
        var_lib (pd.DataFrame): contains all documented variables, include columns of name (index), dtype, doc

    Methods:
        add: add docstring for keyword args
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

    def add(self, **kwargs):
        """Add kwarg arguments to the doc helper var_lib"""

        for key, doc in kwargs.items():
            if isinstance(doc, str):
                self.var_lib.loc[key, 'docstring'] = doc
            elif isinstance(doc, (list, tuple)):
                self.var_lib.loc[key, 'dtype'] = doc[0]
                self.var_lib.loc[key, 'docstring'] = doc[1]

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
        from .func_tools import get_func_params
        if callable(var_names):
            var_names = get_func_params(func=var_names, exclude_x=False)
            var_names = [name for name in var_names if name != 'self']
        else:
            var_names = list(var_names)
        indent = ' ' * indent
        doc = list(self.var_lib.reindex(var_names).apply(self._record_to_string, axis=1))
        return indent + (sep + indent).join(doc)