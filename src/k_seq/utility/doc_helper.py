"""Helper function to control docstrings for arguments/methods/more with same names at one place
"""


class DocHelper(object):
    """Control docstring for arguments/variables/methods/more with same names at one place

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
            self.add(**kwargs)

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
            return f"{variable.name}: {'' if variable.isna()['docstring'] else variable['docstring']}"
        else:
            return f"{variable.name} ({variable['dtype']}): {variable['docstring']}"

    def get(self, var_names, indent=4, sep='\n'):
        """Generate a formatted docstring

        Args:

            var_names (list, tuple, callable): a list or tuple of variable names to retrieve, if the variable name does
                not exist in record, a line with null info will be created

            indent (int): indent for the docstring lines. Default 4

            sep (str): separation symbols between docstring lines (in addition of a natural line break). Default `\n`
        """
        if callable(var_names):
            from .func_tools import get_func_params
            var_names = get_func_params(func=var_names, exclude_x=False)
            var_names = [name for name in var_names if name != 'self']
        elif isinstance(var_names, str):
            var_names = [var_names]
        else:
            var_names = list(var_names)

        # strip numbers in var_name and use as indent
        var_names_t = []
        for var in var_names:
            try:
                var = int(var)
                indent = int(var)
            except:
                var_names_t.append(var)

        var_names = var_names_t
        indent = ' ' * indent
        doc = list(self.var_lib.reindex(var_names).apply(self._record_to_string, axis=1))
        return indent + (sep + indent).join(doc)

    @staticmethod
    def split_string(string):

        def parse_arg_domain(domain):
            import re
            return tuple([arg for arg in re.split(r'\s*[;|,|\s|<<|>>]\s*', domain) if arg != ''])

        split = []
        start_loc = string.find('<<')
        while start_loc >= 0:
            end_loc = string.find('>>', start_loc)
            if end_loc >= 0:
                split.append(string[:start_loc])
                split.append(parse_arg_domain(string[start_loc: end_loc + 2]))
                string = string[end_loc + 2:]
                start_loc = string.find('<<')
            else:
                start_loc = -1
        if string != '':
            split.append(string)
        return split

    def compose(self, docstring, indent=4, sep='\n'):
        docstring = ''.join(
            [s if isinstance(s, str) else self.get(s, indent=indent, sep=sep)
             for s in self.split_string(docstring)]
        )

        def decorator(func):
            func.__doc__ = docstring
            return func

        return decorator


