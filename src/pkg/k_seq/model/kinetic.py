"""A collection of commonly used kinetic models used in the project
"""
from . import ModelBase


def first_order(c, k, A, alpha, t):
    """Base first-order kinetic model
    Args:
        c (float or np.array): initial amount(s), could be composition of a pool
        k (float or np.array): kinetic coefficient(s)
        A (float or np.array): ratio of active molecules
        alpha (float): degradation parameter for substrates
        t (float): reaction time
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * c))


def first_order_w_bias(c, k, A, alpha, t, b):
    """Base first order kinetic model with bias
    Args:
        c (float or np.array): initial amount(s), could be composition of a pool
        k (float or np.array): kinetic coefficient(s)
        A (float or np.array): ratio of active molecules
        alpha (float): degradation parameter for substrates
        t (float): reaction time
        b (float or np.array): bias
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * c)) + b


class BYOModel(ModelBase):
    """A collection of BYO kinetic models

    where some values are fixed:
        exp time (t): 90 min
        BYO degradation factor (\alpha): 0.479

    parameters:
        p0: initial pool composition, needed if using actual composition
        c: controlled variable, in our case, BYO concentration
        k: kinetic coefficient
        A: fraction of reactive molecules
        bias: if include slope in the model for baseline passing rate

    return:
        concentration/composition at the target (time, concentration) point
    """

    def __init__(self, p0=None, c=None, k=None, A=None, bias=False, use_reacted_frac=True,):
        super().__init__()
        self.alpha = 0.479
        self.t = 90
        self.p0 = p0
        self.c = c
        self.k = k
        self.A = A
        self.bias = bias
        if use_reacted_frac:
            if bias:
                raise NotImplementedError('reacted fraction with bias is not implemented yet')
            else:
                self.func = self.func_react_frac
        else:
            if bias:
                self.func = self.composition_first_order_w_bias
            else:
                self.func = self.composition_first_order

    @staticmethod
    def composition_first_order(c, p0, k, A):
        """Function of pool composition w.r.t. BYO concentration (x)
        if x < 0, output is p0

        Parameters:
            - p0: initial pool composition
            - k: kinetic coefficient
            - A: maximal conversion ratio
        """

        import numpy as np

        p0 = np.array(p0)
        k = np.array(k)
        A = np.array(A)

        if c < 0:
            return p0
        else:
            return p0 * first_order(c=c, k=k, A=A, alpha=0.479, t=90)

    @staticmethod
    def composition_first_order_w_bias(c, p0, k, A, b):
        """Function of pool composition w.r.t. BYO concentration (x)
        if x < 0, output is p0

        Parameters:
            c: concentration
            p0: initial pool composition
            k: kinetic coefficient
            A: maximal conversion ratio
            b: slope
        """

        import numpy as np

        p0 = np.array(p0)
        k = np.array(k)
        A = np.array(A)
        b = np.array(b)

        if c < 0:
            return p0
        else:
            return p0 * first_order_w_bias(c=c, k=k, A=A, b=b, alpha=0.479, t=90)

    @staticmethod
    def func_react_frac(c, k, A):
        """Sequence reacted fraction"""
        import numpy as np

        k = np.array(k)
        A = np.array(A)
        return first_order(c=c, k=k, A=A, alpha=0.479, t=90)
