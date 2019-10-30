"""A collection of kinetic models used in the project


"""
from . import ModelBase


def first_order(c, k, A, alpha, t):
    """Base first order kinetic model
    Args:
        c (`float` or np.array): initial amount(s), could be composition of a pool
        k (`float` or np.array): kinetic coefficient(s)
        A (`float` or np.array): ratio of active molecules
        alpha (`float`): degradation parameter for substrates
        t (`float`): reaction time
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * c))


def first_order_w_slope(c, k, A, alpha, t, b):
    """Base first order kinetic model
    Args:
        c (`float` or np.array): initial amount(s), could be composition of a pool
        k (`float` or np.array): kinetic coefficient(s)
        A (`float` or np.array): ratio of active molecules
        alpha (`float`): degradation parameter for substrates
        t (`float`): reaction time
        b (`float` or np.array): slopes
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * c)) + b


class BYOModel(ModelBase):
    """A collection of BYO kinetic models, where
       exp time (t): 90 min
       BYO degradation factor (\alpha): 0.479

    -

    input:
    p0: initial pool, not necessarily composition
    c: controlled variable, in our case, BYO concentration
    k: kinetic coefficient
    A: fraction of reactive molecules
    slope: if include slope in the model for baseline passing rate

    return: concentration in at the target (time, concentration) point
    """

    def __init__(self, p0=None, c=None, k=None, A=None, slope=False):
        super().__init__()
        self.p0 = p0
        self.c = c
        self.k = k
        self.A = A
        self.slope = slope

    @staticmethod
    def composition_first_order_no_slope(c, p0, k, A):
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
    def composition_first_order(c, p0, k, A, b):
        """Function of pool composition w.r.t. BYO concentration (x)
        if x < 0, output is p0

        Parameters:

        - p0: initial pool composition

        - k: kinetic coefficient

        - A: maximal conversion ratio

        - b: slope

        """

        import numpy as np

        p0 = np.array(p0)
        k = np.array(k)
        A = np.array(A)
        b = np.array(b)

        if c < 0:
            return p0
        else:
            return p0 * first_order_w_slope(c=c, k=k, A=A, b=b, alpha=0.479, t=90)
