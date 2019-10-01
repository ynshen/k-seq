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
    """BYO Model
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
    def func(p0, c, k, A, b=None, slope=False):

        import numpy as np

        p0 = np.array(p0)
        c = c
        k = np.array(k)
        A = np.array(A)

        if c < 0:
            return p0
        if slope:
            b = np.array(b)
            return p0 * first_order_w_slope(c=c, k=k, A=A, alpha=0.479, t=90, b=b)
        else:
            return p0 * first_order(c=c, k=k, A=A, alpha=0.479, t=90)