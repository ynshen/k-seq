"""A collection of commonly used kinetic models used in the project
"""
import numpy as np
from ..utility.func_tools import DocHelper

doc_helper = DocHelper(
    c=('float or 1-D list-like', 'Concentration of substrates'),
    k=('float or 1-D list-like', 'Kinetic coefficient for each sequence in first-order kinetics, with length as number '
                                 'of sequences'),
    A=('float or 1-D list-like', 'Asymptotic conversion for each sequence in first-order kinetics, with length as '
                                 'number of sequences'),
    alpha=('float', 'degradation coefficient for substrates'),
    t=('float', 'Reaction time in the experiments'),
    b=('float or 1-D list-like', 'Bias for each sequences, with length as number of sequences'),
    p0=('1-D list-like', 'Composition of sequences in the input pool. sum(p0) == 1'),
    x0=('1-D list-like', 'Initial amount of sequences in the input pool'),

)


def check_scaler(value):
    """Check if value is a scaler or is the single value in a vector/matrix/tensor"""
    return len(np.reshape(value, newshape=(-1))) == 1


def to_scaler(value):
    """Try to convert a value to scaler, if possible"""
    if len(np.reshape(value, newshape=(-1))) == 1:
        return np.reshape(value, newshape=(-1))[0]
    else:
        return np.array(value)


def first_order(c, k, A, alpha, t):
    """Base first-order kinetic model
    Args:
    {}
    """.format(doc_helper.get(first_order), indent=4)

    if check_scaler(k) or check_scaler(c):
        # Two 1-D array, compute outer product
        return to_scaler(A) * (1 - np.exp(- alpha * t * to_scaler(k) * to_scaler(c)))
    else:
        if len(np.shape(A)) == 1:
            A = np.expand_dims(A, axis=-1)
        return np.multiply(A, (1 - np.exp(- alpha * t * np.outer(k, c))))


def first_order_w_bias(c, k, A, alpha, t, b):
    """Base first order kinetic model with bias
    Args:
    {}
    """.format(doc_helper.get(first_order_w_bias))

    if check_scaler(k) or check_scaler(c):
        # Two 1-D array, compute outer product
        return to_scaler(A) * (1 - np.exp(- alpha * t * to_scaler(k) * to_scaler(c))) + to_scaler(b)
    else:
        if len(np.shape(A)) == 1:
            A = np.expand_dims(A, axis=-1)
        if len(np.shape(b)) == 1:
            b = np.expand_dims(b, axis=-1)
        return A * (1 - np.exp(- alpha * t * np.outer(k, c))) + b


class BYOModel:
    """A collection of BYO kinetic models (static functions)

    where some values are fixed:
        exp time (t): 90 min
        BYO degradation factor ($\alpha$): 0.479

    functions:
        - composition_first_order
        -
    parameters:
        p0: initial pool composition, needed if using actual composition
        c: controlled variable, in our case, BYO concentration
        k: kinetic coefficient
        A: fraction of reactive molecules
        bias: if include slope in the model for baseline passing rate

    return:
        concentration/composition at the target (time, concentration) point
    """

    # def __init__(self, p0=None, c=None, k=None, A=None, bias=False, use_reacted_frac=True,):
    #     super().__init__()
    #     self.alpha = 0.479
    #     self.t = 90
    #     self.p0 = p0
    #     self.c = c
    #     self.k = k
    #     self.A = A
    #     self.bias = bias
    #     if use_reacted_frac:
    #         if bias:
    #             raise NotImplementedError('reacted fraction with bias is not implemented yet')
    #         else:
    #             self.func = self.react_frac
    #     else:
    #         if bias:
    #             self.func = self.composition_first_order_w_bias
    #         else:
    #             self.func = self.composition_first_order

    @staticmethod
    def reacted_frac(c, k, A):
        """Reacted fraction for each seq in a pool
        Args:
        {}
        
        Return:
            Reacted fraction for each sequence in each sample
            float, 1-D or 2-D np.ndarray with shape (seq_num, sample_num)
        """.format(doc_helper.get(BYOModel.reacted_frac))

        c = np.array(c)
        k = np.array(k)
        A = np.array(A)
        return first_order(c=c, k=k, A=A, alpha=0.479, t=90)

    @staticmethod
    def amount_first_order(c, x0, k, A):
        """Absolute amount of reacted pool, if c is negative, return x0

        Args:
        {}
        
        Return:
            Absolute abount of each sequence in each sample
            float, 1-D or 2-D np.ndarray with shape (seq_num, sample_num)

        """.format(doc_helper.get(BYOModel.amount_first_order))
        reacted_frac = BYOModel.reacted_frac(c=c, k=k, A=A)

        if check_scaler(c):
            if to_scaler(c) < 0:
                reacted_frac[:] = 1
        else:
            reacted_frac[:, c < 0] = 1

        if len(np.shape(reacted_frac)) == 1:
            return np.array(x0) * reacted_frac
        else:
            np.expand_dims(x0, -1) * reacted_frac

    @staticmethod
    def composition_first_order(c, p0, k, A):
        """Function of pool composition w.r.t. BYO concentration (x)
        if x < 0, output is p0

        Args:
        {}

        Return:
            Pool composition for sequences in each sample
            float, 1-D or 2-D np.ndarray with shape (seq_num, sample_num)
        """.format(BYOModel.composition_first_order)

        amounts = BYOModel.amount_first_order(c=c, x0=p0, k=k, A=A)

        return amounts / np.sum(amounts, axis=0)

    @staticmethod
    def composition_first_order_w_bias(c, p0, k, A, b):
        """Function of pool composition w.r.t. BYO concentration (x)
        TODO: kinetics with bias is not updated yet
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
            p = p0 * first_order_w_bias(c=c, k=k, A=A, b=b, alpha=0.479, t=90)
            return p / np.sum(p)

