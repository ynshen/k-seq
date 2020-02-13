"""A collection of commonly used kinetic models used in the project
"""
import numpy as np
from ..utility.doc_helper import DocHelper

doc_helper = DocHelper(
    c=('float or 1-D list-like', 'Concentration of substrates, return input pool if negative'),
    k=('float or 1-D list-like', 'Kinetic coefficient for each sequence in first-order kinetics, with length as number '
                                 'of sequences'),
    A=('float or 1-D list-like', 'Asymptotic conversion for each sequence in first-order kinetics, with length as '
                                 'number of sequences'),
    alpha=('float', 'degradation coefficient for substrates'),
    t=('float', 'Reaction time in the experiments'),
    b=('float or 1-D list-like', 'Bias for each sequences, with length as number of sequences'),
    p0=('1-D list-like', 'Amount or composition of sequences in the input pool'),

)


def check_scaler(value):
    """Check if value is a scalar or is the single value in a vector/matrix/tensor"""
    return len(np.reshape(value, newshape=(-1))) == 1


def to_scaler(value):
    """Try to convert a value to scaler, if possible"""
    if len(np.reshape(value, newshape=(-1))) == 1:
        return np.reshape(value, newshape=(-1))[0]
    else:
        return np.array(value)


def first_order(c, k, A, alpha, t):
    """Base first-order kinetic model, returns reacted fraction of input seqs given parameters
    broadcast are available on A, k, c and a full return tensor will have shape (A, k, c)
    if any of these 3 parameters is scalar, the dimension is automatically squeezed while maintaining the order
    Note: for c_i < 0, returns ones as it is input pool
     
    Args:
    {}
    """.format(doc_helper.get(first_order), indent=4)

    # dim  param
    #  0     A
    #  1     k
    #  2     c

    if check_scaler(c):
        c = np.array([to_scaler(c)])
    else:
        c = np.array(c)
    if check_scaler(k):
        k = np.array([to_scaler(k)])
    else:
        k = np.array(k)
    if check_scaler(A):
        A = np.array([to_scaler(A)])
    else:
        A = np.array(A)

    y = np.outer(A, (1 - np.exp(- alpha * t * np.outer(k, c))))
    y = y.reshape((len(A), len(k), len(c)))
    y[:, :, c < 0] = 1

    dim_to_squeeze = []
    for dim in (0, 1, 2):
        if y.shape[dim] == 1:
            dim_to_squeeze.append(dim)

    y = np.squeeze(y, axis=tuple(dim_to_squeeze))
    return y


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

        return first_order(c=c, k=k, A=A, alpha=0.479, t=90)

    @staticmethod
    def amount_first_order(c, p0, k, A):
        """Absolute amount of reacted pool, if c is negative, return x0

        Args:
        {}
        
        Return:
            Absolute abount of each sequence in each sample
            float, 1-D or 2-D np.ndarray with shape (seq_num, sample_num)

        """.format(doc_helper.get(BYOModel.amount_first_order))
        reacted_frac = BYOModel.reacted_frac(c=c, k=k, A=A)

        if len(np.shape(reacted_frac)) == 1:
            return np.array(p0) * reacted_frac
        else:
            return np.expand_dims(p0, -1) * reacted_frac

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

        amounts = BYOModel.amount_first_order(c=c, p0=p0, k=k, A=A)

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

