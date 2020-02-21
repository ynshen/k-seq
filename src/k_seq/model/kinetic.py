"""A collection of commonly used kinetic models

Notes:
    kinetic models were currently implemented as callable function. It might migrate to `Model` subclass for
      storing parameters, set required parameters, etc
"""

import numpy as np
from ..utility import DocHelper
from ..utility.log import logging

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
    broadcast=('bool', 'if True, apply broadcast for k and A, outer product between A and k are calculated;'
                       'if False, k and A have to be same length and apply elementwise production')
)


def check_scalar(value):
    """Check if value is a scalar or is the single value in a vector/matrix/tensor"""
    return len(np.reshape(value, newshape=(-1))) == 1


def to_scalar(value):
    """Try to convert a value to scalar, if possible"""
    if len(np.reshape(value, newshape=(-1))) == 1:
        return np.reshape(value, newshape=(-1))[0]
    else:
        return np.array(value)


def first_order(c, k, A, alpha, t, broadcast=True):
    f"""Base first-order kinetic model, returns reacted fraction of input seqs given parameters
    broadcast are available on A, k, c and a full return tensor will have shape (A, k, c)
    if any of these 3 parameters is scalar, the dimension is automatically squeezed while maintaining the order
    Note: for c_i < 0, returns ones as it is input pool
     
    Args:
    {doc_helper.get(first_order, indent=4)} 
    """

    if check_scalar(c):
        c = np.array([to_scalar(c)])
    else:
        c = np.array(c)
    if check_scalar(k):
        k = np.array([to_scalar(k)])
    else:
        k = np.array(k)
    if check_scalar(A):
        A = np.array([to_scalar(A)])
    else:
        A = np.array(A)

    if broadcast:
        # dim  param
        #  0     A
        #  1     k
        #  2     c
        y = np.outer(A, (1 - np.exp(- alpha * t * np.outer(k, c))))
        y = y.reshape((len(A), len(k), len(c)))
        y[:, :, c < 0] = 1

        dim_to_squeeze = []
        for dim in (0, 1, 2):
            if y.shape[dim] == 1:
                dim_to_squeeze.append(dim)
    else:
        # dim param
        #  0   k, A
        #  1    c

        if len(k) != len(A):
            logging.error('k and A should have same length when broadcasting is disabled', error_type=ValueError)

        y = np.expand_dims(A, -1) * (1 - np.exp(- alpha * t * np.outer(k, c)))
        y[:, c < 0] = 1

        dim_to_squeeze = []
        for dim in (0, 1):
            if y.shape[dim] == 1:
                dim_to_squeeze.append(dim)

    y = np.squeeze(y, axis=tuple(dim_to_squeeze))
    return y


class BYOModel:
    r"""A collection of BYO kinetic models (static functions)
    where some values are fixed:
        exp time (t): 90 min
        BYO degradation factor ($\alpha$): 0.479

    functions:
        - reacted_frac
        - amount_first_order
        - composition_first_order
        
    parameters:
    {params}

    return:
        concentration/composition at the name (time, concentration) point
    """.format(params=doc_helper.get(['p0', 'c', 'k', 'A']))

    @staticmethod
    def reacted_frac(c, k, A, broadcast=True):
        f"""Reacted fraction for each seq in a pool
        Args:
        {doc_helper.get(BYOModel.reacted_frac)}
        
        Return:
            Reacted fraction for each sequence in each sample
            float, 1-D or 2-D np.ndarray with shape (uniq_seq_num, sample_num)
        """

        return first_order(c=c, k=k, A=A, broadcast=broadcast, alpha=0.479, t=90)

    @staticmethod
    def amount_first_order(c, p0, k, A, broadcast=False):
        f"""Absolute amount of reacted pool, if c is negative, return 0
        Notice: broadcast should be False if multiple p0 is used.

        Args:
        {doc_helper.get(BYOModel.amount_first_order)}
        
        Return:
            Absolute amount of each sequence in each sample
            float, 1-D or 2-D np.ndarray with shape (uniq_seq_num, sample_num)
        """

        reacted_frac = BYOModel.reacted_frac(c=c, k=k, A=A, broadcast=broadcast)
        if len(np.shape(reacted_frac)) == 1:
            return np.array(p0) * reacted_frac
        else:
            return np.expand_dims(p0, -1) * reacted_frac

    @staticmethod
    def composition_first_order(c, p0, k, A):
        f"""Function of pool composition w.r.t. BYO concentration (c)
        if x < 0, output is p0
        broadcast is not supported that is k and A should have sem length

        Args:
        {BYOModel.composition_first_order}

        Return:
            Pool composition for sequences in each sample
            float, 1-D or 2-D np.ndarray with shape (len(k), sample_num)
        """

        amounts = BYOModel.amount_first_order(c=c, p0=p0, k=k, A=A, broadcast=False)

        if check_scalar(p0):
            return amounts
        else:
            return amounts / np.sum(amounts, axis=0)
