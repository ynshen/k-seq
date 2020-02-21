"""Test code for plugins: transformer, filter, etc"""

from k_seq.data import transform
import pandas as pd
import numpy as np


def test_TotalAmountNormalizer_correct_results():
    mtx = np.array([[4, 2],
                    [0, 3],
                    [5, 0]])
    full_table = pd.DataFrame(mtx, columns=['A', 'B'])
    total_amounts = {'A': 0.5, 'B': 0.3}
    transformed = transform.TotalAmountNormalizer(full_table=full_table,
                                                  total_amounts=total_amounts).apply(full_table)
    np.testing.assert_array_almost_equal(transformed.values, mtx / mtx.sum(axis=0) * [0.5, 0.3])


def test_TotalAmountNormalizer_correct_results_on_sparse():
    mtx = np.array([[4, 2],
                    [0, 3],
                    [5, 0]])
    full_table = pd.DataFrame({'A': pd.arrays.SparseArray(mtx[:, 0]),
                               'B': pd.arrays.SparseArray(mtx[:, 1])})

    total_amounts = {'A': 0.5, 'B': 0.3}
    transformed = transform.TotalAmountNormalizer(full_table=full_table,
                                                  total_amounts=total_amounts).apply(full_table)
    np.testing.assert_array_almost_equal(transformed.values, mtx / mtx.sum(axis=0) * [0.5, 0.3])
