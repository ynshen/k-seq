"""Test code for function tools"""

import pandas as pd
import numpy as np
from k_seq.utility import func_tools


def test_check_sparse_works():

    val = [1, 2, np.nan, np.nan, 4]
    series_sparse = pd.Series(pd.arrays.SparseArray(val))
    assert func_tools.is_sparse(series_sparse)
    series_dense = pd.Series(val)
    assert ~func_tools.is_sparse(series_dense)

    df_sparse = pd.DataFrame({"A": pd.arrays.SparseArray(val), "B": pd.arrays.SparseArray(val)})
    assert func_tools.is_sparse(df_sparse)
    df_dense = pd.DataFrame({"A": val, "B": val})
    assert ~func_tools.is_sparse(df_dense)
