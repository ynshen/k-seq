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


def test_Grouper_type1_works():
    from k_seq.data.grouper import Grouper

    data = pd.DataFrame([[0, 1, 2], [1, 2, 3]], index=['r1', 'r2'], columns=['c1', 'c2', 'c3'])
    data2 = pd.DataFrame([[45, 23, 34], [45, 87, 11]], index=['r1', 'r2'], columns=['c1', 'c2', 'c3'])
    gr = Grouper(target=data, group=['c1', 'c3'], axis=1)
    assert gr.group == ['c1', 'c3']
    pd.testing.assert_frame_equal(gr.split(), data[['c1', 'c3']])
    pd.testing.assert_frame_equal(gr['nonsense'], data[['c1', 'c3']])
    pd.testing.assert_frame_equal(gr(data2), data2[['c1', 'c3']])


def test_Grouper_type2_works():
    from k_seq.data.grouper import Grouper

    data = pd.DataFrame([[0, 1, 2], [1, 2, 3]], index=['r1', 'r2'], columns=['c1', 'c2', 'c3'])
    data2 = pd.DataFrame([[45, 23], [45, 87], [23, 99]], columns=['r1', 'r2'], index=['c1', 'c2', 'c3'])
    gr = Grouper(target=data, group={'a': ['c1', 'c3'], 'b': ['c1', 'c2']}, axis=1)
    assert gr.group == {'a': ['c1', 'c3'], 'b': ['c1', 'c2']}
    split = gr.split()
    pd.testing.assert_frame_equal(split['a'], data[['c1', 'c3']])
    pd.testing.assert_frame_equal(split['b'], data[['c1', 'c2']])
    pd.testing.assert_frame_equal(gr['a'], data[['c1', 'c3']])
    pd.testing.assert_frame_equal(gr(target=data2, group='b', axis=0), data2.loc[['c1', 'c2']])


def test_GrouperCollection_works():
    from k_seq.data.grouper import GrouperCollection
    gc = GrouperCollection(group1=[0, 1, 2], group2={'gp1': [1, 2, 3], 'gp2': [4, 5, 6]})
    assert hasattr(gc, 'group1')
    assert hasattr(gc, 'group2')
    assert gc.group1.group == [0, 1, 2]
    assert gc.group2.group == {'gp1': [1, 2, 3], 'gp2': [4, 5, 6]}


def test_CustomizedFilter_works():
    from k_seq.data.filters import CustomizedFilter

    def filter_fn(df):
        return df.sum(axis=1) > 10

    one_filter = CustomizedFilter(filter_fn)
    df = pd.DataFrame([[0, 1, 2, 3], [9, 1, 2, 23]])
    pd.testing.assert_frame_equal(one_filter(df), df.iloc[1:, :])

