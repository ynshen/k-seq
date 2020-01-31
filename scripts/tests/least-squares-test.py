#!/usr/bin/python3
"""
Test script for least-squares fitting

Includes:
  - Read in a pickled `SeqTable` object
  - SingleFitter
    - construct
    - fit (point_est, bootstrap, convergence)
    - stream results to disk
    - reload results to disk
  - TODO: Batch Fitter
"""
import sys


def test_env():
    pkg_path = '../../src'
    seq_table_path = './test-seqtable.pkl'
    if sys.path[0] != pkg_path:
        sys.path.insert(index=0, object=pkg_path)
    return {'seq_table_path': seq_table_path}


def read_seqtable(path):
    """Test SeqTable read function"""

    from k_seq.utility.file_tools import read_pickle
    print(f'Load SeqTable from {path}...', end='')
    seq_table = read_pickle(path)
    print('success!')
    from k_seq.data.seq_table import SeqTable
    assert isinstance(seq_table, SeqTable)
    return seq_table


def single_fitter(seq_table):
    """Test single fitter functions"""
    table = getattr(seq_table, 'table')
    print('-' * 30 + 'Single Fitter Test' + '-' * 30)
    xvalues = getattr(seq_table, 'x_values')



def main(**kwargs):
    seq_table = read_seqtable(kwargs.pop('seq_table_path', None))


    return 0


if __name__ == '__main__':
    sys.exit(main(**test_env()))