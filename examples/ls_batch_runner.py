#!/usr/bin/python3

TABLE_PATH = './byo_doped.pkl'
PKG_PATH = '.'
TEST_MODE = False
CORE_NUM = 40
BS_NUM = 0
BS_SAVE_NUM = 0
BS_METHOD = 'pct_res'

import sys
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)


def load_table(table_path):
    from pathlib import Path
    table_path = Path(table_path)
    import pickle
    with open(table_path, 'rb') as handle:
        table = pickle.load(handle)
    return table


def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)
    print(f'{obj} Saved to {path}')


def main():
    from k_seq.estimator.least_square import BatchFitter
    from k_seq.model.kinetic import BYOModel

    # todo: make it CL tool using argparse
    seq_table = load_table(table_path=TABLE_PATH)
    if TEST_MODE:
        seq_test = seq_table.table.index.values[:100]
    else:
        seq_test = None

    batch_fitter = BatchFitter(table=seq_table.reacted_frac_filtered, x_values=seq_table.x_values, model=BYOModel.func_react_frac_no_slope, seq_to_fit=seq_test, bootstrap_num=BS_NUM, bs_return_num=BS_SAVE_NUM, bs_method=BS_METHOD)
    batch_fitter.fit(deduplicate=True, parallel_cores=CORE_NUM)
    batch_fitter.summary(save_to=f'./fitting-res_bs{BS_NUM}_m{BS_METHOD}_c{CORE_NUM}.csv')
    save_pickle(batch_fitter, path=f'./fitter_bs{BS_NUM}_m{BS_METHOD}_c{CORE_NUM}.pkl')


if __name__ == '__main__':
    sys.exit(main())
