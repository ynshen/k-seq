#!/usr/bin/python3
import sys


def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)
    print(f'{obj} Saved to {path}')


def main(table_path, fit_partial, bootstrap_num, bs_return_num, bs_method, core_num, output_dir, **kwargs):
    from k_seq.estimator.least_square import BatchFitter
    from k_seq.model.kinetic import BYOModel
    from k_seq.utility.file_tools import read_pickle

    import numpy as np

    seq_table = read_pickle(table_path)
    if fit_partial > 0:
        seq_test = seq_table.table..index.values[:int(fit_partial)]
    else:
        seq_test = None

    batch_fitter = BatchFitter(
        table=seq_table.reacted_frac_filtered, x_values=seq_table.x_values, bounds=[[0, 0], [np.inf, 1]],
        model=BYOModel.func_react_frac_no_slope, seq_to_fit=seq_test,
        bootstrap_num=bootstrap_num, bs_return_num=bs_return_num, bs_method=bs_method
    )
    batch_fitter.fit(deduplicate=True, parallel_cores=core_num)

    from pathlib import Path

    if Path(output_dir).is_dir():
        if fit_partial <= 0:
            name = f'bs-{bootstrap_num}_mtd-{bs_method}_c-{core_num}/'
        else:
            name = f'first-{int(fit_partial)}_bs-{bootstrap_num}_mtd-{bs_method}_c-{core_num}/'
        output_dir = Path(output_dir + '/' + name)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    batch_fitter.summary(save_to=f'{output_dir}/fit_summary.csv')
    save_pickle(batch_fitter, path=f'{output_dir}/fitter.pkl')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Individual least squared kinetic model fitting')
    parser.add_argument('--table_path', '-t', type=str, default='./byo_doped.pkl', help='Path to input seq table')
    parser.add_argument('--fit_partial', '-p', type=int, default=-1,
                        help='Select top p sequences to fit, fit all seq if p is negative')
    parser.add_argument('--table_name', '-t', type=str, default='')
    parser.add_argument('--bootstrap_num', '-n', type=int, default=0,
                        help='Number of bootstraps to perform')
    parser.add_argument('--bs_return_num', '-r',type=int, default=-1,
                        help='Number of bootstrap results to save, save all if negative')
    parser.add_argument('--bs_method', '-m', choices=['pct_res', 'data', 'stratified'], default='pct_res',
                        help='Bootstrap method')
    parser.add_argument('--core_num', '-c', type=int,
                        help='Number of process to use in parallel')
    parser.add_argument('--output_dir', '-o', type=str, default='./')
    parser.add_argument('--pkg_path', type=str, default='../')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    if args['pkg_path'] not in sys.path:
        sys.path.insert(0, args['pkg_path'])
    sys.exit(main(**args))
