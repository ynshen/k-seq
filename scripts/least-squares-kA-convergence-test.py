#!/usr/bin/python3
import sys
from yutility import logging, Timer, dev_mode
dev_mode.on('k-seq')
import pandas as pd
import numpy as np


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convergence test for least-squares fitting for simulated sequences within the params uniformly '
                    'sampled form log(k) in (-1, 3) for log(k), and (0, 1) for A.'
                    'Note that reacted fraction is simulated for calculation rather than counts'
    )

    parser.add_argument('--pkg_path', type=str, default=None, dest='pkg_path',
                        help='Path to local k-seq package')
    parser.add_argument('--output_dir', '-o', type=str, dest='output_dir',
                        help='Output directory')
    parser.add_argument('--include_1250', default=False, action='store_true', dest='include_1250',
                        help='If include 1250 nM concentration, maximal 250 uM BYO concentration is used without this'
                             'flag, and there are 4 replicates for each concentration to balance number of data points')
    parser.add_argument('--uniq_seq_num', dest='uniq_seq_num', type=int, default=int(1e4),
                        help='Number of random sequences ')
    parser.add_argument('--conv_reps', dest='conv_reps', type=int, default=20,
                        help='Number of repeated fitting performed for convergence test')
    parser.add_argument('--cores', dest='core', type=int, default=1,
                        help='Number of parallel processes to use')
    parser.add_argument('--seed', dest='seed', type=int, default=23,
                        help='Interger random seed')

    return parser.parse_args()


def kA(param):
    return param[0] * param[1]


def remove_nan(df):
    return df[~df.isna().any(axis=1)]


def spearman(records):
    from scipy import stats
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.spearmanr(records['k'], records['A']).correlation
    else:
        return np.nan


def pearson(records):
    from scipy import stats
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.pearsonr(records['k'], records['A'])[0]
    else:
        return np.nan


def main(sample_n=int(1e4), include_1250=True, conv_reps=20, seed=23, output_dir=None, n_threads=1):
    from k_seq.model.kinetic import BYOModel
    from k_seq.estimator.convergence import ConvergenceMap

    param1_name = 'k'
    param2_name = 'A'
    param1_range = (1e-1, 1e3)
    param2_range = (0, 1)
    conv_init_range = ((0, 1), (0, 1))
    param1_log = True
    param2_log = False
    conv_stats = {'kA_pearson': pearson, 'kA_spearman': spearman}
    if include_1250:
        logging.info('BYO concentration 1250 uM included, we have 15 samples: [1250, 250, 50, 10, 2] * 3')
        x_values = pd.Series(data=np.repeat([1250, 250, 50, 10, 2], 3) * 1e-6, index=np.arange(15) + 1)
    else:
        logging.info('BYO concentration 1250 uM not included, we have 16 samples: [250, 50, 10, 2] * 4')
        x_values = pd.Series(data=np.repeat([250, 50, 10, 2], 4) * 1e-6, index=np.arange(16) + 1)
    model_kwargs = None

    fitting_kwargs = {'bounds': ((0, 0), (np.inf, 1)),
                      'metrics': {'kA': kA}}

    conv_map = ConvergenceMap(
        model=BYOModel.reacted_frac(broadcast=False), sample_n=int(sample_n), conv_reps=conv_reps, x_values=x_values,
        param1_name=param1_name, param1_range=param1_range, param1_log=param1_log,
        param2_name=param2_name, param2_range=param2_range, param2_log=param2_log,
        save_to=output_dir,
        conv_metric='A_range', conv_stats=conv_stats, conv_init_range=conv_init_range,
        model_kwargs=model_kwargs, fitting_kwargs=fitting_kwargs,
        seed=seed
    )

    conv_map.fit(n_threads=n_threads)


if __name__ == '__main__':

    args = get_args()
    if args.pkg_path and args.pkg_path not in sys.path:
        sys.path.insert(0, args.pkg_path)
    from k_seq.utility.file_tools import check_dir
    check_dir(args.output_dir)
    logging.add_file_handler(f"{args.output_dir}/app_run.log")
    logging.info(f'Results will be saved to {args.output_dir}')
    with Timer():
        main(sample_n=args.uniq_seq_num, include_1250=args.include_1250,
             conv_reps=args.conv_reps, n_threads=args.core, output_dir=args.output_dir, seed=args.seed)
