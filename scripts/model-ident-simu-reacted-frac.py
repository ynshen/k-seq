#!/usr/bin/python3

# Convergence test v3

import sys
from yutility import logging, Timer, dev_mode
dev_mode.on('k-seq')
import pandas as pd
import numpy as np
from scipy import stats


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Convergence test for least-squares fitting for simulated sequences within the params uniformly '
                    'sampled form log(k) in (-1, 3) for log(k)'
                    '(0, 1) for A if not log_A, (-2, 0) for log(A) if log_A'
                    'Note that reacted fraction is simulated for calculation rather than counts'
    )

    parser.add_argument('--pkg', type=str, default=None, dest='pkg_path',
                        help='Path to local k-seq package')
    parser.add_argument('--output', '-o', type=str, dest='output_dir',
                        help='Output directory')
    parser.add_argument('--include_1250', default=False, action='store_true', dest='include_1250',
                        help='If include 1250 nM concentration, maximal 250 uM BYO concentration is used without this'
                             'flag, and there are 4 replicates for each concentration to balance number of data points')
    parser.add_argument('--old_x', default=False, action='store_true', dest='old_x',
                        help='If use the old x-series used in JACS paper, maximal 250 uM BYO concentration is '
                             'used with triplicates for each concentration')
    parser.add_argument('--log_A', dest='log_A', default=False, action='store_true',
                        help='If sample A on log scale between 0.01 and 1'),
    parser.add_argument('--pct_error', dest='pct_error', type=float, default=None,
                        help='percent error applied on simulated reacted fraction'),
    parser.add_argument('-n', '--uniq_seq_num', dest='uniq_seq_num', type=int, default=int(1e4),
                        help='Number of random sequences ')
    parser.add_argument('--cores', dest='core', type=int, default=1,
                        help='Number of parallel processes to use')
    parser.add_argument('--seed', dest='seed', type=int, default=23,
                        help='Integer random seed')

    return parser.parse_args()


def kA(param):
    return param[0] * param[1]


def remove_nan(df):
    return df[~df.isna().any(axis=1)]


def normality_dAgostino_test(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.normaltest(records['kA'], nan_policy='propagate').pvalue
    else:
        return np.nan


def normality_shapiro_test(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.shapiro(records['kA'])[0]
    else:
        return np.nan


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


def spearman_log(records):
    from scipy import stats
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.spearmanr(np.log10(records['k']), np.log10(records['A'])).correlation
    else:
        return np.nan


def pearson_log(records):
    from scipy import stats
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.pearsonr(np.log10(records['k']), np.log10(records['A']))[0]
    else:
        return np.nan


def kendall_log(records):
    from scipy import stats
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.kendalltau(np.log10(records['k']), np.log10(records['A'])).correlation
    else:
        return np.nan


def main(sample_n=int(1e4), include_1250=True, old_x=False, seed=23, output_dir=None,
         log_A=False, pct_error=None, n_threads=1):
    from k_seq.model.kinetic import BYOModel
    from k_seq.estimator.convergence import ConvergenceMap

    param1_name = 'k'
    param2_name = 'A'
    param1_range = (1e-1, 1e3)
    param1_log = True
    if log_A:
        param2_range = (1e-2, 1)
        param2_log = True
    else:
        param2_range = (0, 1)
        param2_log = False

    if old_x:
        logging.info('BYO concentration 1250 uM not included, we have 13 samples: [250, 50, 10, 2] * 3')
        x_values = pd.Series(data=np.repeat([250, 50, 10, 2], 3) * 1e-6, index=np.arange(12) + 1)
    else:
        if include_1250:
            logging.info('BYO concentration 1250 uM included, we have 15 samples: [1250, 250, 50, 10, 2] * 3')
            x_values = pd.Series(data=np.repeat([1250, 250, 50, 10, 2], 3) * 1e-6, index=np.arange(15) + 1)
        else:
            logging.info('BYO concentration 1250 uM not included, we have 16 samples: [250, 50, 10, 2] * 4')
            x_values = pd.Series(data=np.repeat([250, 50, 10, 2], 4) * 1e-6, index=np.arange(16) + 1)

    model_kwargs = None

    fitting_kwargs = {'bounds': ((0, 0), (np.inf, 1)),
                      'metrics': {'kA': kA}}

    bs_stats = {'norm_dAgostino': normality_dAgostino_test,
                'norm_shapiro': normality_shapiro_test,
                'spearman_log': spearman_log,
                'pearson_log': pearson_log,
                'kendall_log': kendall_log}

    conv_stats = {'spearman': spearman,
                  'pearson': pearson,
                  'spearman_log': spearman_log,
                  'pearson_log': pearson_log,
                  'kendall_log': kendall_log}

    conv_map = ConvergenceMap(
        model=BYOModel.reacted_frac(broadcast=False), sample_n=int(sample_n), x_values=x_values, save_to=output_dir,
        model_kwargs=model_kwargs,
        param1_name=param1_name, param1_range=param1_range, param1_log=param1_log,
        param2_name=param2_name, param2_range=param2_range, param2_log=param2_log,
        bootstrap_num=1000,
        fitting_kwargs=fitting_kwargs,
        bs_stats=bs_stats,
        conv_init_range=((0, 10), (0, 1)),
        conv_stats=conv_stats, seed=seed
    )

    conv_map.simulate_samples(grid=True, pct_error=pct_error)
    conv_map.fit(parallel_cores=n_threads, overwrite=True)


if __name__ == '__main__':

    args = get_args()
    if args.pkg_path and args.pkg_path not in sys.path:
        sys.path.insert(0, args.pkg_path)
    from k_seq.utility.file_tools import check_dir
    check_dir(args.output_dir)
    logging.add_file_handler(f"{args.output_dir}/app_run.log")
    logging.info(f'Results will be saved to {args.output_dir}')
    with Timer():
        main(sample_n=args.uniq_seq_num, old_x=args.old_x, include_1250=args.include_1250,
             log_A=args.log_A, pct_error=args.pct_error,
             n_threads=args.core, output_dir=args.output_dir, seed=args.seed)
