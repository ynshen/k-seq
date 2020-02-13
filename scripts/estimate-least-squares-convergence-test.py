#!/usr/bin/python3
import sys
import logging
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
    parser.add_argument('--seq_num', dest='seq_num', type=int, default=int(1e4),
                        help='Number of random sequences ')
    parser.add_argument('--reps', dest='reps', type=int, default=20,
                        help='Number of repeated fitting performed for convergence test')
    parser.add_argument('--cores', dest='core', type=int, default=1,
                        help='Number of parallel processes to use')

    return parser.parse_args()


def simu_seq_reacted_frac(param, c, kinetic_model, percent_error=0):
    """Simulate reacted fraction for a single sequence"""

    if isinstance(param, pd.Series):
        param = param.to_dict()
    if isinstance(c, (list, np.ndarray, tuple)):
        c = pd.Series(data=c, index=np.arange(len(c)) + 1)
    reacted_frac = kinetic_model(**param, c=c)
    if percent_error:
        reacted_frac = np.random.normal(loc=reacted_frac, scale=reacted_frac * percent_error)
        reacted_frac[reacted_frac < 0] = 0
    return pd.Series(data=reacted_frac, index=c.index)


def simulate_reacted_frac(df_seq_param=None, c=None, kinetic_model=None, percent_error=0):
    """Simulate reacted franctions for a list of sequences"""

    from functools import partial
    partial_func = partial(simu_seq_reacted_frac, c=c, kinetic_model=kinetic_model, percent_error=percent_error)
    reacted_frac = df_seq_param.apply(partial_func, axis=1)
    return reacted_frac, c, df_seq_param


def kA(params):
    return params[0] * params[1]


def convergence_test(seq, c, reps=20):
    """seq: a row of reacted frac table"""
    from k_seq.estimator import least_square
    from k_seq.model.kinetic import BYOModel

    fitter = least_square.SingleFitter(
        x_data=c, y_data=seq,
        model=BYOModel.reacted_frac, name=seq.name,
        sigma=None, bounds=[[0, 0], [np.inf, 1]], init_guess=None,
        opt_method='trf', exclude_zero=False, metrics={'kA': kA},
        bootstrap_num=0, bs_record_num=0, bs_method='pct_res', curve_fit_params=None, silent=True
    )

    conv_test_res = [fitter._fit() for _ in range(reps)]
    conv_test_res = pd.DataFrame(np.array([list(res['params']) + [res['metrics']['kA']] for res in conv_test_res]),
                                 columns=['k', 'A', 'kA'])
    return seq.name, conv_test_res


def main(n_seq=int(1e4), include_1250=True, reps=20, seed=23, output_dir=None, core=1):
    from k_seq.data.simu import DistGenerators

    df_seq_param = pd.DataFrame(
        {'A': DistGenerators.uniform(low=0, high=1, size=n_seq, seed=seed),
         'k': 10 ** DistGenerators.uniform(low=-1, high=3, size=n_seq)}
    )

    if include_1250:
        logging.info('BYO concentration 1250 uM included, we have 15 samples: [1250, 250, 50, 10, 2] * 3')
        c = pd.Series(data=np.repeat([1250, 250, 50, 10, 2], 3) * 1e-6, index=np.arange(15) + 1)
    else:
        logging.info('BYO concentration 1250 uM not included, we have 16 samples: [250, 50, 10, 2] * 4')
        c = pd.Series(data=np.repeat([250, 50, 10, 2], 4) * 1e-6, index=np.arange(16) + 1)

    logging.info("Start simulation of reacted fractions")
    from k_seq.model.kinetic import BYOModel
    reacted_frac, c, df_seq_param = simulate_reacted_frac(df_seq_param=df_seq_param, c=c,
                                                          kinetic_model=BYOModel.reacted_frac)
    logging.info(f'Start fitting for each seq for {reps} times on {core} parallel processes...')
    import multiprocess as mp
    from functools import partial
    pool = mp.Pool(processes=core)
    test_results = pool.map(func=partial(convergence_test, c=c, reps=reps),
                            iterable=(reacted_frac.loc[ix] for ix in reacted_frac.index))
    test_results = {res[0]: res[1] for res in test_results}

    if output_dir:
        from k_seq.utility.file_tools import dump_pickle
        dump_pickle(obj=df_seq_param, path=f"{output_dir}/seq_param.pkl")
        dump_pickle(obj=c, path=f"{output_dir}/c.pkl")
        dump_pickle(obj=reacted_frac, path=f"{output_dir}/reacted_frac.pkl")
        dump_pickle(obj=test_results, path=f"{output_dir}/convergence_res.pkl")
        logging.info(f'Fitting finished, results were saved to {args.output_dir}')
    else:
        return df_seq_param, c, reacted_frac, test_results


if __name__ == '__main__':

    args = get_args()
    if args.pkg_path and args.pkg_path not in sys.path:
        sys.path.insert(0, args.pkg_path)
    from k_seq.utility.file_tools import check_dir
    check_dir(args.output_dir)
    logging.basicConfig(filename=f"{args.output_dir}/app_run.log",
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'Results will be saved to {args.output_dir}')
    from k_seq.utility.log import Timer
    with Timer():
        main(n_seq=args.seq_num, include_1250=args.include_1250,
             reps=args.reps, core=args.core, output_dir=args.output_dir)

