#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts to generate reacted fraction simulated dataset and perform k-Seq for model identifiability analysis
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.joinpath('src')))

from yutility import logging, Timer
import pandas as pd
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from k_seq.model.kinetic import BYOModel
from k_seq.estimator.model_ident import ParamMap
from k_seq.utility.file_tools import check_dir

logging.set_level('info')
info = logging.info
debug = logging.debug


def get_args():
    parser = ArgumentParser(
        prog="Simulate analysis for model identifiability",
        description="""
        Analyze model identifiability for sequences with different kinetic parameters in pseudo-first order model.
        Reacted fraction dataset will be simulated and fit using least square fitting.
        
        Sequences have parameters sampled from
            log(k) in (-1, 3)
            log(A) in (-2, 0) or A in (0, 1)
        """,
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--n_seq', '-n', type=int, default=int(1e4),
                        help='Number of sequences to sample from the parameter space')
    parser.add_argument('--x_data', '-x', type=float, nargs='+', required=True,
                        help='A series of x values (e.g. concentration, time) for the simulated exp')
    parser.add_argument('--replicates', type=int, default=1,
                        help='Number of replicates fro each x_data')

    # parser.add_argument('--include_1250', default=False, action='store_true', dest='include_1250',
    #                     help='If include 1250 nM concentration, maximal 250 uM BYO concentration is used without this'
    #                          'flag, and there are 4 replicates for each concentration to balance number of data points')
    # parser.add_argument('--old_x', default=False, action='store_true', dest='old_x',
    #                     help='If use the old x-series used in JACS paper, maximal 250 uM BYO concentration is '
    #                          'used with triplicates for each concentration')

    parser.add_argument('--rel_err', '-e', type=float, default='0.',
                        help='Relative error to applied to simulated reacted fraction'),
    parser.add_argument('--output', '-o', type=str, default='model-ident-test',
                        help='Output directory to save simulated data and fitting results')
    parser.add_argument('--log_A', default=False, action='store_true',
                        help='If sample A on log scale between 0.01 and 1'),
    parser.add_argument('--n_thread', type=int, default=1,
                        help='Number of threads to run in parallel for fitting')
    parser.add_argument('--n_bootstrap', type=int, default=100,
                        help='Number of bootstrapping')
    parser.add_argument('--n_converge', type=int, default=20,
                        help='Number of convergence test from repeated fitting')
    parser.add_argument('--seed', dest='seed', type=int, default=23,
                        help='Random seed')

    args = parser.parse_args()
    args.output = Path(args.output).absolute()
    check_dir(args.output)
    logging.add_file_handler(args.output / 'log.txt')
    info(f'Result is saving to {args.output}')

    args.x_data = pd.Series(data=np.repeat(list(args.x_data), args.replicates),
                            index=np.arange(len(args.x_data * args.replicates)) + 1, name='x_data')
    return args


def kA(param):
    return param[0] * param[1]


def main():
    param1_name = 'k'
    param1_range = (1e-1, 1e3)
    param1_log = True
    param2_name = 'A'
    param2_range = (1e-2, 1) if args.log_A else (0, 1)
    param2_log = args.log_A

    info(f'Simulate dataset with {len(args.x_data)} sample points:\n{args.x_data}')

    fitting_kwargs = {'bounds': ((0, 0), (np.inf, 1)),
                      'metrics': {'kA': kA}}

    conv_map = ParamMap(
        model=BYOModel.reacted_frac(broadcast=False), sample_n=args.n_seq, x_data=args.x_data, save_to=args.output,
        param1_name=param1_name, param1_range=param1_range, param1_log=param1_log,
        param2_name=param2_name, param2_range=param2_range, param2_log=param2_log,
        model_kwargs=None, fitting_kwargs=fitting_kwargs,
        bootstrap_num=args.n_bootstrap, bs_record_num=20, bs_method='rel_res', bs_stats={},
        conv_reps=args.n_converge, conv_init_range=((0, 10), (0, 1)), conv_stats={},
        seed=args.seed
    )

    conv_map.simulate_samples(grid=True, rel_err=args.rel_err)
    conv_map.fit(parallel_cores=args.n_thread, overwrite=True)



if __name__ == '__main__':

    args = get_args()

    with Timer():
        main()
