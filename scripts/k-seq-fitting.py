#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform least-squares k-Seq fitting"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.joinpath('src')))

from argparse import ArgumentParser, RawTextHelpFormatter
from yutility import logging, Timer
from k_seq.utility.file_tools import dump_json, check_dir
import numpy as np


def kA(params):
    """Calculate combined parameter kA"""
    return params[0] * params[1]


def read_table():
    """import SeqTable to fit from SeqData

    Args:
        seq_data (str): path to a SeqData instance (with x_value)
        table_name (str): name of SeqTable to use in SeqData
        fit_top_n (int): if only fit the top n sequences in the seq_table. Fit all sequences if None
        inverse_weight (bool): if weight the data by the inverse of their counts (sigma = counts + 0.5)

    Returns:
        work_table (pd.DataFrame): the work seq_table contains sequences to fit
        x_data (list): list of x values (BYO concentration), same order as samples (col) in work_table
        sigma (pd.DataFrame): sigma same as counts + 0.5. None if inverse_weight is False
    """

    from k_seq.utility.file_tools import read_pickle

    # input is seq_table
    seq_data = read_pickle(args.seq_data)
    info(f'Load SeqData from {args.seq_data}')
    work_table = getattr(seq_data.table, args.table_name)
    info(f'Use table: {args.table_name}')
    if args.seq_list is not None:
        info(f'Look up seq in {args.seq_list}')
        with open(args.seq_list, 'r') as handle:
            seq_list = handle.read().split('\n')
        print(seq_list[:3])
        found = work_table.index.isin(seq_list)
        info(f"{np.sum(found)}/{len(seq_list)} found")
        work_table = work_table.loc[found]

    work_table = work_table.loc[work_table.sum(axis=1).sort_values(ascending=False).index]
    x_data = seq_data.x_values[work_table.columns]

    if args.fit_top_n is not None and 0 < args.fit_top_n < len(work_table):
        info(f'Fit top {args.fit_top_n} seq')
        work_table = work_table.iloc[:args.fit_top_n]

    if args.inverse_weight is True:
        count_table = seq_data.table.original
        sigma = count_table.loc[work_table.index, work_table.columns]
        sigma = sigma + 0.5
    else:
        sigma = None

    return work_table, x_data, sigma, seq_data


def main():
    """Main function for fitting"""

    from k_seq.estimate import BatchFitter
    from k_seq.model.kinetic import BYOModel

    work_table, x_data, sigma, seq_data = read_table()
    if args.bs_method.lower() == 'stratified':
        try:
            grouper = getattr(seq_data.grouper, args.stratified_grouper).group
        except:
            logging.error('Can not find grouper for stratified bootstrapping', error_type=ValueError)
            sys.exit(1)
    else:
        grouper = None

    logging.info(f'exclude_zero: {args.exclude_zero}')
    logging.info(f'inverse_weight: {args.inverse_weight}')
    logging.info(f'fit_top_n: {args.fit_top_n}')
    logging.info(f'large_data: {args.large_data}')
    logging.info(f'convergence: {args.convergence_num > 0}')
    logging.info(f'bootstrap: {args.bootstrap_num > 0}')

    batch_fitter = BatchFitter(
        y_dataframe=work_table, x_data=x_data, sigma=sigma, bounds=[[0, 0], [np.inf, 1]], metrics={'kA': kA},
        model=BYOModel.reacted_frac(broadcast=False), exclude_zero=args.exclude_zero, grouper=grouper,
        bootstrap_num=args.bootstrap_num, bs_record_num=args.bs_record_num, bs_method=args.bs_method,
        bs_stats={},
        conv_reps=args.convergence_num,
        conv_init_range=((0, 10), (0, 1)),
        conv_stats={},
        large_dataset=True,
        note=args.note,
        rnd_seed=args.seed
    )
    stream_to = args.output_dir if args.large_data else None
    batch_fitter.fit(parallel_cores=args.core_num, point_estimate=True,
                     bootstrap=args.bootstrap_num > 0, convergence_test=args.convergence_num > 0,
                     stream_to=stream_to, overwrite=args.overwrite)

    batch_fitter.summary(save_to=f'{args.output_dir}/fit_summary.csv')
    batch_fitter.save_model(output_dir=args.output_dir, results=True, bs_record=False, tables=True)


def parse_args():
    """Parse arguments"""
    parser = ArgumentParser(
        prog="least-squares k-Seq fitting",
        description="""
        Least-squares fitting for first order kinetic model:
            y = A * (1 - exp(-alpha * t * k * [BYO])) 
        """,
        formatter_class=RawTextHelpFormatter
    )

    # General fitting
    parser.add_argument('--seq_data', '-i', type=str, required=True,
                        help='Path to pickled seq_data object')
    parser.add_argument('--table_name', '-t', type=str, required=True,
                        help="Name of reacted fraction table to use")
    parser.add_argument('--seq_list', type=str,
                        help='List of sequences to fit')
    parser.add_argument('--fit_top_n', '-n', type=int, default=-1,
                        help='Select top n sequences to fit, fit all seq if n is negative')
    parser.add_argument('--output_dir', '-o', type=str, required=True)

    # Control
    parser.add_argument('--large_data', '-L', dest='large_data', default=False, action='store_true',
                        help='If treat as large data and stream fitting results to disk')
    parser.add_argument('--overwrite', dest='overwrite', default=False, action='store_true',
                        help="If overwrite results when streaming")
    parser.add_argument('--core_num', '-c', type=int,
                        help='Number of threads to use in parallel')
    # parser.add_argument('--pkg_path', type=str, default=None,
    #                     help='If use local k-seq package')
    parser.add_argument('--note', type=str, default='',
                        help='Extra note for fitting')

    # Bootstrap
    parser.add_argument('--bootstrap_num', type=int, default=0,
                        help='Number of bootstraps to perform, zero or negative means no bootstrap')
    parser.add_argument('--bs_record_num', type=int, default=-1,
                        help='Number of bootstrap results to save, save all if negative')
    parser.add_argument('--bs_method', choices=['pct_res', 'data', 'stratified'], default='data',
                        help='Resample methods for bootstrapping')
    parser.add_argument('--stratified_grouper', type=str, default=None,
                        help='Name of grouper under `seq_data.grouper` for stratified bootstrapping')

    # Convergence, repeated fitting
    parser.add_argument('--convergence_num', type=int, default=0,
                        help='Number of repeated fitting on whole data for convergence test')

    # fitting variations
    parser.add_argument('--exclude_zero', dest='exclude_zero', default=False, action='store_true',
                        help='If exclude zero data in fitting')
    parser.add_argument('--inverse_weight', dest='inverse_weight', default=False, action='store_true',
                        help='Use counts (with pseudo counts 0.5) as the sigma in fitting')

    parser.add_argument('--seed', type=int, default=23,
                        help='Random seed')

    args = parser.parse_args()
    check_dir(args.output_dir)
    dump_json(obj=vars(args), path=f"{args.output_dir}/config.json")
    args.output_dir = Path(args.output_dir)

    return args


if __name__ == '__main__':

    args = parse_args()
    # pkg_path = args.pkg_path
    # if pkg_path is not None and pkg_path not in sys.path:
    #     sys.path.insert(0, pkg_path)

    logging.add_console_handler()
    logging.add_file_handler(args.output_dir/"LOG")
    logging.set_level('info')
    logging.info(f"Log stream to {args.output_dir/'LOG'})")
    info = logging.info
    with Timer():
        main()
