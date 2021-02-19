#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform least-squares k-Seq fitting"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.joinpath('src')))

from argparse import ArgumentParser, RawTextHelpFormatter
from yutility import logging
import numpy as np


def kA(params):
    """Calculate combined parameter kA"""
    return params[0] * params[1]


def get_higher_order_abundant():
    table = 'filtered_reacted_frac_all_input_median'


def read_table(seq_data, table_name, fit_top_n=None, inverse_weight=False):
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
    seq_data = read_pickle(seq_data)
    work_table = getattr(seq_data.table, table_name)
    work_table = work_table.loc[work_table.sum(axis=1).sort_values(ascending=False).index]
    x_data = seq_data.x_values[work_table.columns]

    if fit_top_n is not None and fit_top_n > 0:
        work_table = work_table.iloc[:fit_top_n]

    if inverse_weight is True:
        count_table = seq_data.table.original
        sigma = count_table.loc[work_table.index, work_table.columns]
        sigma = sigma + 0.5
    else:
        sigma = None

    return work_table, x_data, sigma


def main(seq_data, table_name, fit_top_n=None, exclude_zero=False, inverse_weight=False,
         bootstrap_num=None, bs_record_num=None, bs_method='data', stratified_grouper=None,
         convergence_num=0,
         core_num=1, large_data=False, output_dir=None,
         overwrite=False, note=None, rnd_seed=23):
    """Main function for fitting"""

    from k_seq.estimate import BatchFitter
    from k_seq.model.kinetic import BYOModel

    work_table, x_data, sigma = read_table(seq_data=seq_data, table_name=table_name,
                                           fit_top_n=fit_top_n, inverse_weight=inverse_weight)
    if bs_method.lower() == 'stratified':
        try:
            grouper = getattr(seq_data.grouper, stratified_grouper).group
        except:
            logging.error('Can not find grouper for stratified bootstrapping', error_type=ValueError)
    else:
        grouper = None

    logging.info(f'exclude_zero: {exclude_zero}')
    logging.info(f'inverse_weight: {inverse_weight}')
    logging.info(f'fit_top_n: {fit_top_n}')
    logging.info(f'large_data: {large_data}')
    logging.info(f'convergence: {convergence_num > 0}')
    logging.info(f'bootstrap: {bootstrap_num > 0}')

    batch_fitter = BatchFitter(
        y_dataframe=work_table, x_data=x_data, sigma=sigma, bounds=[[0, 0], [np.inf, 1]], metrics={'kA': kA},
        model=BYOModel.reacted_frac(broadcast=False), exclude_zero=exclude_zero, grouper=grouper,
        bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method,
        bs_stats={},
        conv_reps=convergence_num,
        conv_init_range=((0, 10), (0, 1)),
        conv_stats={},
        large_dataset=True,
        note=note,
        rnd_seed=rnd_seed
    )
    stream_to = output_dir if large_data else None
    batch_fitter.fit(parallel_cores=core_num, point_estimate=True,
                     bootstrap=bootstrap_num > 0, convergence_test=convergence_num > 0,
                     stream_to=stream_to, overwrite=overwrite)

    batch_fitter.summary(save_to=f'{output_dir}/fit_summary.csv')
    batch_fitter.save_model(output_dir=output_dir, results=True, bs_record=False, tables=True)


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
                        help="""Name of reacted fraction table to use""")
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
    parser.add_argument('--pkg_path', type=str, default=None,
                        help='If use local k-seq package')
    parser.add_argument('--note', type=str, default='',
                        help='Extra note for fitting')

    # Bootstrap
    parser.add_argument('--bootstrap_num', type=int, default=0,
                        help='Number of bootstraps to perform, zero or negative means no bootstrap')
    parser.add_argument('--bs_record_num', type=int, default=-1,
                        help='Number of bootstrap results to save, save all if negative')
    parser.add_argument('--bs_method', choices=['pct_res', 'data', 'stratified'], default='data',
                        help='Resample methods for bootstrapping')
    parser.add_argument('--stratified_grouper', default=None,
                        help='Name of grouper under `seq_data.grouper` for stratified bootstrapping')

    # Convergence, repeated fitting
    parser.add_argument('--convergence_num', type=int, default=0,
                        help='Number of repeated fitting on whole data for convergence test')

    # fitting variations
    parser.add_argument('--exclude_zero', dest='exclude_zero', default=False, action='store_true',
                        help='If exclude zero data in fitting')
    parser.add_argument('--inverse_weight', dest='inverse_weight', default=False, action='store_true',
                        help='Use counts (with pseudo counts 0.5) as the sigma in fitting')

    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    pkg_path = args.pop('pkg_path', None)
    if pkg_path and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

    from k_seq.utility.file_tools import dump_json, check_dir
    check_dir(args['output_dir'])
    dump_json(obj=args, path=f"{args['output_dir']}/config.json")
    logging.add_console_handler()
    logging.add_file_handler(f"{args['output_dir']}/app.log")
    logging.set_level('info')
    logging.info(f"Log stream to {args['output_dir']}/app.log")
    from yutility import Timer
    with Timer():
        sys.exit(main(**args))
