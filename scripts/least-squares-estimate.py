#!/usr/bin/python3
import sys
import logging


def kA(params):
    """Calculate kA metric"""
    return params[0] * params[1]


def read_table(seq_table=None, table_name=None, fit_partial=-1, inverse_weight=False):
    """Parse data table source SeqTable

    Args:
        seq_table (str): path to a SeqTable instance with x_value
        table_name (str): the table to use in SeqTable, default 'table'
        fit_partial (int): if fit the first k sequences in the table. Fit all sequences if negative
        inverse_weight (bool): if weight the data by the inverse of their counts (sigma = counts + 0.5)

    Returns:
        work_table (pd.DataFrame): the work table contains sequences to fit
        x_data (list): list of x values (BYO concentration), same order as samples in work_table
        sigma (pd.DataFrame): sigma same as counts + 0.5 or None if not weighted
    """

    from k_seq.utility.file_tools import read_pickle

    # input is seq_table
    seq_table = read_pickle(seq_table)
    count_table = seq_table.table
    work_table = getattr(seq_table, table_name)
    work_table = work_table.loc[work_table.sum(axis=1).sort_values(ascending=False).index]
    x_data = seq_table.x_values[work_table.columns]

    if fit_partial > 0:
        work_table = work_table.iloc[:fit_partial]

    if inverse_weight is True:
        sigma = count_table.loc[work_table.index, work_table.columns]
        sigma = sigma + 0.5
    else:
        sigma = None

    return work_table, x_data, sigma


def main(seq_table=None, table_name=None, fit_partial=-1, exclude_zero=False, inverse_weight=False,
         bootstrap_num=None, bs_record_num=None, bs_method='data', core_num=1, deduplicate=False, output_dir=None,
         stream=False, overwrite=False):
    """Main function
    """

    from k_seq.estimator.least_square import BatchFitter
    from k_seq.model.kinetic import BYOModel
    import numpy as np

    work_table, x_data, sigma = read_table(seq_table=seq_table, table_name=table_name,
                                           fit_partial=fit_partial, inverse_weight=inverse_weight)
    if bs_method.lower() == 'stratified':
        try:
            grouper = seq_table.grouper.byo.group
        except:
            raise ValueError('Can not find grouper for stratified bootstrapping')
    else:
        grouper = None

    logging.info(f'exclude_zero: {exclude_zero}')
    logging.info(f'inverse_weight: {inverse_weight}')
    logging.info(f'deduplicate: {deduplicate}')
    batch_fitter = BatchFitter(
        y_data_batch=work_table, x_data=x_data, sigma=sigma, bounds=[[0, 0], [np.inf, 1]], metrics={'kA': kA},
        model=BYOModel.reacted_frac, exclude_zero=exclude_zero, grouper=grouper,
        bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method,
    )
    stream_to_disk = f"{output_dir}/results" if stream else None
    batch_fitter.fit(deduplicate=deduplicate, parallel_cores=core_num,
                     stream_to_disk=stream_to_disk, overwrite=overwrite)

    return batch_fitter

    batch_fitter.summary(save_to=f'{output_dir}/fit_summary.csv')
    if stream:
        batch_fitter.save_model(output_dir=output_dir, results=True, bs_results=False, sep_files=True, tables=True)
    else:
        batch_fitter.save_model(output_dir=output_dir, results=True, bs_results=True, sep_files=False, tables=True)


def parse_args():
    """Parse arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Least-squares kinetic model fitting')
    parser.add_argument('--pkg_path', type=str, default=None,
                        help='Path to customize k-seq package')
    parser.add_argument('--seq_table', '-T', type=str,
                        help='Path to input seq_table object')
    parser.add_argument('--table_name', '-t', type=str,
                        help='table name in seq_table to use')
    parser.add_argument('--fit_partial', '-p', type=int, default=-1,
                        help='Select top p sequences to fit, fit all seq if p is negative')
    parser.add_argument('--exclude_zero', dest='exclude_zero', default=False, action='store_true',
                        help='If exclude zero data in fitting')
    parser.add_argument('--inverse_weight', dest='inverse_weight', default=False, action='store_true',
                        help='Apply counts (pseudo counts 0.5) as the sigma in fitting')
    parser.add_argument('--bootstrap_num', '-n', type=int, default=0,
                        help='Number of bootstraps to perform')
    parser.add_argument('--bs_record_num', '-r', type=int, default=-1,
                        help='Number of bootstrap results to save, save all if negative')
    parser.add_argument('--bs_method', '-m', choices=['pct_res', 'data', 'stratified'], default='pct_res',
                        help='Bootstrap method')
    parser.add_argument('--deduplicate', dest='deduplicate', default=False, action='store_true',
                        help='If deduplicate seq with same data')
    parser.add_argument('--stream-results', dest='stream', default=False, action='store_true',
                        help="If stream fitting results to disk")
    parser.add_argument('--overwrite', dest='overwrite', default=False, action='store_true',
                        help="If overwrite results when streaming")
    parser.add_argument('--core_num', '-c', dest='core_num', type=int, help='Number of process to use in parallel')
    parser.add_argument('--output_dir', '-o', type=str, default='./')

    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()
    pkg_path = args.pop('pkg_path', '')
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    from k_seq.utility.file_tools import to_json, check_dir
    check_dir(args['output_dir'])
    to_json(obj=args, path=f"{args['output_dir']}/config.json")
    logging.basicConfig(filename=f"{args['output_dir']}/app_run.log",
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
                        filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Standard IO added')
    from k_seq.utility.log import Timer
    with Timer():
        sys.exit(main(**args))
