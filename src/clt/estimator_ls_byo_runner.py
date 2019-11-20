#!/usr/bin/python3
import sys


def kA(params):
    return params[0] * params[1]

def read_table(seq_table=None, table_name=None, simu_data=None, fit_partial=-1):
    """Parse data table source SeqTable or a Simu Folder

    Args:
        seq_table (str): path to a SeqTable instance with x_value
        table_name (str): the table to use in SeqTable

    Returns:
        work_table (pd.DataFrame): the work table contains sequences to fit
        x_data (list): list of x values (BYO concentration), same order as samples in work_table
    """
    from k_seq.utility.file_tools import read_pickle
    from pathlib import Path

    if seq_table is not None:
        # input is seq_table
        seq_table = read_pickle(seq_table)
        work_table = getattr(seq_table, table_name)
        work_table = work_table.loc[work_table.sum(axis=1).sort_values(ascending=False).index]
        x_data = seq_table.x_values[work_table.columns]
    elif simu_data is not None:
        # input is simu data folder
        # TODO: realize following use SeqTable
        import pandas as pd
        count_table = pd.read_csv(simu_data + '/Y.csv', index_col='seq')
        x_data = pd.read_csv(simu_data + '/x.csv', index_col='param').loc['c']
        input_samples = list(x_data[x_data < 0].index)
        reacted_samples = list(x_data[x_data >= 0].index)
        total_dna = pd.read_csv(simu_data + 'dna_amount.csv', header=None, index_col=0)[1]
        amount_table = (count_table / count_table.sum(axis=0) * total_dna)
        input_base = amount_table[input_samples].median(axis=1)
        work_table = amount_table[input_base > 0][reacted_samples].divide(input_base[input_base > 0], axis=0)
        work_table = work_table.loc[work_table.sum(axis=1).sort_values(ascending=False).index]
        work_table = work_table[work_table.sum(axis=1) > 0]
    else:
        raise ValueError('Indicate seq_table or simu_data')

    if fit_partial > 0:
        work_table = work_table.iloc[:fit_partial]

    return work_table, x_data


def main(seq_table=None, table_name=None, simu_data=None, fit_partial=-1,
         bootstrap_num=None, bs_record_num=None, bs_method='data', core_num=1, output_dir=None, **kwargs):
    from k_seq.estimator.least_square import BatchFitter
    from k_seq.model.kinetic import BYOModel
    import numpy as np

    work_table, x_data = read_table(seq_table=seq_table, table_name=table_name, simu_data=simu_data,
                                    fit_partial=fit_partial)

    if bs_method.lower() == 'stratified':
        raise NotImplementedError()
        try:
            grouper = seq_table.grouper.byo.group
        except:
            raise ValueError('Can not find grouper for stratified bootstrapping')

    batch_fitter = BatchFitter(
        table=work_table, x_data=seq_table.x_values, bounds=[[0, 0], [np.inf, 1]], metrics={'kA': kA},
        model=BYOModel.func_react_frac,
        bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method
    )
    batch_fitter.fit(deduplicate=True, parallel_cores=core_num)

    from pathlib import Path

    if Path(output_dir).is_dir():
        if fit_partial <= 0:
            name = f'bs_{int(bootstrap_num)}-mtd_{bs_method}-core_{core_num}/'
        else:
            name = f'first_{int(fit_partial)}-bs_{bootstrap_num}-mtd_{bs_method}-core_{core_num}/'
        output_dir = Path(output_dir + '/' + name)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    batch_fitter.summary(save_to=f'{output_dir}/fit_summary.csv')
    batch_fitter.save_model(model_path=f'{output_dir}/model.pkl',
                            result_path=f'{output_dir}/results.pkl',
                            table_path=f'{output_dir}/table.pkl')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Individual least squared kinetic model fitting')
    parser.add_argument('--pkg_path', type=str, default='./')
    parser.add_argument('--simu_data', type=str,
                        help='Path to folder of simulated data')
    parser.add_argument('--seq_table', '-T', type=str, help='Path to input seq table')
    parser.add_argument('--table_name', '-t', type=str, help='table to use')
    parser.add_argument('--fit_partial', '-p', type=int, default=-1,
                        help='Select top p sequences to fit, fit all seq if p is negative')
    parser.add_argument('--bootstrap_num', '-n', type=int, default=0,
                        help='Number of bootstraps to perform')
    parser.add_argument('--bs_record_num', '-r',type=int, default=-1,
                        help='Number of bootstrap results to save, save all if negative')
    parser.add_argument('--bs_method', '-m', choices=['pct_res', 'data', 'stratified'], default='pct_res',
                        help='Bootstrap method')
    parser.add_argument('--core_num', '-c', type=int,
                        help='Number of process to use in parallel')
    parser.add_argument('--output_dir', '-o', type=str, default='./')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    if args['pkg_path'] not in sys.path:
        sys.path.insert(0, args['pkg_path'])
    sys.exit(main(**args))
