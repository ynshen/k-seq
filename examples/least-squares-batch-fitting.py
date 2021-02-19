#!/usr/bin/python3
import sys


def kA(params):
    return params[0] * params[1]


def main(table_path, table_name, fit_partial, bootstrap_num, bs_record_num, bs_method, core_num, output_dir, **kwargs):
    from src.k_seq import read_pickle

    seq_table = read_pickle(table_path)
    work_table = getattr(seq_table, table_name)

    if fit_partial > 0:
        seq_test = work_table.index.values[:int(fit_partial)]
    else:
        seq_test = None

    if bs_method.lower() == 'stratified':
        try:
            grouper = seq_table.grouper.byo.group
        except:
            raise ValueError('Can not find grouper for stratified bootstrapping')
    batch_fitter = BatchFitter(
        y_data_batch=work_table, x_data=seq_table.x_values, bounds=[[0, 0], [np.inf, 1]], metrics={'kA': kA},
        model=BYOModel.func_react_frac_no_slope, seq_to_fit=seq_test,
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
                            table_path=f'{output_dir}/seq_table.pkl')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Individual least squared kinetic model fitting')
    parser.add_argument('--table_path', '-T', type=str, default='./byo_doped.pkl', help='Path to input seq seq_table')
    parser.add_argument('--fit_partial', '-p', type=int, default=-1,
                        help='Select top p sequences to fit, fit all seq if p is negative')
    parser.add_argument('--table_name', '-t', type=str, default='table_filtered_reacted_frac', help='seq_table to use')
    parser.add_argument('--bootstrap_num', '-n', type=int, default=0,
                        help='Number of bootstraps to perform')
    parser.add_argument('--bs_record_num', '-r',type=int, default=-1,
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
