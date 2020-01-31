#!usr/bin/python3

from yuning_util.dev_mode import DevMode
dev_mode = DevMode(pkg='k-seq')
dev_mode.on()

from k_seq.data.count_data import CountData
from k_seq.utility.log import Timer
import sys
import logging


def main(output_dir, seq_table=None, table_name=None, input_pools=None,
         simu_data_path=None, solver='SCS', notes=None):

    from k_seq.utility.file_tools import dump_pickle
    from k_seq.estimator.count_model_mle_cvxpy import get_convex_model, run_solver
    import cvxpy as cp
    from pathlib import Path
    import pandas as pd

    if seq_table is not None:
        count_data = CountData.from_SeqTable(seq_table=seq_table, table_name=table_name,
                                             input_pools=input_pools, note=notes)
    elif simu_data_path is not None:
        count_data = CountData.from_simu_path(path=simu_data_path, input_pools=input_pools, note=notes)
    else:
        logging.error("Please indicate either seq_table and table_name or simu_data_path")
        sys.exit(1)

    prob, params = get_convex_model(count_data)
    logging.info(f'Problem created, start solving with {solver}')

    if isinstance(solver, str):
        if solver.upper() == 'SCS':
            solver = cp.SCS
        if solver.upper() == 'MOSEK':
            solver = cp.MOSEK

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    solution = prob.solve(solver, verbose=True)

    logging.info('Problem solving finished. Status unknown')
    dump_pickle(prob, path=output_dir + '/problem.pkl')
    dump_pickle(pd.DataFrame({'a': params.a.value, 'k': params.k.value, 'p0': params.p0.value}),
                path=output_dir + '/parameters.pkl')
    logging.info(f'Optimization results saved to {output_dir}')


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Run count model MLE using cvxpy')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to save outputs')
    parser.add_argument('--seq_table', '-t', default=None,
                        help='path to pickled seqTable object')
    parser.add_argument('--table_name', '-n', default=None,
                        help='table name if use seq_table')
    parser.add_argument('--input_pools', '-i', default=None,
                        help="Name for input samples")
    parser.add_argument('--simu_data_path', '-s', default=None,
                        help='Path to the folder of simulated data')
    parser.add_argument('--solver', '-m', default='SCS',
                        help='Which solver to use, support SCS and MOSEK')
    parser.add_argument('--notes', default=None,
                        help='Optional notes for data')

    args = parser.parse_args()
    from pathlib import Path
    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir(parents=True)
        logging.basicConfig(filename=f"{args.output_path}/app_run.log",
                            format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.DEBUG,
                            filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(f"output_path {args.output_path} does not exist...created")
    else:
        logging.basicConfig(filename=f"{args.output_path}/app_run.log",
                            format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.DEBUG,
                            filemode='w')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(f"output_path {args.output_path} found, logging into this folder")

    with open(f"{args.output_path}/config.txt", 'w') as handle:
        import json
        json.dump(obj=vars(args), fp=handle)
        logging.info(f"App run config saved to {args.output_path}/config.txt")

    return args


if __name__ == '__main__':

    args = parse_args()

    with Timer():
        main(output_dir=args.output_path,
             seq_table=args.seq_table,
             table_name=args.table_name,
             input_pools=args.input_pools,
             simu_data_path=args.simu_data_path,
             solver=args.solver,
             notes=args.notes)
