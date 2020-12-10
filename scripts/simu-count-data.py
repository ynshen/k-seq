#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts to generated count data for sequence pool, simulated from experimental measurements/conditions (BYO variant pool)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.joinpath('src')))

from argparse import ArgumentParser, RawTextHelpFormatter
from k_seq.data import simu
from k_seq.utility.file_tools import read_pickle


def main():
    dataset = read_pickle(args.dataset)
    _ = simu.simulate_on_byo_doped_condition_from_exp_results(
        dataset=dataset,
        fitting_res=args.k_seq_result,
        table_name='filtered',
        uniq_seq_num=args.uniq_seq_num,
        total_dna_error_rate=args.total_dna_rel_error,
        sequencing_depth=args.seq_depth,
        save_to=args.output,
        seed=args.seed,
        plot_dist=False
    )


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Simulate count data",
        description="Simulate count data from experimental data and k-Seq results (e.g., BYO variant-pool)",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to experimental dataset (e.g., BYO variant pool)')
    parser.add_argument('--k_seq_result', type=str, required=True,
                        help='Path to k-Seq result (fit_summary.csv) for the experimental dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path to save simulated dataset')
    parser.add_argument('--table_name', type=str, default='filtered',
                        help="Table name in dataset to use for calculating input fraction for each sequence. "
                             "Default 'filtered'")
    parser.add_argument('--uniq_seq_num', type=int, default=int(1e6),
                        help='Number of unique sequences in the simulated pool. Default 10^6 sequences')
    parser.add_argument('--total_dna_rel_error', type=float, default=0.15,
                        help='Relative error rate for simulated total DNA quantification. Default 0.15')
    parser.add_argument('--seq_depth', type=float, default=40,
                        help='Mean seq_depth for the pool. Total number of reads will be seq_depth * uniq_seq_num. '
                             'Default 40')
    parser.add_argument('--seed', type=int, default=23, help='Random seed')
    args = parser.parse_args()

    args.dataset = Path(args.dataset).absolute()
    args.k_seq_result = Path(args.k_seq_result).absolute()
    if args.k_seq_result.is_dir():
        args.k_seq_result = args.k_seq_result / 'fit_summary.csv'
    args.output = Path(args.output).absolute()

    main()
