#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate BYO enriched pool dataset from count data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.joinpath('src')))


from argparse import ArgumentParser, RawTextHelpFormatter
from k_seq.data.byo_selected import load_byo_selected
from k_seq.utility.file_tools import dump_pickle, check_dir


def main():
    dataset = load_byo_selected(from_count_file=True, count_file_path=args.count_file_dir, norm_path=args.norm_file)
    dump_pickle(dataset, args.output)


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Generate BYO enriched-pool dataset",
        description="Generate a pickled BYO enriched-pool dataset from preprocessed count files",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--count_file_dir', type=str, required=True,
                        help='Path to the directory containing count files')
    parser.add_argument('--norm_file', type=str, required=True,
                        help='Normalization file from total DNA quantification')
    parser.add_argument('--output', type=str, required=True,
                        help='Folder to save `byo-variant.pkl`')
    args = parser.parse_args()
    args.output = Path(args.output).absolute()
    check_dir(args.output)
    args.output = args.output / 'byo-variant.pkl'
    main()
