#!/usr/bin/python3

"""This script contains the code to generate the simulated count data from real data
"""

import logging
from k_seq.data import simu

def parse_args():
    """Parse arguments"""
   import argparse

   parser = argparse.ArgumentParser(description="Simulate the count dataset from real data")
   parser.add_argument()
   parser.add_argument('--out', '-o', type=str, default='./simulated-dataset',
                       help="Output path for simulated dataset")






def main():
    """Main function to run


    Returns:

    """

    _ = simu.simulate_from_sample(
        sample_table=simu.get_sample_table(
            seq_table='byo-doped',
            estimation='/mnt/storage/projects/k-seq/working/byo_doped/least_square/all-seq-point-est_2019-11/table-spike_in_bs-num-0_bs-mtd-data_no-zero-false_inv-weight-false_core-40/fit_summary.csv'),
        seq_num=1e3,
        dna_amount_error=simu.dna_amount_error,
        save_to='/mnt/storage/projects/k-seq/datasets/simulated/on_doped_s1000_d40/'

    )



if __name__ == '__main__':
    args = parse_args()
    main(**args)
