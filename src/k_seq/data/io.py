"""
This module contains the methods for input and output
"""

# from .pre_processing import SequenceSet


def read_count_file(file_dirc):
    """Read single count file generated from Galaxy

    Args:
        file_dirc (str): full directory to the count file

    Returns:
        unique_seqs (int): number of unique sequences in the count file
        total_counts (int): number of total reads in the count file
        sequences (dict): {sequence: count}
    """

    with open(file_dirc, 'r') as file:
        unique_seqs = int([elem for elem in next(file).strip().split()][-1])
        total_counts = int([elem for elem in next(file).strip().split()][-1])
        next(file)
        sequences = {}
        for line in file:
            seq = line.strip().split()
            sequences[seq[0]] = int(seq[1])
    return unique_seqs, total_counts, sequences


def export_to_csv():
    pass