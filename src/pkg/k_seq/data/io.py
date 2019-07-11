"""
This module contains the methods for input and output
"""

# from .pre_processing import SequenceSet


def read_count_file(file_path, as_dict=False):
    """Read a single count file generated from Chen lab's customized scripts

    Count file format:
    ::
        number of unique sequences = 2825
        total number of molecules = 29348173

        AAAAAAAACACCACACA               2636463
        AATATTACATCATCTATC              86763
        ...

    Args:
        file_path (`str`): full directory to the count file
        dict ('bool'): return a dictionary instead of a `pd.DataFrame`

    Returns:
        unique_seqs (`int`): number of unique sequences in the count file
        total_counts (`int`): number of total reads in the count file
        sequence_counts (`pd.DataFrame`): with `sequence` as index and `counts` as the first column
    """
    import pandas as pd

    with open(file_path, 'r') as file:
        unique_seqs = int([elem for elem in next(file).strip().split()][-1])
        total_counts = int([elem for elem in next(file).strip().split()][-1])
        next(file)
        sequence_counts = {}
        for line in file:
            seq = line.strip().split()
            sequence_counts[seq[0]] = int(seq[1])

    if as_dict:
        return unique_seqs, total_counts, sequence_counts
    else:
        return unique_seqs, total_counts, pd.DataFrame.from_dict(sequence_counts, orient='index', columns=['counts'])

def export_to_csv():
    pass