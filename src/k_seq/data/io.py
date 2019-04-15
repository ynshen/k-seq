"""
This module contains the methods for input and output
"""

def read_count_file(file_dirc):
    """
    read count file generated by Galaxy
    :param file_dirc: string, full directory to the count file
    :return: a tuple of (unique_seqs, total_counts, sequences)
             unique_seqs: int, number of unique sequences in the count file
             total_counts: int, number of total reads in the count file
             sequences: dictionary, {sequence: count}
    """

    with open(file_dirc, 'r') as file:
        unique_seqs = int([elem for elem in next(file).strip().split()][-1])
        total_counts = int([elem for elem in next(file).strip().split()][-1])
        next(file)
        sequences = {}
        for line in file:
            seq = line.strip().split()
            sequences[seq[0]] = int(seq[1])
    return (unique_seqs, total_counts, sequences)