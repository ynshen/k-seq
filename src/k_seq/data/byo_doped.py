"""Private
BYO doped-pool dataset from Abe's k-seq experiments"""

from yutility import logging


def byo_doped_rename_sample(name):
    """Rename results loaded from raw reads and samples as

    A1/d-A1_S1 --> 1250 μM-1
    ...
    R/R0 --> Unreacted
    """

    if len(name) > 2:
        name = name.split('_')[0].split('-')[-1]

    if 'R' in name:
        return 'Unreacted'
    else:
        concen_mapper = {
            'A': '1250',
            'B': '250',
            'C': '50',
            'D': '10',
            'E': '2'
        }
        return "{} $\\mu$ M-{}".format(concen_mapper[name[0]], name[1])


def load_byo_doped(from_count_file=False, count_file_path=None, doped_norm_path=None, pickled_path=None,
                   pandaseq_joined=True, radius=2):
    """BYO doped pool k-seq datatable contains k-seq results for seqs from doped-pool for BYO aminoacylation,

    this dataset contains following pre-computed table (``.table`` accessor) to use

        - original (4313709, 16): count table contains all sequences detected in any samples and all the samples
        - filtered (3290337, 16): count table with non 21 nt sequences and spike-in sequences filtered
        - reacted_frac_spike_in (764756, 15): reacted fraction for valid seqs quantified by spike-in
        - reacted_frac_qpcr (764756, 15): reacted fraction for valid seqs quantified by qPCR
        - reacted_frac_spike_in_seq_in_all_smpl (22525, 15): only contains seqs with counts >= 1 in all samples
        - reacted_frac_qpcr_seq_in_all_smpl (22525, 15): only contains seqs with counts >= 1 in all samples

    Note:
       By default, sequences within 2 edit distance (including insertion and deletion) of spike-in sequences were
           considered as spike-in seq
    """
    import os
    from .seq_data import SeqData

    if pickled_path:
        BYO_DOPED_PKL = pickled_path
    else:
        BYO_DOPED_PKL = os.getenv('BYO_DOPED_PKL_PANDASEQ', None) if pandaseq_joined else \
            os.getenv('BYO_DOPED_PKL_FASTQJOIN', None)
    if count_file_path:
        BYO_DOPED_COUNT_FILE = count_file_path
    else:
        BYO_DOPED_COUNT_FILE = os.getenv('BYO_DOPED_COUNT_FILE_PANDASEQ', None) if pandaseq_joined else \
            os.getenv('BYO_DOPED_COUNT_FILE_FASTQJOIN', None)
    BYO_DOPED_NORM_FILE = os.getenv('BYO_DOPED_NORM_FILE', None) if doped_norm_path is None \
        else doped_norm_path

    pattern_filter = '_counts.' if pandaseq_joined else 'counts-'
    name_pattern = 'd-[{byo}{exp_rep}]_S{smpl}_counts.txt' if pandaseq_joined else 'counts-d-[{byo}{exp_rep}].txt'

    if from_count_file:
        import numpy as np
        import pandas as pd

        logging.info('Generate SeqData instance for BYO-doped pool...')
        logging.info(f'Importing from {BYO_DOPED_COUNT_FILE}...this could take a couple of minutes...')

        # parse dna amount file, original data is 1/total_dna
        dna_amount = pd.read_table(BYO_DOPED_NORM_FILE, header=None).rename(columns={0: 'dna_inv'})
        dna_amount['total_amounts'] = 1 / dna_amount['dna_inv']
        indices = ['R0']
        for sample in 'ABCDE':
            for rep in range(3):
                indices.append(f'{sample}{rep + 1}')
        dna_amount = {name: dna_amount['total_amounts'][ix] for ix, name in enumerate(indices)}

        byo_doped = SeqData.from_count_files(
            count_files=BYO_DOPED_COUNT_FILE,
            pattern_filter=pattern_filter,
            name_template=name_pattern,
            dry_run=False,
            sort_by='name',
            x_values=np.concatenate((
                np.repeat([1250, 250, 50, 10, 2], repeats=3) * 1e-6,
                np.array([np.nan])), axis=0
            ),
            x_unit='mol',
            input_sample_name=['R0'],
            note='k-seq results of doped-pool BYO aminoacylation. Total DNA amount in each reacted sample were '
                 'quantified with spike-in sequence with 2 edit distance as radius or qPCR + Qubit'
        )

        # temp note: spike-in norm factor were calculated on original seq_table when a SpikeInNormalizer is created,
        # notice the seq_table normalized on already excludes some (~10 %) sequence Abe used for qPRC quantification
        # this is equivalent to using 1.11 of spike-in amount than intended, however, it should not affect the reacted
        # fraction we get
        byo_doped.add_spike_in(
            base_table=byo_doped.table.original,
            spike_in_seq='AAAAACAAAAACAAAAACAAA',
            spike_in_amount=np.concatenate((
                np.repeat([2, 2, 1, 0.2, .04], repeats=3),
                np.array([10])), axis=0  # input pool sequenced is 3-times of actual initial pool
            ) * 1.11,
            radius=radius,
            dist_type='edit',
            unit='ng',
        )

        # temp note: DNA Amount normalizer is calculated on whichever seq_table it applies to
        from . import filters
        spike_in_filter = filters.SpikeInFilter(target=byo_doped)  # remove spike-in seqs
        seq_length_filter = filters.SeqLengthFilter(target=None, min_len=21, max_len=21)  # remove non-21 nt seq
        no_ambiguity_filter = filters.NoAmbiguityFilter(target=None)  # remove sequences with ambiguity nucleotides

        # filtered seq_table by removing spike-in within 2 edit distance and seqs not with 21 nt
        byo_doped.table.filtered = no_ambiguity_filter(
            target=seq_length_filter(target=spike_in_filter(target=byo_doped.table.original))
        )
        byo_doped.add_sample_total(
            total_amounts=dna_amount,
            unit='ng',
            full_table=byo_doped.table.filtered
        )

        # add landscape to filter mutants
        from . import landscape
        pool_peaks = {
            'pk2': 'ATTACCCTGGTCATCGAGTGA',
            'pk1A': 'CTACTTCAAACAATCGGTCTG',
            'pk1B': 'CCACACTTCAAGCAATCGGTC',
            'pk3': 'AAGTTTGCTAATAGTCGCAAG'
        }
        byo_doped.pool_peaks = [landscape.Peak(seqs=byo_doped.table.filtered, center_seq=seq,
                                               name=name, dist_type='hamming') for name, seq in pool_peaks.items()]
        byo_doped.pool_peaks_merged = landscape.PeakCollection(peaks=byo_doped.pool_peaks)

        # Add replicates grouper
        byo_doped.grouper.add(byo={
            1250: ['A1', 'A2', 'A3'],
            250: ['B1', 'B2', 'B3'],
            50: ['C1', 'C2', 'C3'],
            10: ['D1', 'D2', 'D3'],
            2: ['E1', 'E2', 'E3'],
            'target': byo_doped.table.filtered
        })

        # calculate reacted faction, remove seqs are not in input pools
        from .transform import ReactedFractionNormalizer
        reacted_frac = ReactedFractionNormalizer(input_samples=['R0'],
                                                 reduce_method='median',
                                                 remove_empty=True)
        # normalized using spike-in, 10 % were taken for qPCR, in each sample it shouldn't effect reacted fraction
        byo_doped.table.reacted_frac_spike_in = reacted_frac(byo_doped.spike_in(target=byo_doped.table.filtered))

        # normalized using qPCR
        byo_doped.table.reacted_frac_qpcr = reacted_frac(byo_doped.sample_total(target=byo_doped.table.filtered))

        # further filter out sequences that are not detected in all samples
        min_detected_times_filter = filters.DetectedTimesFilter(
            min_detected_times=byo_doped.table.reacted_frac_spike_in.shape[1]
        )
        byo_doped.table.reacted_frac_spike_in_seq_in_all_smpl = min_detected_times_filter(
            byo_doped.table.reacted_frac_spike_in
        )
        byo_doped.table.reacted_frac_qpcr_seq_in_all_smpl = min_detected_times_filter(
            byo_doped.table.reacted_frac_qpcr
        )

        peak_filter = filters.PeakFilter(max_dist=2,
                                         dist_to_center=byo_doped.pool_peaks_merged.dist_to_center)
        byo_doped.table.reacted_frac_qpcr_2mutants = peak_filter(byo_doped.table.reacted_frac_qpcr)
        logging.info('Finished!')
    else:
        logging.info(f'Load BYO-doped pool data from pickled record from {BYO_DOPED_PKL}')
        from ..utility.file_tools import read_pickle
        byo_doped = read_pickle(BYO_DOPED_PKL)
        logging.info('Imported!')

    return byo_doped
