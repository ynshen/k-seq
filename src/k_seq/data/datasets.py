"""Pipeline for generating datasets for k-seq project
Available datasets:
  - BYO doped pool: byo-doped
  - BYO selection pool: byo-selected
"""

import os
from yutility import logging
from .seq_data import SeqData, SeqTable


def load_dataset(dataset, from_count_file=False, **kwargs):
    """Load default dataset
    Available dataset:
      - BYO-doped: 'byo-doped'
      - BYO-selected: 'byo-selected'
      - BFO: not implemented
    """
    if dataset.lower() in ['byo_doped', 'byo-doped']:
        return load_byo_doped(from_count_file=from_count_file, **kwargs)
    elif dataset.lower() in ['byo-doped-test', 'byo_doped_test']:
        return load_byo_doped(from_count_file=from_count_file,
                              count_file_path=os.getenv('BYO_DOPED_COUNT_FILE_TEST'),
                              **kwargs)
    elif dataset.lower() in ['byo_selected', 'byo-selected', 'selected', 'byo-selection']:
        return load_byo_selected(from_count_file=from_count_file, **kwargs)
    else:
        logging.error(f'Dataset {dataset} is not implemented', error_type=NotImplementedError)


def byo_doped_rename_sample(name):
    """Rename results loaded from raw reads and samples as

    A1/d-A1_S1 --> 1250uM-1
    ...
    R/R0 --> input
    """

    if len(name) > 2:
        name = name.split('_')[0].split('-')[-1]

    if 'R' in name:
        return 'Input'
    else:
        concen_mapper = {
            'A': '1250',
            'B': '250',
            'C': '50',
            'D': '10',
            'E': '2'
        }
        return "{} $\mu M$-{}".format(concen_mapper[name[0]], name[1])


def load_byo_doped(from_count_file=False, count_file_path=None, doped_norm_path=None, pickled_path=None,
                   pandaseq_joined=True, radius=2):
    """BYO doped pool k-seq datatable contains k-seq results for seqs from doped-pool for BYO aminoacylation,

    this dataset contains following pre-computed seq_table (``.table`` accessor) to use

        - original (4313709, 16): count seq_table contains all sequences detected in any samples and all the samples
        - filtered (3290337, 16): count seq_table with non 21 nt sequences and spike-in sequences filtered
        - reacted_frac_spike_in (764756, 15): reacted fraction for valid seqs quantified by spike-in
        - reacted_frac_qpcr (764756, 15): reacted fraction for valid seqs quantified by qPCR
        - reacted_frac_spike_in_seq_in_all_smpl (22525, 15): only contains seqs with counts >= 1 in all samples
        - reacted_frac_qpcr_seq_in_all_smpl (22525, 15): only contains seqs with counts >= 1 in all samples

    Note:
       By default, sequences within 2 edit distance (including insertion and deletion) of spike-in sequences were
           considered as spike-in seq
    """
    import os

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
        # notice the seq_table normalized on were already without some (~10%) sequence Abe used for qPRC quantification
        byo_doped.add_spike_in(
            base_table=byo_doped.table.original,
            spike_in_seq='AAAAACAAAAACAAAAACAAA',
            spike_in_amount=np.concatenate((
                np.repeat([2, 2, 1, 0.2, .04], repeats=3),
                np.array([10])), axis=0  # input pool sequenced is 3-times of actual initial pool
            ),
            radius=radius,
            dist_type='edit',
            unit='ng',
        )

        # temp note: DNA Amount normalizer is calculated on whichever seq_table it applies to
        from . import filters
        spike_in_filter = filters.SpikeInFilter(target=byo_doped)  # remove spike-in seqs
        seq_length_filter = filters.SeqLengthFilter(target=byo_doped, min_len=21, max_len=21)  # remove non-21 nt seq

        # filtered seq_table by removing spike-in within 2 edit distance and seqs not with 21 nt
        byo_doped.table.filtered = seq_length_filter(target=spike_in_filter(target=byo_doped.table.original))
        byo_doped.add_sample_total(
            total_amounts=dna_amount,
            unit='ng',
            full_table=byo_doped.table.filtered
        )

        from . import landscape
        pool_peaks = {
            'pk2': 'ATTACCCTGGTCATCGAGTGA',
            'pk1A': 'CTACTTCAAACAATCGGTCTG',
            'pk1B': 'CCACACTTCAAGCAATCGGTC',
            'pk3': 'AAGTTTGCTAATAGTCGCAAG'
        }
        byo_doped.pool_peaks = [landscape.Peak(seqs=byo_doped.table.filtered, center_seq=seq,
                                               name=name, dist_type='hamming') for name, seq in pool_peaks.items()]
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
        logging.info('Finished!')
    else:
        logging.info(f'Load BYO-doped pool data from pickled record from {BYO_DOPED_PKL}')
        from ..utility.file_tools import read_pickle
        byo_doped = read_pickle(BYO_DOPED_PKL)
        logging.info('Imported!')

    return byo_doped


_byo_selected_description = """
        contains k-seq results for seqs from BYO AA selections, this dataset contains following pre-computed seq_table to 
          use

            - seq_table: original count seq_table contains all sequences detected in any samples and all the samples
            - table_no_failed: count seq_table with sample `2C`, `3D`, `3E`, `3F`, `4D`, `4F` removed (failed in sequencing)

            Tables based on curated quantification factor
            - table_nf_reacted_frac_curated: reacted fraction of sequences (failed samples are removed, sequences only
                detected input or reacted samples are removed)
            - table_nf_filtered_reacted_frac_curated: reacted fraction for only sequences that are not spike-in and
                has length equals to 21 nt

            Tables based on standard pipeline
            - table_nf_filtered: count seq_table for sequences that are not spike-in and have length of 21 nt
            - table_nf_filtered_abs_amnt: absolute amount seq_table from table_nf_filterd, quantified by spike-in for
                reacted samples, total DNA amount for input pool
            - table_nf_filtered_reacted_frac: reacted fractions for sequences that are at least detected in both input
                pool and one reacted pool
    """


# TODO: update byo_selected dataset pipeline
def load_byo_selected(from_count_file=False, count_file_path=None, norm_path=None, pickled_path=None):
    """Load k-seq results for BYO selected pool
    {description}
    """.format(description=_byo_selected_description)
    import os

    PKL_FILE = os.getenv('BYO_SELECTED_PKL', None) if pickled_path is None else pickled_path
    COUNT_FILE = os.getenv('BYO_SELECTED_COUNT_FILE', None) if count_file_path is None else count_file_path
    NORM_FILE = os.getenv('BYO_SELECTED_NORM_FILE', None) if norm_path is None else norm_path

    if from_count_file:
        import numpy as np
        import pandas as pd

        logging.info('Generate SeqData instance for BYO-selected pool...')
        logging.info(f'Importing from {COUNT_FILE}...this could take a couple of minutes...')

        import numpy as np
        import pandas as pd

        byo_selected = SeqData.from_count_files(
            file_root=COUNT_FILE,
            pattern_filter='counts-',
            name_pattern='counts-[{byo}{exp_rep}].txt',
            dry_run=False,
            sort_by='name',
            x_values=np.concatenate((
                np.repeat([250, 50, 10, 2], repeats=6) * 1e-6,
                np.array([np.nan])), axis=0
            ),
            x_unit='mol',
            spike_in_seq='AAAAACAAAAACAAAAACAAA',
            spike_in_amount=np.repeat([1, 0.4, 0.3, 0.1], repeats=6),
            radius=4,
            dna_amount={'R0': 2000},
            dna_unit='ng',
            input_sample_name=['R0']
        )

        # Add filters and normalizers
        from . import filters
        spike_in_filter = filters.SpikeInFilter(target=byo_selected)  # remove spike-in seqs
        seq_length_filter = filters.SeqLengthFilter(target=byo_selected, min_len=21, max_len=21)
        sample_filter = filters.SampleFilter(samples_to_remove=[
            '2C',
            '3D', '3E', '3F',
            '4D', '4F'
        ])
        from .transform import ReactedFractionNormalizer, BYOSelectedCuratedNormalizerByAbe
        reacted_frac = ReactedFractionNormalizer(input_samples=['R0'],
                                                 reduce_method='median',
                                                 remove_empty=True)
        # Add replicates grouper
        byo_selected.grouper.add({'byo': {
            250: ['1A', '1B', '1C', '1D', '1E', '1F'],
            50: ['2A', '2B', '2C', '2D', '2E', '2F'],
            10: ['3A', '3B', '3C', '3D', '3E', '3F'],
            2: ['4A', '4B', '4C', '4D', '4E', '4F'],
        }}, target=byo_selected.table)

        # Remove failed exp
        byo_selected.table_no_failed = sample_filter(byo_selected.table)
        # Recover Abe's dataset
        curated_normalizer = BYOSelectedCuratedNormalizerByAbe(target=byo_selected.table_no_failed, q_factor=NORM_FILE)
        # Note: in original code the normalization was applied to all seqs including spike-in sequences
        byo_selected.table_nf_reacted_frac_curated = reacted_frac.apply(curated_normalizer.apply())
        byo_selected.table_nf_filtered_reacted_frac_curated = seq_length_filter(
            spike_in_filter(byo_selected.table_nf_reacted_frac_curated)
        )

        # Prepare sequences with general pipeline
        # filtered seq_table by removing spike-in within 4 edit distance and seqs not with 21 nt
        byo_selected.table_nf_filtered = seq_length_filter.get_filtered_table(
            spike_in_filter(byo_selected.table_no_failed)
        )

        # normalized using spike-in and total DNA amount
        table_reacted_spike_in = byo_selected.spike_in.apply(target=byo_selected.table_nf_filtered)
        table_input_dna_amount = byo_selected.dna_amount.apply(target=byo_selected.table_nf_filtered)
        byo_selected.table_nf_filtered_reacted_frac = reacted_frac.apply(
            pd.concat([table_reacted_spike_in, table_input_dna_amount], axis=1)
        )

        # further filter out sequences that are not detected in all samples
        min_detected_times_filter = filters.DetectedTimesFilter(
            min_detected_times=byo_selected.table_nf_filtered_reacted_frac.shape[1]
        )
        byo_selected.table_nf_filtered_seq_in_all_smpl_reacted_frac = min_detected_times_filter(
            target=byo_selected.table_nf_filtered_reacted_frac
        )
        byo_selected.table_nf_filtered_seq_in_all_smpl_reacted_frac_curated = min_detected_times_filter(
            target=byo_selected.table_nf_filtered_reacted_frac_curated
        )
        logging.info('Finished!')
    else:
        logging.info(f'Load BYO-selected pool data from pickled record from {PKL_FILE}')
        from ..utility.file_tools import read_pickle
        byo_selected = read_pickle(PKL_FILE)
        logging.info('Imported!')

    return byo_selected
