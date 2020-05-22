"""kSeq results from BYO-selected a.k.a enriched pool by Abe BYO aminoacylation ribozyme selection
kSeq was applied on the output pool of Round 5 of selection, and reacted with 2, 10, 50, 250 uM of BYO
"""

from yutility import logging
from doc_helper import DocHelper
from .transform import Transformer
import pandas as pd
import numpy as np

doc = DocHelper()


class BYOSelectedCuratedNormalizerByAbe(Transformer):
    """This normalizer contains the quantification factor used by Abe"""

    def __init__(self, q_factor=None, target=None):
        super().__init__()
        self.target = target
        # import curated quantification factor by Abe
        # q_facter is defined in this way: abs_amnt = counts / ( total_counts * q)
        self.q_factor = pd.read_csv(q_factor, index_col=0) if isinstance(q_factor, str) else q_factor
        self.unit = 'ng'

        # q_factor should be:
        # seq_data.q_factors = [0.0005,
        #                       0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133,
        #                       0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812,
        #                       0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207,
        #                       0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596]

    @staticmethod
    def func(target, q_factor):
        total_counts = target.sum(axis=0)
        if isinstance(q_factor, pd.DataFrame):
            q_factor = q_factor.iloc[:, 0]
        q_factor = q_factor.reindex(total_counts.index)
        return target / total_counts / q_factor

    def apply(self, target=None, q_factor=None):
        """Normalize counts using Abe's curated quantification factor
            Args:
                target (pd.DataFrame): this should be the original count seq_table from BYO-selected k-seq exp.
                q_factor (pd.DataFrame or str): seq_table contains first col as q-factor with sample as index
                    or path to stored csv file

            Returns:
                A normalized seq_table of absolute amount of sequences in each sample
        """
        if target is None:
            target = self.target
        if q_factor is None:
            q_factor = self.q_factor
        q_factor = pd.read_csv(q_factor, index_col=0) if isinstance(q_factor, str) else q_factor

        from .seq_data import SeqTable
        return SeqTable(self.func(target=target, q_factor=q_factor), unit='ng')


_byo_selected_description = """
contains k-seq results for seqs from BYO AA selections, this dataset contains following pre-computed seq_table to use

Count tables:
    - original: original count table contains all sequences detected in any samples and all the samples
    - no_failed: count seq_table with sample `2C`, `3D`, `3E`, `3F`, `4D`, `4F` removed (failed in sequencing)
    - nf_filtered: count seq_table with spike-in sequence (2 edit distance) and non-21 nt length sequence removed

Tables based on curated quantification factor
    - nf_reacted_frac_curated: reacted fraction of sequences (failed samples are removed, sequences only
      detected input or reacted samples are removed)
    - nf_filtered_seq_in_all_smpl_reacted_frac_curated: reacted fraction of sequences that were detected in all 
      available samples

Tables based on standard pipeline
    - nf_filtered_reacted_frac: reacted fraction with standard quantification methods (spike-in for reacted samples,
      total DNA amount for input samples)
    - nf_filtered_seq_in_all_smpl_reacted_frac: reacted fraction from standard pipeline, sequences were detected in all
      available samples
    """


@doc.compose(_byo_selected_description)
def load_byo_selected(from_count_file=False, count_file_path=None, norm_path=None, pickled_path=None):

    import os

    PKL_FILE = os.getenv('BYO_SELECTED_PKL', None) if pickled_path is None else pickled_path
    COUNT_FILE = os.getenv('BYO_SELECTED_COUNT_FILE', None) if count_file_path is None else count_file_path
    NORM_FILE = os.getenv('BYO_SELECTED_NORM_FILE', None) if norm_path is None else norm_path

    if from_count_file:
        from .seq_data import SeqData
        from . import filters

        logging.info('Generate SeqData instance for BYO-selected pool...')
        logging.info(f'Importing from {COUNT_FILE}...this could take a couple of minutes...')

        byo_selected = SeqData.from_count_files(
            count_files=COUNT_FILE,
            pattern_filter='counts-',
            name_template='counts-[{byo}{exp_rep}].txt',
            dry_run=False,
            sort_by='name',
            x_values=np.concatenate((
                np.repeat([250, 50, 10, 2], repeats=6) * 1e-6,
                np.array([np.nan])), axis=0
            ),
            x_unit='mol',
            input_sample_name=['R0']
        )
        sample_filter = filters.SampleFilter(samples_to_remove=[
            '2C',
            '3D', '3E', '3F',
            '4D', '4F'
        ])
        # Remove failed exp
        byo_selected.table.no_failed = sample_filter(byo_selected.table.original)
        byo_selected.add_spike_in(
            base_table=byo_selected.table.no_failed,
            spike_in_seq='AAAAACAAAAACAAAAACAAA',
            spike_in_amount=pd.Series(data=np.repeat([1, 0.4, 0.3, 0.1], repeats=6),
                                      index=byo_selected.grouper.reacted.group),
            radius=2,
            unit='ng',
            dist_type='edit'
        )
        byo_selected.add_sample_total(
            full_table=byo_selected.table.no_failed,
            total_amounts={'R0': 2000},
            unit='ng'
        )

        # Add replicates grouper
        byo_selected.grouper.add(byo={
            250: ['1A', '1B', '1C', '1D', '1E', '1F'],
            50: ['2A', '2B', '2C', '2D', '2E', '2F'],
            10: ['3A', '3B', '3C', '3D', '3E', '3F'],
            2: ['4A', '4B', '4C', '4D', '4E', '4F'],
            'target': None
        })

        # Remove sequences are not 21 nt long or within 2 edit distance to spike-in
        spike_in_filter = filters.SpikeInFilter(target=byo_selected)  # remove spike-in seqs
        seq_length_filter = filters.SeqLengthFilter(target=byo_selected, min_len=21, max_len=21)
        byo_selected.table.nf_filtered = seq_length_filter.get_filtered_table(
            spike_in_filter(byo_selected.table.no_failed)
        )

        from .transform import ReactedFractionNormalizer
        reacted_frac = ReactedFractionNormalizer(input_samples=['R0'],
                                                 reduce_method='median',
                                                 remove_empty=True)
        # Recover Abe's dataset
        # Reacted fraction was applied on non-filtered
        curated_normalizer = BYOSelectedCuratedNormalizerByAbe(target=byo_selected.table.no_failed, q_factor=NORM_FILE)
        # Note: in original code the normalization was applied to all seqs including spike-in sequences
        byo_selected.table.nf_reacted_frac_curated = reacted_frac.apply(curated_normalizer.apply(
            byo_selected.table.no_failed
        ))
        byo_selected.table.nf_filtered_reacted_frac_curated = seq_length_filter(
            spike_in_filter(byo_selected.table.nf_reacted_frac_curated)
        )

        # Prepare sequences with general pipeline
        # normalized using spike-in and total DNA amount
        table_reacted_spike_in = byo_selected.spike_in.apply(target=byo_selected.table.nf_filtered)
        table_input_dna_amount = byo_selected.sample_total.apply(target=byo_selected.table.nf_filtered)
        byo_selected.table.nf_filtered_reacted_frac = reacted_frac.apply(
            pd.concat([table_reacted_spike_in, table_input_dna_amount], axis=1)
        )
        # further filter out sequences that are not detected in all samples
        min_detected_times_filter = filters.DetectedTimesFilter(
            min_detected_times=byo_selected.table.nf_filtered_reacted_frac.shape[1]
        )
        byo_selected.table.nf_filtered_seq_in_all_smpl_reacted_frac = min_detected_times_filter(
            target=byo_selected.table.nf_filtered_reacted_frac
        )
        byo_selected.table.nf_filtered_seq_in_all_smpl_reacted_frac_curated = min_detected_times_filter(
            target=byo_selected.table.nf_filtered_reacted_frac_curated
        )
        logging.info('Finished!')
    else:
        logging.info(f'Load BYO-selected pool data from pickled record from {PKL_FILE}')
        from ..utility.file_tools import read_pickle
        byo_selected = read_pickle(PKL_FILE)
        logging.info('Imported!')

    return byo_selected
