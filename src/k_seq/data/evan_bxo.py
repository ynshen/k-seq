"""Code for Evan's multi-substrates BXO k-seq experiments"""

from yutility import logging
import pandas as pd
import os
from .transform import Transformer


def load_dataset(dataset, kseq_data_config=None, from_count_file=False):
    """Load the dataset"""
    pass


def get_actual_bxo_concen(bxo, x_values, bxo_stock_csv=None):
    """Get the actual bxo concentration (x values) from measured values"""

    if bxo_stock_csv is None:
        bxo_stock_csv = os.getenv('EVAN_BXO_SUB_STOCK', None)

    stock_file = pd.read_csv(bxo_stock_csv, index_col=0)
    stock_file = stock_file.loc[[sample.split('-')[0].upper() == bxo.upper() for sample in stock_file.index], 'c']
    ratio = {name.split('-')[-1]: value / 25.0 for name, value in zip(stock_file.index, stock_file.values)}
    return pd.Series({
        sample: value * ratio[sample.split('_')[-1]] for sample, value in zip(x_values.index, x_values.values)
    })


def from_count_file(kseq_data_config):
    """Process sequence data from count file, config indicated in kseq_data_config"""
    from . import seq_data, filters

    logging.info('Dataset config file imported...')
    rna_amount_csv = kseq_data_config.pop('rna_amount', None)
    bxo_stock_csv = kseq_data_config.pop('bxo_stock_csv', None)
    bxo = kseq_data_config.pop('bxo', None)
    all_inputs_ng = kseq_data_config.pop('all_inputs', None)

    logging.info('Surveying count files...')
    dataset = seq_data.SeqData.from_count_files(**kseq_data_config)
    dataset.x_values = dataset.x_values * 1e-6
    dataset.x_values_true = get_actual_bxo_concen(bxo=bxo, x_values=dataset.x_values, bxo_stock_csv=bxo_stock_csv)

    logging.info('Filtering tables')
    # filter out sequences are not 21 length and sequences contains ambiguous nt
    dataset.table.filtered = filters.NoAmbiguityFilter.filter(
        target=filters.SeqLengthFilter.filter(min_len=21, max_len=21, target=dataset.table.original, axis=0),
        axis=0
    )

    # quantify total RNA
    total_rna = pd.read_csv(rna_amount_csv, index_col=0)
    dataset.add_sample_total(full_table=dataset.table.original,
                             total_amounts=total_rna['ng'],
                             unit='ng')

    # filter tables
    from k_seq.data.transform import ReactedFractionNormalizer

    dataset.table.filtered_reacted_frac = ReactedFractionNormalizer(input_samples=dataset.grouper.input.group)(
        dataset.sample_total(dataset.table.filtered)
    )

    # AltQuant 1: separately normalized
    dataset.table.filtered_reacted_frac_separate_normed = ReactedFractionSeparateNormalizer(
        input_sample_mapper={sample: f"input_{sample.split('_')[-1]}" for sample in dataset.grouper.reacted.group}
    )(dataset.sample_total(dataset.table.filtered))

    # AltQuant 2: input total is 500 ng
    from copy import deepcopy
    sample_total_altQuant = deepcopy(dataset.sample_total)
    sample_total_altQuant.total_amounts = {
        key: 500 if 'input' in key else amount for key, amount in dataset.sample_total.total_amounts.items()
    }
    dataset.sample_total_altQuant = sample_total_altQuant
    dataset.table.filtered_reacted_frac_input_500 = ReactedFractionNormalizer(input_samples=dataset.grouper.input.group)(
        dataset.sample_total_altQuant(dataset.table.filtered)
    )

    # AltQuant 3: use average over ABCDEF
    all_inputs_ng = pd.read_csv(all_inputs_ng, index_col=0)['ng']
    altquant_3_table = dataset.table.filtered
    altquant_3_table['input'] = all_inputs_ng.reindex(altquant_3_table.index)
    altquant_3_table.drop(columns=dataset.grouper.input.group)
    dataset.table.filtered_reacted_frac_all_input_median = ReactedFractionNormalizer(input_samples=['input'])(altquant_3_table)


    # filter up to double mutants
    from k_seq.data import landscape

    pool_peaks = {
        'pk2.1': 'ATTACCCTGGTCATCGAGTGA',
        'pk2.2': 'ATTCACCTAGGTCATCGGGTG',
        'pk1A.1': 'CTACTTCAAACAATCGGTCTG',
        'pk1B.1': 'CCACACTTCAAGCAATCGGTC',
        'pk3.1': 'AAGTTTGCTAATAGTCGCAAG'
    }
    dataset.pool_peaks = [
        landscape.Peak(seqs=dataset.table.filtered, center_seq=seq,
                       name=name, dist_type='hamming') for name, seq in pool_peaks.items()
    ]

    dataset.pool_peaks_merged = landscape.PeakCollection(peaks=dataset.pool_peaks)
    peak_filter = filters.PeakFilter(max_dist=2,
                                     dist_to_center=dataset.pool_peaks_merged.dist_to_center)
    dataset.table.filtered_reacted_frac_2mutant = peak_filter(dataset.table.filtered_reacted_frac)
    dataset.table.filtered_reacted_frac_separate_normed_2mutant = peak_filter(
        dataset.table.filtered_reacted_frac_separate_normed
    )
    dataset.table.filtered_reacted_frac_input_500_2mutant = peak_filter(
        dataset.table.filtered_reacted_frac_input_500
    )
    dataset.table.filtered_reacted_frac_all_input_median_2mutant = peak_filter(
        dataset.table.filtered_reacted_frac_all_input_median
    )

    return dataset


class ReactedFractionSeparateNormalizer(Transformer):
    """Get reacted fraction of each sequence from an absolute amount seq_table, normalization was separately done for
    each reacted sample"""

    def __init__(self, input_sample_mapper, target=None, remove_empty=True):
        super().__init__()
        self.target = target
        self.input_sample_mapper = input_sample_mapper
        self.remove_empty = remove_empty

    @staticmethod
    def func(target, input_sample_mapper, remove_empty=True):

        def get_reacted_fraction(sample):
            input_sample = target[input_sample_mapper[sample.name]]
            mask = input_sample > 0
            return sample[mask] / input_sample[mask]

        reacted_frac = pd.DataFrame(
            {sample: get_reacted_fraction(target[sample]) for sample in input_sample_mapper.keys()}
        ).fillna(0.)

        if remove_empty:
            return reacted_frac.loc[reacted_frac.sum(axis=1) > 0]
        else:
            return reacted_frac

    def apply(self, target=None, input_sample_mapper=None, remove_empty=None):
        """Convert absolute amount to reacted fraction
            Args:
                target (pd.DataFrame): the seq_table with absolute amount to normalize on inputs, including input pools
                input_sample_mapper (dict or pd.Series): mapper of reacted sample name to corresponding input sample
                remove_empty (bool): if will remove all-zero seqs from output seq_table

            Returns:
                SeqTable
        """
        from ..utility.func_tools import update_none
        target = update_none(target, self.target)
        if target is None:
            logging.error('No target found', ValueError)
        input_sample_mapper = update_none(input_sample_mapper, self.input_sample_mapper)
        if input_sample_mapper is None:
            logging.error('No input_sample_mapper found', ValueError)
        remove_empty = update_none(remove_empty, self.remove_empty)

        from .seq_data import SeqTable
        return SeqTable(self.func(target=target, input_sample_mapper=input_sample_mapper,
                                  remove_empty=remove_empty), unit='fraction')