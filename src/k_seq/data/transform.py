"""
Module contains the classes to transform tables (`pd.DataFrame` or `SeqTable`) instance as well as related calculation
    and visualizations

Current available transformers:
    - SpikeInNormalizer: normalize by spike-in
    - DnaAmountNormalizer: normalize by total dna amount
    - ReactedFractionNormalizer: transform to relative abundance
    - BYOSelectedPoolNormalizerByAbe: curated quantification factor used by Abe
"""

import numpy as np
import pandas as pd
from .seq_table import SeqTable
from abc import ABC, abstractmethod
from doc_helper import DocHelper
from ..utility.func_tools import update_none
from ..utility.log import logging


class Transformer(ABC):
    """Abstract class type for transformer

    Transformers are classes transform a table instance (`pd.DataFrame` or `SeqTable`) to another table

    To write your transformer, components are:
        - Attributes to store parameters
        - A static `func` function to perform transformation
        - A `apply` wrapper function for ease of use
        - Other Transformer specific utility and visualization functions
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def func():
        """core function, input `pd.Dataframe`, output an transformed `pd.Dataframe`"""
        raise NotImplementedError()

    @abstractmethod
    def apply(self, target):
        """Run the transformation, with class attributes or arguments
        Logic and preprocessing of data and arguments should be done here
        """
        raise NotImplementedError()


_spike_in_doc = DocHelper(
    radius=('int', 'Radius of spike-in peak, seqs less or equal to radius away from center are spike-in seqs'),
)


class SpikeInNormalizer(Transformer):
    """Normalized counts to absolute amount using spike-in information
    todo: add attributes of spike-in


    Attributes:

    """

    def __repr__(self):
        return f"SpikeIn Normalizer (center seq: {self.peak.center_seq}," \
               f"radius: {self.radius}, dist type: {self.dist_type})"

    def __init__(self, spike_in_seq, spike_in_amount, base_table, radius, unit, dist_type='edit'):
        """Initialize a SpikeInNormalizer

        Args:


        """
        from landscape import Peak

        super().__init__()

        self.peak = Peak(center_seq=spike_in_seq, radius=radius,
                         seqs=base_table,
                         dist_type=dist_type, name='spike-in')
        self.base_table = self.peak.seqs
        self.spike_in_amount = spike_in_amount
        self.unit = unit
        self.radius = radius
        self.dist_type = dist_type
        self.plot_spike_in_peak = self.peak.vis.peak_plot

    def _update_spike_in_members(self):
        """Update spike-in members based on current distance and radius"""
        self.spike_in_members = self.peak.dist_to_center[self.peak.dist_to_center <= self.radius]

    def _update_norm_factors(self):
        """Update normalization factor for each sample"""
        spike_in_counts = self.base_table.loc[self.spike_in_members, self.spike_in_amount.index].sum(axis=0)
        self.norm_factor = self.spike_in_amount / spike_in_counts

    @property
    def spike_in_amount(self):
        return self._spike_in_amount

    @spike_in_amount.setter
    def spike_in_amount(self, spike_in_amount):
        """Check and reformat spike_in_amount type, and update norm_factors"""

        if isinstance(spike_in_amount, (list, np.ndarray)):
            # if unkey array, must be same length as base_table's columns
            if len(self.base_table.columns) != len(spike_in_amount):
                logging.error('Length of spike_in_amount does not match sample number')
                ValueError('Length of spike_in_amount does not match sample number')
            else:
                self._spike_in_amount = pd.Series(data=spike_in_amount, index=self.base_table.columns)
        elif isinstance(spike_in_amount, dict):
            self._spike_in_amount = pd.Series(spike_in_amount)

        self._update_norm_factors()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        """Update spike_in_members and norm_factors when radius change"""
        if value is None:
            self._radius = None
            self.norm_factor = None
            self.spike_in_amount = None
        else:
            if value != self.radius:
                self._radius = value
                self._update_spike_in_members()
                self._update_norm_factors()

    @staticmethod
    def func(target, norm_factor):
        """Normalize counts in target by norm_factor"""

        if not isinstance(target, pd.DataFrame):
            logging.error('target needs to be pd.DataFrame')
            raise TypeError('target needs to be pd.DataFrame')

        sample_list = norm_factor.index
        sample_not_in = list(target.columns[~target.columns.isin(sample_list)])

        for sample in sample_not_in:
            logging.warning(f'Sample {sample} is not found in the normalizer, normalization is not performed')

        return target.loc[:, sample_list] * norm_factor

    def apply(self, target):
        return self.func(target=target, norm_factor=self.norm_factor)


_total_dna_doc = DocHelper(
    total_amounts=('dict or pd.Series', 'Total DNA amount for samples measured in experiment'),
    target=('pd.DataFrame', 'Target table to convert, samples are columns'),
    unit=('str', 'Unit of amount measured'),
    norm_factor=('dict', 'Amount per sequence. Sequence amount = norm_factor * reads'),
    full_table=('pd.DataFrame', 'table where the total amount were measured and normalize to'),

)


class TotalAmountNormalizer(Transformer):
    f"""Quantify the DNA amount by total DNA amount measured in each sample

    Amount for each sequence were normalized by direct fraction of the sequence
        seq amount = (seq counts / total counts) * total amount
    
    Attributes:
    {_total_dna_doc.get(['full_table', 'total_amounts', 'norm_factor', 'unit'])}
    """

    def __init__(self, total_amounts, full_table, unit=None):
        f"""Initialize a TotalAmountNormalizer
        Args:
        {_total_dna_doc.get(['total_amounts', 'full_table', 'target', 'unit'])}
        """

        super().__init__()
        self._full_table = None
        self._total_amounts = None
        self.full_table = full_table
        self.total_amounts = total_amounts
        self.unit = unit

    @property
    def total_amounts(self):
        return self._total_amounts

    @total_amounts.setter
    def total_amounts(self, value):
        if isinstance(value, pd.Series):
            value = value.to_dict()
        self._total_amounts = value
        self._update_norm_factor()

    @property
    def full_table(self):
        return self._full_table

    @full_table.setter
    def full_table(self, value):
        if not isinstance(value, pd.DataFrame):
            logging.error("full table needs to be a pd.DataFrame", error_type=TypeError)
        self._full_table = value
        self._update_norm_factor()

    def _update_norm_factor(self):
        """Update norm factor value based on current self.total_amounts and self.full_table"""
        if (self.full_table is not None) and (self.total_amounts is not None):
            for sample in self.full_table.columns:
                if sample not in self.total_amounts.keys():
                    logging.info(f'Notice: {sample} is not in total_amount, skip this sample')

            self.norm_factor = {}
            from ..utility.func_tools import is_sparse

            for sample, amount in self.total_amounts.items():
                if sample in self.full_table.columns:
                    if is_sparse(self.full_table[sample]):
                        self.norm_factor[sample] = amount / self.full_table[sample].sparse.to_dense().sum()
                    else:
                        self.norm_factor[sample] = amount / self.full_table[sample].sum()
                else:
                    logging.info(f"Notice: {sample} is not in full_table")

    @staticmethod
    def func(target, norm_factor):
        """Normalize target table w.r.t norm_factor

        Returns:
            pd.DataFrame of normalized target table with only samples provided in norm_factor
        """

        def sample_normalize(col):
            return col * norm_factor[col.name]

        sample_list = []
        for sample in target.columns:
            if sample in norm_factor.keys():
                sample_list.append(sample)
            else:
                logging.warning(f'Sample {sample} is not in norm_factor, skip this sample')

        return target[sample_list].apply(sample_normalize, axis=0)

    def apply(self, target):
        """Transform counts to absolute amount on columns in target that are in norm_factor 
        
        Args:
        {args}

        Returns:
            pd.DataFrame of normalized target table with only samples in norm_factor
        """.format(args=_total_dna_doc.get(['target']))
        return self.func(target=target, norm_factor=self.norm_factor)


class ReactedFractionNormalizer(Transformer):
    """Get reacted fraction of each sequence from an absolute amount table"""

    def __init__(self, input_samples, target=None, reduce_method='median', remove_zero=True):
        super().__init__()
        self.target = target
        self.input_samples = input_samples
        self.reduce_method = reduce_method
        self.remove_zero = remove_zero

    @staticmethod
    def func(target, input_samples, reduce_method='median', remove_zero=True):

        method_mapper = {
            'med': np.nanmedian,
            'median': np.nanmedian,
            'mean': np.nanmean,
            'avg': np.nanmean
        }
        if callable(reduce_method):
            base = reduce_method(target[input_samples])
        else:
            if reduce_method.lower() not in method_mapper.keys():
                raise ValueError('Unknown reduce_method')
            else:
                base = method_mapper[reduce_method](target[input_samples], axis=1)

        mask = base > 0  # if any does not exist in input samples
        reacted_frac = target.loc[mask, ~target.columns.isin(input_samples)].divide(base[mask], axis=0)
        if remove_zero:
            return reacted_frac[reacted_frac.sum(axis=1) > 0]
        else:
            return reacted_frac

    def apply(self, target=None, input_samples=None, reduce_method=None, remove_zero=None):
        """Convert absolute amount to reacted fraction
            Args:
                target (pd.DataFrame): the table with absolute amount to normalize on inputs, including input pools
                input_samples (list of str): list of indices of input pools
                reduce_method (str or callable): 'mean' or 'median' or a callable apply on a pd.DataFrame to list-like
                remove_zero (bool): if will remove all-zero seqs from output table

            Returns:
                pd.DataFrame
        """
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if input_samples is None:
            input_samples = self.input_samples
        if input_samples is None:
            raise ValueError('No input_samples found')
        if reduce_method is None:
            reduce_method = self.reduce_method
        if not isinstance(target, pd.DataFrame):
            target = getattr(target, 'table')
        if remove_zero is None:
            remove_zero = self.remove_zero

        return self.func(target=target, input_samples=input_samples,
                          reduce_method=reduce_method, remove_zero=remove_zero)


class BYOSelectedCuratedNormalizerByAbe(Transformer):
    """This normalizer contains the quantification factor used by Abe"""

    def __init__(self, q_factor=None, target=None):
        super().__init__()
        self.target = target
        # import curated quantification factor by Abe
        # q_facter is defined in this way: abs_amnt = q * counts / total_counts
        self.q_factor = pd.read_csv(q_factor, index_col=0) if isinstance(q_factor, str) else q_factor

        # q_factor should be:
        # self.q_factors = {'0.0005,
        #             0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133, 0.023823133,
        #             0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812, 0.062784812,
        #             0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207, 0.159915207,
        #             0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596, 0.53032596]

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
                target (pd.DataFrame): this should be the original count table from BYO-selected k-seq exp.
                q_factor (pd.DataFrame or str): table contains first col as q-factor with sample as index
                    or path to stored csv file

            Returns:
                A normalized table of absolute amount of sequences in each sample
        """
        if target is None:
            target = self.target
        if q_factor is None:
            q_factor = self.q_factor
        q_factor = pd.read_csv(q_factor, index_col=0) if isinstance(q_factor, str) else q_factor

        return self.func(target=target, q_factor=q_factor)
