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
from ..utility import DocHelper
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
        return f"SpikeIn Normalizer (center seq: {self.spike_in_seq}, radius= {self.radius}, dist measure={self.dist_measure})"

    def __init__(self, spike_in_seq, spike_in_amount, base_table, radius=None, unit='ng', dist_measure='hamming',
                 blacklist=None):
        """Initialize a SpikeInNormalizer

        Args:


        """
        super().__init__()
        self.spike_in_seq = spike_in_seq
        self.unit = unit
        self.target = base_table
        self.blacklist = blacklist
        self.spike_in_amount = spike_in_amount
        if isinstance(self.target, pd.DataFrame):
            self.dist_to_center = pd.Series(self.target.index.map(self._get_edit_dist), index=self.target.index)
        else:
            try:
                self.dist_to_center = pd.Series(self.target.table.index.map(self._get_edit_dist),
                                                index=self.target.table.index)
            except AttributeError:
                raise AttributeError('target for SpikeInNormalizer needs to be either `SeqTable` or `pd.Dataframe`')

        self.spike_in_members = None
        self.norm_factor = None
        self.radius = None
        self.radius = radius
        self.dist_measure = dist_measure

    @property
    def spike_in_amount(self):
        return self._spike_in_amount

    @spike_in_amount.setter
    def spike_in_amount(self, spike_in_amount):
        if isinstance(spike_in_amount, (list, np.ndarray)):
            if isinstance(self.target, pd.DataFrame):
                sample_list = self.target.columns
            else:
                sample_list = self.target.sample_list
            if self.blacklist is not None:
                sample_list = [sample for sample in sample_list if sample not in self.blacklist]
            self._spike_in_amount = {key: value for key, value in zip(sample_list, spike_in_amount)}
        elif isinstance(spike_in_amount, dict):
            self._spike_in_amount = spike_in_amount
        else:
            raise TypeError('spike_in_amount needs to be list or dict')

    def _get_edit_dist(self, seq):
        from Levenshtein import distance
        return distance(self.spike_in_seq, seq)

    def _get_norm_factor(self, sample):
        """Calculate norm factor from a sample column"""
        return self.spike_in_amount[sample.name] / sample[self.spike_in_members].sparse.to_dense().sum()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value is not None:
            if value != self.radius:
                self._radius = value
                self.spike_in_members = self.dist_to_center[self.dist_to_center <= value].index.values
                if isinstance(self.target, pd.DataFrame):
                    self.norm_factor = self.target[list(self.spike_in_amount.keys())].apply(self._get_norm_factor,
                                                                                            axis=0)
                else:
                    self.norm_factor = self.target.table[list(self.spike_in_amount.keys())].apply(
                        self._get_norm_factor, axis=0
                    )
        else:
            self._radius = None
            self.spike_in_members = None
            self.norm_factor = None

    @staticmethod
    def func(target, norm_factor, *args, **kwargs):

        def sample_normalizer(sample):
            return sample * norm_factor[sample.name]

        import pandas as pd
        if not isinstance(target, pd.DataFrame):
            raise TypeError('target needs to be pd.DataFrame')

        import warnings
        sample_list = []
        for sample in target.columns:
            if sample in norm_factor.keys():
                sample_list.append(sample)
            else:
                warnings.warn(f'Sample {sample} is not found in the normalizer, normalization is not performed')

        return target[sample_list].apply(sample_normalizer, axis=0)

    def apply(self, target=None, norm_factor=None):
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if norm_factor is None:
            norm_factor = self.norm_factor
        if norm_factor is None:
            raise ValueError('No valid norm_factor found')

        if isinstance(target, SeqTable):
            target = target.table

        return self.func(target=target, norm_factor=norm_factor)

    @staticmethod
    def spike_in_peak_plot(spike_in, seq_table=None, sample_list=None, max_dist=15, norm_on_center=True, log_y=True,
                           marker_list=None, color_list=None, err_guild_lines=None, label_map=None,
                           legend_off=False, legend_col=2, ax=None, figsize=None, save_fig_to=None):
        """Plot the distribution of spike_in peak
        Plot a scatter-line plot of [adjusted] number of sequences with i edit distance from center sequence (spike-in seq)

        Args:
            target (`SeqTable` or `pd.DataFrame`): dataset to plot
            sample_list (list of `str`): samples to show. All samples will show if None
            max_dist (`int`): maximal edit distance to survey. Default 15
            norm_on_center (`bool`): if the counts/abundance are normalized to then center (exact spike in)
            log_y (`bool`): if set the y scale as log
            marker_list (list of `str`): overwrite default marker scheme if not `None`, same length and order as valid samples
            color_list (list of `str`): overwrite default color scheme if not `None`, same length and order as valid samples
            label_map (dict or callable): alternative label for samples
            err_guild_lines (list of `float`): add a series of guild lines indicate the distribution only from given error rate,
              if not `None`
            legend_off (`bool`): do not show the legend if True
            ax (`matplotlib.Axis`): if use external ax object to plot. Create a new figure if `None`
            save_fig_to (`str`): save the figure to file if not None

        """

        from ..utility.plot_tools import PlotPreset
        import matplotlib.pyplot as plt
        import numpy as np

        if seq_table is None:
            seq_table = spike_in.target.table
        if sample_list is None:
            sample_list = seq_table.columns.values
        if marker_list is None:
            marker_list = PlotPreset.markers(num=len(sample_list), with_line=True)
        elif len(marker_list) != len(sample_list):
            raise Exception('Error: length of marker_list does not align with the number of valid samples to plot')
        if color_list is None:
            color_list = PlotPreset.colors(num=len(sample_list))
        elif len(color_list) != len(sample_list):
            raise Exception('Error: length of color_list does not align with the number of valid samples to plot')

        if ax is None:
            if figsize is None:
                figsize = (max_dist / 2, 6) if legend_off else (max_dist / 2 + 5, 6)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        dist_series = pd.Series(data=np.arange(max_dist + 1), index=np.arange(max_dist + 1))

        def get_sample_counts(dist):
            seqs = spike_in.dist_to_center[spike_in.dist_to_center == dist].index
            return seq_table.loc[seqs].sum(axis=0)

        peak_counts = dist_series.apply(get_sample_counts)
        if norm_on_center:
            peak_counts = peak_counts / peak_counts.loc[0]

        if label_map:
            if callable(label_map):
                label_map = {sample: label_map(sample) for sample in sample_list}
        else:
            label_map = {sample: sample for sample in sample_list}
        for sample, color, marker in zip(sample_list, color_list, marker_list):
            ax.plot(dist_series, peak_counts[sample], marker, color=color, label=label_map[sample],
                    ls='-', alpha=0.5, markeredgewidth=2)
        if log_y:
            ax.set_yscale('log')
        ylim = ax.get_ylim()
        if err_guild_lines is not None:
            if not norm_on_center:
                raise ValueError('Can only add guidelines if peaks are normed on center')
            # assuming a fix error rate per nt, iid on binom
            from scipy.stats import binom
            if isinstance(err_guild_lines, (float, int)):
                err_guild_lines = [err_guild_lines]
            colors = PlotPreset.colors(num=len(err_guild_lines))
            for ix, (p, color) in enumerate(zip(err_guild_lines, colors)):
                rv = binom(len(spike_in.spike_in_seq), p)
                pmfs = np.array([rv.pmf(x) for x in dist_series])
                pmfs_normed = pmfs / pmfs[0]
                ax.plot(dist_series, pmfs_normed,
                        color=color, ls='--', alpha=(ix + 1) / len(err_guild_lines), label=f'p = {p}')
        ax.set_ylim(ylim)
        y_label = ''
        if norm_on_center:
            y_label += ' normed'
        y_label += ' counts'
        ax.set_ylabel(y_label.title(), fontsize=14)
        ax.set_xlabel('Edit Distance to Spike-in Center', fontsize=14)
        if not legend_off:
            ax.legend(loc=[1.02, 0], fontsize=9, frameon=False, ncol=legend_col)
        plt.tight_layout()

        if save_fig_to:
            fig.patch.set_facecolor('none')
            fig.patch.set_alpha(0)
            plt.savefig(save_fig_to, bbox_inches='tight', dpi=300)
        return ax


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
