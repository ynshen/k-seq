"""
Module contains the method of transform `SeqTable` including
    - normalize by spike-in
    - normalize by total amount
    - transform to relative abundance
    - etc
"""

import numpy as np
import pandas as pd
from .seq_table import SeqTable


class TransformerBase(object):

    def __init__(self, target=None):
        self.target = target
        pass

    @staticmethod
    def func(target):
        """core function, input an `pd.Dataframe`, output an transformed `pd.Dataframe`. Should NOT be used alone"""
        raise NotImplementedError()

    def apply(self, target=None):
        """Run the transformation, with class attributes or arguments
        Logic and preprocessing of data and arguments should be done here
        """
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')

        return self.func(target=target)


class SpikeInNormalizer(TransformerBase):
    """
    Normalizer using spike-in information
    """

    def __repr__(self):
        return f"Spike-in normalizer (seq: {self.spike_in_seq}, radius= {self.radius})"

    def __init__(self, spike_in_seq, spike_in_amount, target, radius=None, unit='ng', blacklist=None):
        super().__init__(target)
        self.spike_in_seq = spike_in_seq
        self.unit = unit
        self.target = target
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
                    self.norm_factor = self.target.apply(self._get_norm_factor, axis=0)
                else:
                    self.norm_factor = self.target.table.apply(self._get_norm_factor, axis=0)
        else:
            self._radius = None
            self.spike_in_members = None
            self.norm_factor = None

    @staticmethod
    def func(target, norm_factor, **kwargs):

        def sample_normalize_wrapper(norm_factor):

            def sample_normalizer(sample):
                if sample.name in norm_factor.keys():
                    return sample * norm_factor[sample.name]
                else:
                    import warnings
                    warnings.warn(
                        f'Sample {sample.name} is not found in the normalizer, normalization is not performed')
                    return sample

            return sample_normalizer

        import pandas as pd
        if not isinstance(target, pd.DataFrame):
            raise TypeError('target needs to be pd.DataFrame')
        return target.apply(sample_normalize_wrapper(norm_factor=norm_factor), axis=0)

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

    def spike_in_peak_plot(self, target=None, sample_list=None, max_dist=15, norm_on_center=True, log_y=True,
                           marker_list=None, color_list=None, err_guild_lines=None,
                           legend_off=False, ax=None, figsize=None, save_fig_to=None):
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
            err_guild_lines (list of `float`): add a series of guild lines indicate the distribution only from given error rate,
              if not `None`
            legend_off (`bool`): do not show the legend if True
            ax (`matplotlib.Axis`): if use external ax object to plot. Create a new figure if `None`
            save_fig_to (`str`): save the figure to file if not None

        """

        from ..utility.plot_tools import PlotPreset
        import matplotlib.pyplot as plt
        import numpy as np

        if target is None:
            target = self.target

        if sample_list is None:
            sample_list = target.columns.values
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
            seqs = self.dist_to_center[self.dist_to_center == dist].index
            return target.loc[seqs].sum(axis=0)

        peak_counts = dist_series.apply(get_sample_counts)
        if norm_on_center:
            peak_counts = peak_counts / peak_counts.loc[0]

        for sample, color, marker in zip(sample_list, color_list, marker_list):
            ax.plot(dist_series, peak_counts[sample], marker, color=color, label=sample, alpha=0.5, markeredgewidth=2)
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
                rv = binom(len(self.spike_in_seq), p)
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
            ax.legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
        plt.tight_layout()

        if save_fig_to:
            plt.savefig(save_fig_to, bbox_inches='tight', dpi=300)
        return ax


class DnaAmountNormalizer(TransformerBase):
    """todo: check if finished"""

    def __init__(self, target, dna_amount, exclude_spike_in=True, spike_in_members=None):
        super().__init__(target)
        self.target = target
        self.dna_amount = dna_amount
        self.exclude_spike_in = exclude_spike_in
        self.spike_in_members = spike_in_members

    @staticmethod
    def func(target, norm_factor):
        """Norm factor here should be per seq amount calculated from total DNA amount"""

        def sample_normalize(sample):
            return sample * norm_factor[sample.name]

        return target.apply(sample_normalize, axis=0)

    @staticmethod
    def _get_norm_factor(count_table, dna_amount, exclude_spike_in=True, spike_in_members=None):
        """Calculate the norm factor for per seq amount

        Args:
            count_table (`pd.DataFrame`): contains count info
            dna_amount (`dict`): a dictionary {sample_name: dna_amount}
            exclude_spike_in (`bool`): don't include spike-in sequences in total amount if True
            spike_in_members (list of `str`): list of spike_in sequences

        Returns:
            a `dict`: {sample_name: norm_factor}
        """

        if exclude_spike_in is True:
            if spike_in_members is None:
                raise ValueError('spike_in_members is None')
            else:
                count_table = count_table[~count_table.index.isin(spike_in_members)]

        return {sample: dna_am/count_table[sample].sum() for sample, dna_am in dna_amount.items()}

    def apply(self, target=None, dna_amount=None, exclude_spike_in=True, spike_in_members=None):
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if dna_amount is None:
            dna_amount = self.dna_amount
        if dna_amount is None:
            raise ValueError('No valid dna_amount found')
        if spike_in_members is None:
            spike_in_members = self.spike_in_members
        if spike_in_members is None:
            # try to extract if still None
            try:
                spike_in_members = target.spike_in.spike_in_members
            except:
                raise ValueError('No valid spike_in_members found')
        if isinstance(target, pd.DataFrame):
            pass
        else:
            target = target.table
        self.func(target=target.table,
                  norm_factor=self._get_norm_factor(target, dna_amount=dna_amount, exclude_spike_in=exclude_spike_in,
                                                    spike_in_members=spike_in_members))


class ReactedFractionNormalizer(TransformerBase):
    """Get reacted fraction of each sequence from an absolute amount table"""

    def __init__(self, target, input_pools, abs_amnt_table=None, reduce_method='median', remove_zero=True):
        super().__init__()
        self.target = target
        self.input_pools = input_pools
        self.abs_amnt_table = abs_amnt_table
        self.reduce_method = reduce_method
        self.remove_zero = True

    @staticmethod
    def func(target, input_pools, reduce_method='median'):
        """Convert absolute amount to reacted fraction

        Args:
            target (`pd.DataFrame`): the table of absolute amount, including input pools
            input_pools (list of `str`): list of indices of input pools
            reduce_method (str or callable):

        Returns:

        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError('target is not a pd.DataFrame')
        method_mapper = {
            'med': np.nanmedian,
            'median': np.nanmedian,
            'mean': np.nanmean,
            'avg': np.nanmean
        }
        if not callable(reduce_method):
            if reduce_method.lower() not in method_mapper.keys():
                raise ValueError('Unknown reduce_method')
            else:
                reduce_method = method_mapper[reduce_method]
        base = reduce_method(target[input_pools], axis=1)

        return target.loc[:, ~target.columns.isin(input_pools)].divide(base, axis=0)

    def apply(self, target=None, abs_amnt_table=None, input_pools=None, reduce_method=None, remove_zero=None):
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if abs_amnt_table is None:
            abs_amnt_table = self.abs_amnt_table
        if input_pools is None:
            input_pools = self.input_pools
        if input_pools is None:
            # try to extract from target
            try:
                input_pools = target.grouper.input.group
            except:
                raise ValueError('No input_pools found')
        if reduce_method is None:
            reduce_method = self.reduce_method

        if not isinstance(target, pd.DataFrame):
            if abs_amnt_table is None:
                target = getattr(target, 'table')
            else:
                target = getattr(target, abs_amnt_table)

        if remove_zero is None:
            remove_zero = self.remove_zero

        frac_table = self.func(target=target, input_pools=input_pools, reduce_method=reduce_method)
        if remove_zero:
            return frac_table[frac_table.sum(axis=1) > 0]
        else:
            return frac_table
