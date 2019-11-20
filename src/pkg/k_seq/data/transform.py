"""
Module contains the classes to transform tables (`pd.DataFrame` or `SeqTable`) instance including
    - normalize by spike-in
    - normalize by total amount
    - transform to relative abundance
    - etc
"""

import numpy as np
import pandas as pd
from .seq_table import SeqTable


class Transformer(object):
    """Base class type for transformer

    Necessary components for a Transformer
        - Attributes to store parameters
        - A target table to apply transformation
        - A static `func` function to calculate transformation
        - A `apply` function to wrap func
    """

    def __init__(self, target=None):
        self.target = target
        pass

    @staticmethod
    def _func(target):
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

        return self._func(target=target)


class SpikeInNormalizer(Transformer):
    """
    Normalizer using spike-in information
    todo: add attributes of spike-in
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
        self.radius = radius # calculate radius calculates the norm factor

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
            self._spike_in_amount = {**self._spike_in_amount, **{key: np.nan for key in self.blacklist}}
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
    def _func(target, norm_factor, **kwargs):

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

        return self._func(target=target, norm_factor=norm_factor)


def spike_in_peak_plot(spike_in, seq_table=None, sample_list=None, max_dist=15, norm_on_center=True, log_y=True,
                       marker_list=None, color_list=None, err_guild_lines=None,
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

    for sample, color, marker in zip(sample_list, color_list, marker_list):
        ax.plot(dist_series, peak_counts[sample], marker, color=color, label=sample,
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


class DnaAmountNormalizer(Transformer):
    """Normalize the DNA amount by total DNA amount in each sample
    """

    def __init__(self, dna_amount, target=None, unit=None):
        super().__init__(target)
        self.target = target
        self.dna_amount = dna_amount
        self.unit = unit
        self.norm_factor = None

    @staticmethod
    def _func(target, norm_factor):
        """Norm factor here should be per seq amount calculated from total DNA amount"""

        def sample_normalize(sample):
            return sample * norm_factor[sample.name]

        return target.apply(sample_normalize, axis=0)

    def apply(self, target=None, dna_amount=None):
        """Transform counts to absolute amount based on given total DNA amount

            Args:
                target (pd.DataFrame): contains counts of sequences in samples

                dna_amount (dict): a dictionary {sample_name: total_dna_amount}

                    exclude_spike_in (bool): if exclude spike-in sequences in total amount
                    spike_in_members (list of str): list of spike_in sequences

                Returns:
                    a dict: {sample_name: norm_factor}
                    where sequence amount = sequence counts * norm_factor
        """

        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if dna_amount is None:
            dna_amount = self.dna_amount
        if dna_amount is None:
            raise ValueError('No valid dna_amount found')
        if not isinstance(target, pd.DataFrame):
            target = target.table
        self.norm_factor = {sample: dna_am / target[sample].sparse.to_dense().sum()
                            for sample, dna_am in dna_amount.items()}
        self.norm_factor = {**self.norm_factor,
                            **{sample: np.nan for sample in target.columns if sample not in dna_amount.keys()}}
        return self._func(target=target, norm_factor=self.norm_factor)


class ReactedFractionNormalizer(Transformer):
    """Get reacted fraction of each sequence from an absolute amount table"""

    def __init__(self, input_samples, target=None, reduce_method='median', remove_zero=True):
        super().__init__()
        self.target = target
        self.input_samples = input_samples
        self.reduce_method = reduce_method
        self.remove_zero = remove_zero

    @staticmethod
    def _func(target, input_samples, reduce_method='median', remove_zero=True):

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

        mask = base > 0 # if any does not exist in input samples
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

        return self._func(target=target, input_samples=input_samples,
                          reduce_method=reduce_method, remove_zero=remove_zero)
