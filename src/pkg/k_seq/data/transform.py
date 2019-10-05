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

    def func(self, target):
        """input an `SeqTable` or `pd.Dataframe`, output an transformed `pd.Dataframe`"""
        pass

    def apply(self, target=None, add_to_SeqTable_as=None):
        """Run the transformation"""
        if target is None:
            target = self.target
        if target is None:
            raise ValueError('No valid target found')
        if add_to_SeqTable_as is None:
            return self.func(target=target)
        else:
            setattr(self.target, add_to_SeqTable_as, self.func(target=target))


class SpikeInNormalizer(TransformerBase):
    """this is the temporary version of spike-in normalizer, depending on CountFileSet info"""

    def __init__(self, spike_in_seq, spike_in_amount, target, radius=2, unit='ng', blacklist=None):
        if not isinstance(target, (SeqTable, pd.DataFrame)):
            raise TypeError('target for SpikeInNormalizer needs to be either `SeqTable` or `pd.Dataframe`')
        super().__init__(target)
        self.spike_in_seq = spike_in_seq
        self.radius = radius
        self.unit = unit
        if isinstance(spike_in_amount, (list, np.ndarray)):
            if isinstance(target, pd.DataFrame):
                sample_list = target.columns
            else:
                sample_list = target.sample_list
            if blacklist is not None:
                sample_list = [sample for sample in sample_list if sample not in blacklist]
            self.spike_in_amount = {key: value for key, value in zip(sample_list, spike_in_amount)}
        elif isinstance(spike_in_amount, dict):
            self.spike_in_amount = spike_in_amount
        else:
            raise TypeError('spike_in_amount needs to be list or dict')
        if isinstance(self.target, pd.DataFrame):
            self.dist_to_center = pd.Series(self.target.index.map(self._get_edit_dist), index=self.target.index)
        else:
            self.dist_to_center = pd.Series(self.target.table.index.map(self._get_edit_dist),
                                            index=self.target.table.index)

    def _get_edit_dist(self, seq):
        from Levenshtein import distance
        return distance(self.spike_in_seq, seq)

    @property
    def spike_in_members(self):
        return self.dist_to_center[self.dist_to_center <= self.radius].index.values

    def _sample_normalize(self, sample):
        if sample.name in self.spike_in_amount.keys():
            norm_factor = self.spike_in_amount[sample.name] / sample[self.spike_in_members].sparse.to_dense().sum()
            return sample * norm_factor
        else:
            import warnings
            warnings.warn(f'Sample {sample.name} if not found in the normalizer, normalization is not performed')
            return sample

    def func(self, target):
        if isinstance(target, pd.DataFrame):
            table = target
        elif hasattr(target, 'table'):
            table = target.table
        else:
            raise TypeError('table is not a recognized type')

        return table.apply(self._sample_normalize, axis=0)

    # def spike_in_peak_plot(seq_table, black_list=None, max_dist=15,
    #                        norm_on_center=True, log_y=True, accumulate=False,
    #                        marker_list=None, color_list=None, guild_lines=None,
    #                        legend_off=False, ax=None, save_fig_to=None):
    #     """Plot the distribution of spike_in peak
    #     Plot a scatter-line plot of [adjusted] number of sequences with i edit distance from center sequence (spike-in seq)
    #
    #     Args:
    #         sample_set (`SeqSampleSet`): dataset to plot
    #         black_list (list of `str`): to exclude some samples if not `None`
    #         max_dist (`int`): maximal edit distance to survey. Default 15
    #         norm_on_center (`bool`): if the counts/abundance are normalized to then center (exact spike in)
    #         log_y (`bool`): if set the y scale as log
    #         accumulate (`bool`): if show the accumulated abundance within i edit distance instead
    #         marker_list (list of `str`): overwrite default marker scheme if not `None`, same length and order as valid samples
    #         color_list (list of `str`): overwrite default color scheme if not `None`, same length and order as valid samples
    #         guild_lines (list of `float`): add a series of guild lines indicate the distribution only from given error rate,
    #           if not `None`
    #         legend_off (`bool`): do not show the legend if True
    #         ax (`matplotlib.Axis`): if use external ax object to plot. Create a new figure if `None`
    #         fig_save_to (`str`): save the figure as ``.jpeg`` file if not `None`
    #
    #     """
    #
    #     from k_seq.utility import PlotPreset
    #     import numpy as np
    #
    #     if black_list is None:
    #         black_list = []
    #     samples_to_plot = [sample for sample in sample_set.sample_set if sample.name not in black_list]
    #     if marker_list is None:
    #         marker_list = PlotPreset.markers(num=len(samples_to_plot), with_line=True)
    #     elif len(marker_list) != len(samples_to_plot):
    #         raise Exception('Error: length of marker_list does not align with the number of valid samples to plot')
    #
    #     if color_list is None:
    #         color_list = PlotPreset.colors(num=len(samples_to_plot))
    #     elif len(color_list) != len(samples_to_plot):
    #         raise Exception('Error: length of color_list does not align with the number of valid samples to plot')
    #
    #     if ax is None:
    #         if legend_off:
    #             fig = plt.figure(figsize=[10, 8])
    #         else:
    #             fig = plt.figure(figsize=[16, 8])
    #         ax = fig.add_subplot(111)
    #         show_ax = True
    #     else:
    #         show_ax = False
    #
    #     for sample, color, marker in zip(samples_to_plot, color_list, marker_list):
    #         if not hasattr(sample, 'spike_in'):
    #             raise Exception('Error: please survey the spike-in counts before plot')
    #         elif len(sample.spike_in['spike_in_peak']['total_counts'] < max_dist + 1):
    #             spike_in = sample.survey_spike_in_peak(spike_in_seq=sample.spike_in['spike_in_seq'],
    #                                                    max_dist_to_survey=max_dist,
    #                                                    silent=True, inplace=False)
    #         else:
    #             spike_in = sample.spike_in
    #
    #         if accumulate:
    #             counts = np.array(
    #                 [np.sum(spike_in['spike_in_peak']['total_counts'][:i + 1]) for i in range(max_dist + 1)])
    #         else:
    #             counts = np.array(spike_in['spike_in_peak']['total_counts'][:max_dist + 1])
    #         if norm_on_center:
    #             counts = counts / counts[0]
    #         ax.plot([i for i in range(max_dist + 1)], counts, marker, color=color,
    #                 label=sample.name, alpha=0.5)
    #     if guild_lines:
    #         from scipy.stats import binom
    #         for ix, p in enumerate(guild_lines):
    #             rv = binom(len(spike_in['spike_in_seq']), p)
    #             pmfs = np.array([rv.pmf(x) for x in range(max_dist)])
    #             pmfs_normed = pmfs / pmfs[0]
    #             ax.plot([i for i in range(max_dist)], pmfs_normed,
    #                     color='k', ls='--', alpha=(ix + 1) / len(guild_lines), label='p={}'.format(p))
    #     if log_y:
    #         ax.set_yscale('log')
    #     y_label = ''
    #     if norm_on_center:
    #         y_label += ' normed'
    #
    #     if accumulate:
    #         y_label += ' accumulated'
    #
    #     y_label += ' counts'
    #     ax.set_ylabel(y_label.title(), fontsize=14)
    #     ax.set_xlabel('Edit Distance to Spike-in Center', fontsize=14)
    #     if not legend_off:
    #         ax.legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
    #     plt.tight_layout()
    #
    #     if save_fig_to:
    #         plt.savefig(save_fig_to, bbox_inches='tight', dpi=300)
    #     if show_ax:
    #         plt.show()
    #
    # def rep_spike_in_plot(sample_set, group_by, plot_spike_in_frac=True, plot_entropy_eff=True,
    #                       ax=None, save_fig_to=None):
    #     """Scatter plot to show the variability (outliers) for each group of sample on
    #         - spike in fraction if applicable
    #         - entropy efficiency of the pool
    #
    #     Args:
    #         sample_set (`SeqSampleSet`): sample_set to plot
    #         group_by (`list` of `list`, `dict` of `list`, or `str`): indicate the grouping of samples. `list` of `list` to
    #           to contain sample names in each group as a nested `list`, or named group as a `dict`, or group on attribute
    #           using `str`, e.g. 'byo'
    #         plot_spike_in_frac (`bool`): if plot the fraction of spike-in seq in each sample
    #         plot_entropy_eff (`bool`): if plot the entropy efficiency for each sample
    #         ax (`matplotlib.Axes`): plot in a given axis is not None
    #         save_fig_to (`str`): directory to save the figure
    #
    #     Returns:
    #
    #     """
    #
    #     from k_seq.utility import PlotPreset
    #
    #     tagged_sample_set = {sample.name: sample for sample in sample_set.sample_set}
    #     groups = {}
    #     if isinstance(group_by, list):
    #         for ix, group in enumerate(group_by):
    #             if not isinstance(group, list):
    #                 raise Exception('Error: if use list, group by should be a 2-D list of sample names')
    #             else:
    #                 groups['Set_{}'.format(ix)] = [tagged_sample_set[sample_name] for sample_name in group]
    #     elif isinstance(group_by, dict):
    #         for group in group_by.values():
    #             if not isinstance(group, list):
    #                 raise Exception('Error: if use dict, all values should be a 1-D list of sample names')
    #         groups = {
    #             group_name: [tagged_sample_set[sample_name] for sample_name in group]
    #             for group_name, group in group_by.items()
    #         }
    #     elif isinstance(group_by, str):
    #         groups = {}
    #         for sample in sample_set.sample_set:
    #             if group_by not in sample.metadata.keys():
    #                 raise Exception('Error: sample {} does not have attribute {}'.format(sample.name, group_by))
    #             else:
    #                 if sample.metadata[group_by] in groups:
    #                     groups[sample.metadata[group_by]].append(sample)
    #                 else:
    #                     groups[sample.metadata[group_by]] = [sample]
    #
    #     if plot_spike_in_frac + plot_entropy_eff == 2:
    #         if ax is None:
    #             fig, axes = plt.subplots(2, 1, figsize=[12, 12], sharex=True)
    #             fig.subplots_adjust(wspace=0, hspace=0)
    #             ax_show = True
    #         else:
    #             ax_show = False
    #     elif plot_spike_in_frac + plot_entropy_eff == 1:
    #         if ax is None:
    #             fig = plt.figure(figsize=[12, 6])
    #             ax = fig.add_subplot(111)
    #             ax_show = True
    #         else:
    #             ax_show = False
    #         axes = [ax, ax]
    #
    #     def get_spike_in_frac(sample):
    #         return np.sum(sample.spike_in['spike_in_peak']['total_counts'][
    #                       :sample.spike_in['quant_factor_max_dist']]) / sample.total_counts
    #
    #     def get_entropy_eff(sample):
    #         sequences = sample.sequences(with_spike_in=False)['counts']
    #         seq_rel_abun = sequences / sequences.sum()
    #         return -np.sum(np.multiply(seq_rel_abun, np.log2(seq_rel_abun))) / np.log2(len(seq_rel_abun))
    #
    #     # texts1 = []
    #     # texts2 = []
    #     markers = PlotPreset.markers(num=len(groups))
    #     for ix, (samples, marker) in enumerate(zip(groups.values(), markers)):
    #         for sample, color in zip(samples, PlotPreset.colors(num=len(samples))):
    #             if plot_spike_in_frac:
    #                 axes[0].scatter([ix], [get_spike_in_frac(sample)], marker=marker, color=color)
    #                 tx = axes[0].text(s=sample.name, x=ix + 0.1, y=get_spike_in_frac(sample), va='center', ha='left',
    #                                   fontsize=10)
    #                 # texts1.append(tx)
    #
    #             if plot_entropy_eff:
    #                 axes[1].scatter([ix], [get_entropy_eff(sample)], marker=marker, color=color)
    #                 tx = axes[1].text(s=sample.name, x=ix + 0.1, y=get_entropy_eff(sample), va='center', ha='left',
    #                                   fontsize=10)
    #                 # texts2.append(tx)
    #     # from adjustText import adjust_text
    #     # adjust_text(texts1, arrowprops=dict(arrowstyle='->', color='#151515'))
    #     # adjust_text(texts2, arrowprops=dict(arrowstyle='->', color='#151515'))
    #     axes[0].set_ylabel('Spike in fraction', fontsize=14)
    #     axes[1].set_ylabel('Entropy Efficiency', fontsize=14)
    #     axes[1].set_xlim([-0.5, len(groups)])
    #     axes[1].set_xticks([ix for ix in range(len(groups))])
    #     axes[1].set_xticklabels(groups.keys())
    #
    #     if save_fig_to:
    #         fig.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    #     if ax_show:
    #         plt.show()


class TotalAmountNormalizer(TransformerBase):
    """this is the temporary version of spike-in normalizer, depending on CountFileSet info"""

    def __init__(self, target=None):
        super().__init__(target)

    def func(self, target):
        if isinstance(target, pd.DataFrame):
            raise TypeError('`pd.DataFrame` is not supported for SpikeInNormalizer')

        def sample_normalize(sample):
            norm_factor = target.metadata.samples[sample]['spike-in']['norm_factor']
            return sample * norm_factor

        return target.table.apply(sample_normalize, axis=0)


class ReactedFractionNormalizer(TransformerBase):
    """Get the reacted fraction from table"""


# class SpikeInNormalizer(TransformerBase):
#
#     def __init__(self, target, spike_in_info=None, spike_in_seq=None, spike_in_amount=None, spike_in_dia=2, unit=None):
#         """
#         Add spike-in sequences for all samples, for each parameter, single value or dictionary could be passed,
#         if a dictionary is passed, the key should be sample name and explicitly list parameters for all samples
#         """
#         from .count_file import SpikeIn
#
#         super().__init__(target)
#         from ..utility.func_tools import param_to_dict
#         if hasattr(target, 'sample_list'):
#             sample_keys = target.sample_list
#         elif hasattr(target, 'columns'):
#             sample_keys = target.columns.to_series()
#         else:
#             raise TypeError('Can\'t identify target type, it should be `SeqTable` or `pd.Dataframe`')
#
#         spike_in_config = param_to_dict(key_list=sample_keys,
#                                    spike_in_info=spike_in_info, spike_in_seq=spike_in_seq,
#                                    spike_in_amount=spike_in_amount, spike_in_dia=spike_in_dia,
#                                    unit=unit)
#         self.spike_in
#         for key, args in param_dict.items():
#             self._sample_indexer[key].add_spike_in(**args)
#
#     def _get_edit_distance(self, seq):
#         """use `python-levenshtein` to calculate edit distances of seqs to the center"""
#         import pandas as pd
#         from Levenshtein import distance
#         if isinstance(seq, pd.Series):
#             seq = seq.name
#         return distance(seq, self.center)
#
#     def func(self, target=None):