"""
This module contains the methods used for k-seq dataset analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch
from . import pre_processing
from IPython.display import HTML


# -------------- Belows are SeqSample Visualizer ---------------

def length_dist_plot_single(sample, y_log=True, legend_off=False, title_off=False, labels_off=False,
                            ax=None, save_fig_to=None):
    """
    To plot a histogram of sequence length for a single sample

    Args:
        sample (`SeqSample`): the sample to plot
        y_log (`bool`): set y scale as log if True
        legend_off (`bool`): do not show legend if True
        title_off (`bool`): do not show title if True; use `sample.name` as title if False
        labels_off (`bool`): do not show x label if True
        ax (`matplotlib.Axes`): plot in a given axis if not None
        save_fig_to (`str`): if save file to a given directory
    """

    if ax is None:
        fig = plt.figure(figsize=[6, 4])
        ax = fig.add_subplot(111)
        show_fig = True
    else:
        show_fig = False

    sample.sequences['length'] = sample.sequences.index.map(mapper=len)
    bins = np.linspace(min(sample.sequences['length']), max(sample.sequences['length']), 50)
    ax.hist(sample.sequences['length'], bins=bins, weights=sample.sequences['counts'], color='#AEAEAE',
            zorder=1, label='counts')
    ax.hist(sample.sequences['length'], bins=bins, color='#2C73B4', zorder=2, label='unique seqs')
    if y_log:
        ax.set_yscale('log')
    if not labels_off:
        ax.set_xlabel('Sequence Length (nt)', fontsize=14)
    if not title_off:
        ax.set_title(sample.name, fontsize=14)
    if not legend_off:
        ax.legend(frameon=False)
    if save_fig_to:
        plt.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()


def sample_count_cut_off_plot_single(sample, thresholds=None, on_counts=False, include_spike_in=False,
                                     x_log=False, y_log=True, legend_off=False, labels_off=False,
                                     ax=None, save_fig_to=None):
    """Plot cutoff test on sequences on single file
    Args:
        sample (`SeqSample`): sample to plot
        thresholds (list of `float`): optional, manual input of thresholds for cutoff
        on_counts (`bool`): cutoff is calculated on absolute counts if True, on relative abundance if False
        include_spike_in (`bool`): if remove spike in sequences before calculation
        x_log (`bool`): if set x axis as log
        y_log (`bool`): if set y axis as log
        legend_off:
        labels_off:
        ax:
        save_fig_to:

    Returns:

    """

    import numpy as np

    if not include_spike_in:
        sequences = sample.remove_spike_in(inplace=False, silent=True)
    else:
        sequences = sample.sequences

    if not on_counts:
        series_to_use = sequences['counts']/np.sum(sequences['counts'])
    else:
        series_to_use = sequences['counts']

    if thresholds is None:
        if on_counts:
            thresholds = np.logspace(0, np.log10(sample.sequences['counts'].max()), 10)
        else:
            thresholds = np.logspace(np.log10(1/sample.total_counts),
                                     np.log10(sample.sequences['counts'].max()/sample.total_counts),
                                     10)

    filtered_uniques = [np.sum(series_to_use >= threshold) for threshold in thresholds]
    filtered_total = [np.sum(series_to_use[series_to_use >= threshold]) for threshold in thresholds]

    if ax is None:
        fig = plt.figure(figsize=[8, 6])
        ax = fig.add_subplot(111)
        fig_show = True
    else:
        fig_show = False

    ax.plot(thresholds, filtered_uniques, 'o-', color='#2C73B4', label='Unique seqs')
    ax2 = ax.twinx()
    if on_counts:
        label = 'Total counts'
    else:
        label = 'Total rel abun'
    ax2.plot(thresholds, filtered_total, 'o-', color='#F39730', label=label)
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    if on_counts:
        ax2.set_ylim([1, np.sum(filtered_total[0] * 1.2)])
    else:
        ax2.set_ylim([1/(sample.total_counts * 1.2), 1.2])

    if not legend_off:
        handles = [plt.Line2D([], [], marker='o', linestyle='-', color='#2C73B4'),
                   plt.Line2D([], [], marker='o', linestyle='-', color='#F39730')]
        labels = ['Unique seqs', label]
        plt.legend(handles=handles, labels=labels, frameon=False)

    if not labels_off:
        ax.set_xlabel('Cut off threshold', fontsize=14)
        ax.set_ylabel('Unique Sequences', fontsize=14)
        ax2.set_ylabel(label, fontsize=14)

    if save_fig_to:
        plt.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    if fig_show:
        plt.show()


# --------------------- Belows are SeqSampleSet visualizers ----------------------

def count_file_info_table(sample_set, return_table=False):
    """Generate an overview info table for all samples in sample_set
    Print out HTML table for sequencing samples, including spike-in info if applicable

    Args:
        sample_set (`SeqSampleSet`): sample set to survey
        return_table (`bool`): return a `pd.DataFrame` if True

    Returns: a `pd.DataFrame` equivalence if `return_table` is True

    """

    sample_set = sample_set.sample_set
    table = pd.DataFrame()
    table['sample type'] = [sample.sample_type for sample in sample_set]
    table['name'] = [sample.name for sample in sample_set]
    table['total counts'] = [sample.total_counts for sample in sample_set]
    table['unique sequences'] = [sample.unique_seqs for sample in sample_set]
    table['x_value'] = [sample.x_value for sample in sample_set]
    if hasattr(sample_set[0], 'spike_in'):
        table['spike-in amount'] = [sample.spike_in['spike_in_amount'] for sample in sample_set]
        table['spike-in counts (dist={})'.format(sample_set[0].spike_in['quant_factor_max_dist'])] = [
            np.sum(sample.spike_in['spike_in_counts'][0:sample.spike_in['quant_factor_max_dist'] + 1])
            for sample in sample_set
        ]
        table['spike-in percent'] = [
            np.sum(sample.spike_in['spike_in_counts'][0:sample.spike_in['quant_factor_max_dist'] + 1])/sample.total_counts
            for sample in sample_set
        ]
        table['quantification factor'] = [sample.quant_factor for sample in sample_set]
        table_html = table.to_html(
            formatters={
                'total counts': lambda x: '{:,}'.format(x),
                'unique sequences': lambda x: '{:,}'.format(x),
                'spike-in counts (dist{})'.format(sample_set[0].spike_in['quant_factor_max_dist']): lambda x: '{:,}'.format(x),
                'spike-in percent': lambda x: '{:.3f}'.format(x),
                'quantification factor': lambda x:'{:.3e}'.format(x)
            }
        )
    else:
        table_html = table.to_html(
            formatters={
                'total counts': lambda x: '{:,}'.format(x),
                'unique sequences': lambda x: '{:,}'.format(x)
            }
        )
    if return_table:
        return table
    else:
        display(HTML(table_html))


def count_file_info_plot(sample_set, plot_unique_seq=True, plot_total_counts=True, plot_spike_in_frac=True,
                         black_list=None, sep_plot=False, fig_save_to=None):
    """Generate overview plot(s) of unique seqs, total counts and spike-in fractions in the samples

    Args:

        sample_set (`SeqSampleSet`): sample set to survey

        plot_unique_seq (`bool`): plot bar plot for unique sequences if True

        plot_total_counts (`bool`): plot bar plot for total counts if True

        plot_spike_in_frac (`bool`): plot scatter plot for spike in fraction if True

        black_list (list of `str`): list of sample name to exlude from the plots

        sep_plot (`bool`): plot separate plots for unique sequences, total counts and spike_in fractions if True

        fig_save_to (`str`): save figure to the directory if not None

    """
    if black_list is None:
        black_list = []
    sample_set = [sample for sample in sample_set.sample_set if sample.name not in black_list]
    sample_num = len(sample_set)
    plot_num = np.sum(np.array([plot_unique_seq, plot_total_counts, plot_spike_in_frac]))
    if sep_plot and plot_num > 1:
        fig, axes = plt.subplots(plot_num, 1, figsize=[sample_num * 0.5, 3 * plot_num],
                                 sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0)
        ax_ix = 0
        if plot_unique_seq:
            ax = axes[ax_ix]
            ax.bar(x=[i for i in range(sample_num)],
                   height=[sample.unique_seqs for sample in sample_set],
                   align='center', width=0.6, color='#2C73B4')
            ax.set_ylabel('Number of unique seqs', fontsize=12)
            ax_ix += 1
        if plot_total_counts:
            ax = axes[ax_ix]
            ax.bar(x=[i for i in range(sample_num)],
                   height=[sample.total_counts for sample in sample_set],
                   align='center', width=0.6, color='#FC820D')
            ax.set_ylabel('Number of total counts', fontsize=12)
            ax_ix += 1
        if plot_spike_in_frac and hasattr(sample_set[0], 'spike_in'):
            ax = axes[ax_ix]
            ax.scatter([i for i in range(sample_num)],
                       [np.sum(sample.spike_in['spike_in_counts'][0:sample.spike_in['quant_factor_max_dist'] + 1]) / sample.total_counts
                         for sample in sample_set],
                        color='#B2112A', marker='x')
            ax.set_ylabel('Fraction of spike-in', fontsize=12)
            ax.plot([-0.5, sample_num - 0.5], [0.2, 0.2], '#B2112A', ls='--', alpha=0.3)
            ax.text(s='20%', x=-0.55, y=0.2, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax.plot([-0.5, sample_num - 0.5], [0.4, 0.4], '#B2112A', ls='--', alpha=0.3)
            ax.text(s='40%', x=-0.55, y=0.4, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax.plot([-0.5, sample_num - 0.5], [0.6, 0.6], '#B2112A', ls='--', alpha=0.3)
            ax.text(s='60%', x=-0.55, y=0.6, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax.plot([-0.5, sample_num - 0.5], [0.8, 0.8], '#B2112A', ls='--', alpha=0.3)
            ax.text(s='80%', x=-0.55, y=0.8, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax.set_ylim([0, 1])
        ax.set_xticks([i for i in range(sample_num)])
        ax.set_xticklabels([sample.name for sample in sample_set], rotation=90)
        fig.align_ylabels(axes)
    else:
        fig = plt.figure(figsize=[sample_num * 0.5, 6])
        ax = fig.add_subplot(111)
        lgd = []
        if plot_unique_seq:
            if plot_total_counts:
                shift = 0.2
            else:
                shift = 0.0
            ax.bar(x=[i - shift for i in range(sample_num)],
                   height=[sample.unique_seqs for sample in sample_set],
                   align='center',width=0.4, color='#2C73B4')
            lgd.append(mpatch.Patch(color='#2C73B4', label='Unique seqs'))
            ax.set_ylabel('Number of total reads in the sample', fontsize=14)

        if plot_total_counts:
            if plot_unique_seq:
                shift = 0.2
                ax2 = ax.twinx()
            else:
                shift = 0.0
                ax2 = ax
            ax2.bar(x=[i + shift for i in range(sample_num)],
                    height=[sample.total_counts for sample in sample_set],
                    align='center', width=0.4, color='#FC820D')
            lgd.append(mpatch.Patch(color='#FC820D', label='Total counts'))
            ax2.set_ylabel('Number of unique sequences in the sample', fontsize=14)

        if plot_spike_in_frac and hasattr(sample_set[0], 'spike_in'):
            ax3 = ax.twinx()
            ax3.scatter([i for i in range(sample_num)],
                        [np.sum(sample.spike_in['spike_in_counts'][0:sample.spike_in['quant_factor_max_dist'] + 1])/sample.total_counts
                         for sample in sample_set],
                        color='#B2112A', marker='x')
            ax3.plot([-0.5, sample_num - 0.5], [0.2, 0.2], '#B2112A', ls='--', alpha=0.3)
            ax3.text(s='20%', x=-0.55, y=0.2, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax3.plot([-0.5, sample_num - 0.5], [0.4, 0.4], '#B2112A', ls='--', alpha=0.3)
            ax3.text(s='40%', x=-0.55, y=0.4, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax3.plot([-0.5, sample_num - 0.5], [0.6, 0.6], '#B2112A', ls='--', alpha=0.3)
            ax3.text(s='60%', x=-0.55, y=0.6, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax3.plot([-0.5, sample_num - 0.5], [0.8, 0.8], '#B2112A', ls='--', alpha=0.3)
            ax3.text(s='80%', x=-0.55, y=0.8, ha='right', va='center', color='#B2112A', fontsize=10, alpha=0.5)
            ax3.set_ylim([0, 1])
            ax3.set_yticks([])
            lgd = lgd + [plt.plot([], [], lw=0, marker='x', color='#B2112A',
                                  label='Percent of spike-in')[0]]

        ax.set_xticks([i for i in range(sample_num)])
        ax.set_xticklabels([sample.name for sample in sample_set], rotation=90)
        plt.legend(handles=lgd, frameon=True)

    if fig_save_to:
        fig.savefig(fig_save_to, dpi=300)
    plt.show()


def spike_in_peak_plot(sample_set, black_list=None, max_dist=15,
                       norm_on_center=True, log_y=True, accumulate=False,
                       marker_list=None, color_list=None, guild_lines=None,
                       legend_off=False, ax=None, fig_save_to=None):
    """Plot the distribution of spike_in peak
    Plot a scatter-line plot of [adjusted] number of sequences with i edit distance from center sequence (spike-in seq)

    Args:
        sample_set (`SeqSampleSet`): dataset to plot
        black_list (list of `str`): to exclude some samples if not `None`
        max_dist (`int`): maximal edit distance to survey. Default 15
        norm_on_center (`bool`): if the counts/abundance are normalized to then center (exact spike in)
        log_y (`bool`): if set the y scale as log
        accumulate (`bool`): if show the accumulated abundance within i edit distance instead
        marker_list (list of `str`): overwrite default marker scheme if not `None`, same length and order as valid samples
        color_list (list of `str`): overwrite default color scheme if not `None`, same length and order as valid samples
        guild_lines (list of `float`): add a series of guild lines indicate the distribution only from given error rate,
          if not `None`
        legend_off (`bool`): do not show the legend if True
        ax (`matplotlib.Axis`): if use external ax object to plot. Create a new figure if `None`
        fig_save_to (`str`): save the figure as ``.jpeg`` file if not `None`

    """

    from k_seq.utility import PlotPreset
    import numpy as np

    if black_list is None:
        black_list = []
    samples_to_plot = [sample for sample in sample_set.sample_set if sample.name not in black_list]
    if marker_list is None:
        marker_list = PlotPreset.markers(num=len(samples_to_plot), with_line=True)
    elif len(marker_list) != len(samples_to_plot):
        raise Exception('Error: length of marker_list does not align with the number of valid samples to plot')

    if color_list is None:
        color_list = PlotPreset.colors(num=len(samples_to_plot))
    elif len(color_list) != len(samples_to_plot):
        raise Exception('Error: length of color_list does not align with the number of valid samples to plot')

    if ax is None:
        if legend_off:
            fig = plt.figure(figsize=[10, 8])
        else:
            fig = plt.figure(figsize=[16, 8])
        ax = fig.add_subplot(111)
        show_ax = True
    else:
        show_ax = False

    for sample, color, marker in zip(samples_to_plot, color_list, marker_list):
        if not hasattr(sample, 'spike_in'):
            raise Exception('Error: please survey the spike-in counts before plot')
        elif len(sample.spike_in['spike_in_counts'] < max_dist + 1):
            spike_in = sample.survey_spike_in(spike_in_seq=sample.spike_in['spike_in_seq'],
                                              max_dist_to_survey=max_dist,
                                              silent=True, inplace=False)
        else:
            spike_in = sample.spike_in

        if accumulate:
            counts = np.array([np.sum(spike_in['spike_in_counts'][:i + 1]) for i in range(max_dist + 1)])
        else:
            counts = np.array(spike_in['spike_in_counts'][:max_dist + 1])
        if norm_on_center:
            counts = counts/counts[0]
        ax.plot([i for i in range(max_dist + 1)], counts, marker, color=color,
                label=sample.name, alpha=0.5)
    if guild_lines:
        from scipy.stats import binom
        for ix, p in enumerate(guild_lines):
            rv = binom(len(spike_in['spike_in_seq']), p)
            pmfs = np.array([rv.pmf(x) for x in range(max_dist)])
            pmfs_normed = pmfs / pmfs[0]
            ax.plot([i for i in range(max_dist)], pmfs_normed,
                    color='k', ls='--', alpha=(ix + 1)/len(guild_lines), label='p={}'.format(p))
    if log_y:
        ax.set_yscale('log')
    y_label = ''
    if norm_on_center:
        y_label += ' normed'

    if accumulate:
        y_label += ' accumulated'

    y_label += ' counts'
    ax.set_ylabel(y_label.title(), fontsize=14)
    ax.set_xlabel('Edit Distance to Spike-in Center', fontsize=14)
    if not legend_off:
        ax.legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
    plt.tight_layout()

    if fig_save_to:
        plt.savefig(fig_save_to, bbox_inches='tight', dpi=300)
    if show_ax:
        plt.show()


def rep_spike_in_plot(sample_set, group_by, plot_spike_in_frac=True, plot_entropy_eff=True,
                      ax=None, save_fig_to=None):
    """Scatter plot to show the variability (outliers) for each group of sample on
        - spike in fraction if applicable
        - entropy efficiency of the pool

    Args:
        sample_set (`SeqSampleSet`): sample_set to plot
        group_by (`list` of `list`, `dict` of `list`, or `str`): indicate the grouping of samples. `list` of `list` to
          to contain sample names in each group as a nested `list`, or named group as a `dict`, or group on attribute
          using `str`, e.g. 'byo'
        plot_spike_in_frac (`bool`): if plot the fraction of spike-in seq in each sample
        plot_entropy_eff (`bool`): if plot the entropy efficiency for each sample
        ax (`matplotlib.Axes`): plot in a given axis is not None
        save_fig_to (`str`): directory to save the figure

    Returns:

    """

    from k_seq.utility import PlotPreset

    tagged_sample_set = {sample.name: sample for sample in sample_set.sample_set}
    groups = {}
    if isinstance(group_by, list):
        for ix, group in enumerate(group_by):
            if not isinstance(group, list):
                raise Exception('Error: if use list, group by should be a 2-D list of sample names')
            else:
                groups['Set_{}'.format(ix)] = [tagged_sample_set[sample_name] for sample_name in group]
    elif isinstance(group_by, dict):
        for group in group_by.values():
            if not isinstance(group, list):
                raise Exception('Error: if use dict, all values should be a 1-D list of sample names')
        groups = {
            group_name: [tagged_sample_set[sample_name] for sample_name in group]
            for group_name, group in group_by.items()
        }
    elif isinstance(group_by, str):
        groups = {}
        for sample in sample_set.sample_set:
            if group_by not in sample.metadata.keys():
                raise Exception('Error: sample {} does not have attribute {}'.format(sample.name, group_by))
            else:
                if sample.metadata[group_by] in groups:
                    groups[sample.metadata[group_by]].append(sample)
                else:
                    groups[sample.metadata[group_by]] = [sample]

    if plot_spike_in_frac + plot_entropy_eff == 2:
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=[12, 12], sharex=True)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax_show = True
        else:
            ax_show = False
    elif plot_spike_in_frac + plot_entropy_eff == 1:
        if ax is None:
            fig = plt.figure(figsize=[12, 6])
            ax = fig.add_subplot(111)
            ax_show = True
        else:
            ax_show = False
        axes = [ax, ax]

    def get_spike_in_frac(sample):
        return sample.spike_in['spike_in_counts'][sample.spike_in['quant_factor_max_dist']] / sample.total_counts

    def get_entropy_eff(sample):
        seq_rel_abun = sample.sequences['counts'] / sample.total_counts
        return -np.sum(np.multiply(seq_rel_abun, np.log2(seq_rel_abun))) / np.log2(len(seq_rel_abun))
    # texts1 = []
    # texts2 = []
    markers = PlotPreset.markers(num=len(groups))
    for ix, (samples, marker) in enumerate(zip(groups.values(), markers)):
        for sample, color in zip(samples, PlotPreset.colors(num=len(samples))):
            if plot_spike_in_frac:
                axes[0].scatter([ix], [get_spike_in_frac(sample)], marker=marker, color=color)
                tx = axes[0].text(s=sample.name, x=ix + 0.1, y=get_spike_in_frac(sample), va='center', ha='left', fontsize=10)
                # texts1.append(tx)

            if plot_entropy_eff:
                axes[1].scatter([ix], [get_entropy_eff(sample)], marker=marker, color=color)
                tx = axes[1].text(s=sample.name, x=ix + 0.1, y=get_entropy_eff(sample), va='center', ha='left', fontsize=10)
                # texts2.append(tx)
    # from adjustText import adjust_text
    # adjust_text(texts1, arrowprops=dict(arrowstyle='->', color='#151515'))
    # adjust_text(texts2, arrowprops=dict(arrowstyle='->', color='#151515'))
    axes[0].set_ylabel('Spike in fraction', fontsize=14)
    axes[1].set_ylabel('Entropy Efficiency', fontsize=14)
    axes[1].set_xlim([-0.5, len(groups)])
    axes[1].set_xticks([ix for ix in range(len(groups))])
    axes[1].set_xticklabels(groups.keys())

    if save_fig_to:
        fig.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    if ax_show:
        plt.show()





def length_dist_plot_all(sample_set, black_list=None, fig_layout=None, y_log=True, save_fig_to=None):
    """
    Wrapper of `length_dist_plot_single` to plot histogram for all given samples
    Args:
        sample_set (`SeqSampleSet`): sample set to use
        black_list (list of `str`): optional, exclude samples with given names if not None
        fig_layout (``(nrow, ncol)``): optional, indicate layout of a figure. 4 figs in a row if None.
        y_log (`bool`): set y scale as log if True
        save_fig_to (`str`): optional, directory to save the figure

    """

    if black_list is None:
        black_list = []
    samples_to_plot = [sample for sample in sample_set.sample_set if sample.name not in black_list]
    if fig_layout is None:
        from math import ceil
        fig_layout = [ceil(len(samples_to_plot) / 4), 4]
    fig, axes = plt.subplots(fig_layout[0], fig_layout[1], figsize=[fig_layout[1] * 3, fig_layout[0] * 2])
    for ix, sample in enumerate(samples_to_plot):
        ax = axes[int(ix/fig_layout[1]), ix % fig_layout[1]]
        length_dist_plot_single(sample, y_log=y_log, legend_off=True, title_off=True, labels_off=True, ax=ax)
        ax.set_title(sample.name, fontsize=10)
    import matplotlib.patches as mpatch
    handle = [mpatch.Patch(color='#AEAEAE', label='Counts'), mpatch.Patch(color='#2C73B4', label='Unique Seqs')]
    fig.legend(handles=handle, loc=(0.7, 0), frameon=False, ncol=2)
    fig.text(s='Sequence Length (nt)', x=0.5, y=0, ha='center', va='top', fontsize=16)
    plt.tight_layout()
    if save_fig_to:
        fig.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    plt.show()




def sample_count_cut_off_plot_all(sample_set, black_list=None, thresholds=None, on_counts=False, include_spike_in=False,
                                  fig_layout=None, y_log=True, x_log=True, save_fig_to=None):
    """Plot cutoff tests on a sample set

    Args:
        sample_set (`SeqSampleSet`): sample set to work
        black_list (list of `str`): optional. List of sample names to exclude
        thresholds (list of `float`): optional. manual list of thresholds to apply
        on_counts (`bool`): cutoff is calculated on absolute counts if True, on relative abundance if False
        include_spike_in (`bool`): if remove spike in sequences before calculation
        x_log (`bool`): if set x axis as log
        y_log (`bool`): if set y axis as log
        fig_layout ((`int`, `int`)): a tuple of (ncol, nrow) to indicate the layout of plot
        save_fig_to:

    """
    if black_list is None:
        black_list = []
    samples_to_plot = [sample for sample in sample_set.sample_set if sample.name not in black_list]
    if fig_layout is None:
        from math import ceil
        fig_layout = [ceil(len(samples_to_plot) / 4), 4]
    fig, axes = plt.subplots(fig_layout[0], fig_layout[1], figsize=[fig_layout[1] * 3, fig_layout[0] * 2])
    if thresholds is None:
        thresholds = None
    for ix, sample in enumerate(samples_to_plot):
        ax = axes[int(ix/fig_layout[1]), ix % fig_layout[1]]
        sample_count_cut_off_plot_single(sample, thresholds=thresholds, on_counts=on_counts,
                                         include_spike_in=include_spike_in,
                                         x_log=x_log, y_log=y_log, legend_off=True, labels_off=True,
                                         ax=ax)
        ax.set_title(sample.name, fontsize=10)
    if on_counts:
        label = 'Total counts'
    else:
        label = 'Total rel abun'
    handle = [plt.Line2D([], [], marker='o', linestyle='-', color='#2C73B4', label='Unique Seqs'),
              plt.Line2D([], [], marker='o', linestyle='-', color='#F39730', label=label)]
    fig.legend(handles=handle, loc=(0.7, 0), frameon=False, ncol=2)
    fig.text(s='Cut-off Threshold', x=0.5, y=0, ha='center', va='top', fontsize=16)
    fig.text(s='Unique Sequences', x=0, y=0.5, ha='right', va='top', fontsize=16, rotation=90)
    fig.text(s=label, x=1, y=0.5, ha='left', va='top', fontsize=16, rotation=90)
    plt.tight_layout()
    if save_fig_to:
        fig.savefig(save_fit_to, bbox_inches='tight', dpi=300)
    plt.show()


# ---------------------- Belows are SeqTable Visualizers --------------------

def seq_occurrence_plot(seq_table, sample_range='reacted', fig_save_to=None):
    import numpy as np

    if sample_range == 'reacted':
        sample_num = seq_table.count_table_reacted.shape[1]
        occurrence = seq_table.seq_info['occurred_in_reacted']
        rel_abun = seq_table.seq_info['avg_rel_abun_in_reacted']
    elif sample_range == 'inputs':
        sample_num = seq_table.count_table_inputs.shape[1]
        occurrence = seq_table.seq_info['occurred_in_inputs']
        rel_abun = seq_table.seq_info['avg_rel_abun_in_inputs']
    else:
        sample_num = seq_table.count_table_inputs.shape[1]
        occurrence = seq_table.seq_info['occurred_in_inputs']
        rel_abun = seq_table.seq_info['avg_rel_abun_in_inputs']
        sample_num += seq_table.count_table_reacted.shape[1]
        occurrence += seq_table.seq_info['occurred_in_reacted']
        rel_abun += seq_table.seq_info['avg_rel_abun_in_reacted']

    count_bins = np.bincount(occurrence, minlength=sample_num + 1)[1:]
    count_bins_weighted = np.bincount(occurrence, minlength=sample_num + 1, weights=rel_abun)[1:]

    fig = plt.figure(figsize=[16, 8])
    gs = gridspec.GridSpec(2, 3, figure=fig)

    occur_num = [i+1 for i in range(sample_num)]
    ax11 = fig.add_subplot(gs[0, 0])
    ax11.pie(x=count_bins, labels=occur_num, radius=1.2, textprops={'fontsize':12})
    ax12 = fig.add_subplot(gs[0, 1:])
    ax12.bar(height=count_bins, x=occur_num)
    ax12.set_xticks(occur_num)
    ax21 = fig.add_subplot(gs[1, 0])
    ax21.pie(x=count_bins_weighted, labels=occur_num, radius=1.2, textprops={'fontsize':12})
    ax22 = fig.add_subplot(gs[1, 1:])
    ax22.bar(height=count_bins_weighted, x=occur_num)
    ax22.set_xticks([i + 1 for i in range(sample_num)])
    y_lim = ax11.get_ylim()
    x_lim = ax11.get_xlim()
    ax11.text(s='Unique sequences', x=x_lim[0]*1.5, y=(y_lim[0] + y_lim[1])/2,
              ha='left', va='center', rotation=90, fontsize=14)
    y_lim = ax21.get_ylim()
    x_lim = ax21.get_xlim()
    ax21.text(s='Average Relative Abundance', x=x_lim[0]*1.5, y=(y_lim[0] + y_lim[1]) / 2,
              ha='left', va='center', rotation=90, fontsize=14)
    ax21.text(s='Percentage', x=(x_lim[0] + x_lim[1]) / 2, y=y_lim[0] - (y_lim[1] - y_lim[0]) * 0.1,
              ha='center', va='top', fontsize=14)
    y_lim = ax22.get_ylim()
    x_lim = ax22.get_xlim()
    ax22.text(s='Number of occurrence', x=(x_lim[0] + x_lim[1]) / 2, y=y_lim[0] - (y_lim[1] - y_lim[0]) * 0.12,
              ha='center', va='top', fontsize=14)
    plt.tight_layout()
    if fig_save_to is not None:
        fig.savefig(dirc=fig_save_to, dpi=300, bbox_inches='tight')
    plt.show()


def _get_groups(seq_table, group_by):
    """Return a dictionary of indicate groups of samples

    Args:
        seq_table:
        group_by:

    Returns:

    """
    groups = {}
    if isinstance(group_by, list):
        for ix, group in enumerate(group_by):
            if not isinstance(group, list):
                raise Exception('Error: if use list, group by should be a 2-D list of sample names')
            else:
                groups['Set_{}'.format(ix)] = group
    elif isinstance(group_by, dict):
        for group in group_by.values():
            if not isinstance(group, list):
                raise Exception('Error: if use dict, all values should be a 1-D list of sample names')
        groups = group_by
    elif isinstance(group_by, str):
        groups = {}
        for sample,content in seq_table.sample_info.items():
            if group_by not in content['metadata'].keys():
                raise Exception('Error: sample {} does not have attribute {}'.format(sample.name, group_by))
            else:
                if content['metadata'][group_by] in groups:
                    groups[content['metadata'][group_by]].append(sample)
                else:
                    groups[content['metadata'][group_by]] = [sample]

    return groups


def rep_variability_plot(seq_table, group_by, subsample_size=1000, var_method='MAD', percentage=True):
    import pandas as pd
    import numpy as np

    def get_sub_table(master_table, col_names, rep_num):
        return master_table[col_names][np.sum(master_table[col_names] > 0, axis=1) == rep_num]

    def get_variability(sub_table):
        if var_method.upper() == 'MAD':
            variability_list = abs(sub_table.subtract(sub_table.median(axis=1), axis='index')).median(axis=1)
            if percentage:
                variability_list = variability_list.divide(sub_table.median(axis=1), axis='index')
        elif var_method.upper() == 'SD':
            variability_list = sub_table.std(axis=1, ddof=1)
            if percentage:
                variability_list = variability_list.divide(sub_table.mean(axis=1), axis='index')
        else:
            raise Exception('Error: indicate var_method as MAD or SD')
        if len(variability_list) > subsample_size:
            variability_list = np.random.choice(variability_list.values, size=subsample_size)
        return variability_list

    results = {}
    groups = _get_groups(seq_table, group_by=group_by)

    for (group_name, sample_names) in groups.items():
        variability_list = {}
        sample_names = [sample for sample in sample_names
                        if (sample in seq_table.reacted_frac_table.columns)
                        or (sample in seq_table.count_table_input.columns)]
        if len(sample_names) > 1:
            for i in range(len(sample_names) - 1):
                rep_num = i + 2
                if sample_names[0] in seq_table.reacted_frac_table.columns:
                    variability_list[rep_num] = get_variability(sub_table=get_sub_table(
                            master_table=seq_table.reacted_frac_table,
                            col_names=sample_names,
                            rep_num=rep_num
                        ))
                elif sample_names[0] in seq_table.count_table_input.columns:
                    variability_list[rep_num] = get_variability(sub_table=get_sub_table(
                        master_table=seq_table.count_table_input/seq_table.count_table_input.sum(axis=0),
                        col_names=sample_names,
                        rep_num=rep_num
                    ))
            results[group_name] = variability_list
    # return results

    fig, axes = plt.subplots(1, len(results), figsize=[3 * len(results), 3], sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    for (ix, (group_name, variability_list)) in enumerate(results.items()):
        axes[ix].violinplot(variability_list.values(), positions=list(variability_list.keys()), showmedians=True)
        axes[ix].set_title(group_name, fontsize=14)
        # axes[ix].set_xlabel('Replicates', fontsize=14)
        axes[ix].set_xticks(list(variability_list.keys()))
        axes[ix].set_xticklabels(['{}\n({})'.format(rep_num, len(variabilityies))
                                  for rep_num,variabilityies in variability_list.items()])
        axes[0].set_ylabel('{}{}'.format('P' if percentage else '', var_method), fontsize=14)
    plt.show()



