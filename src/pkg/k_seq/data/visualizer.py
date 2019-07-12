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

marker_list = ['o', 'x', '^', 's', '*', 'D', '+', 'v', '1', 'p']
color_list = ['#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080',
              '#F8DB36', '#AEAEAE', '#87554C', '#151515']


######################### Sequencing sample analysis ###############################
def count_file_info_table(sample_set, return_table=False):
    """Generate an overview info table for all samples in sample_set
    Print out HTML table for sequencing samples, including spike-in info if applicable

    Args:
        sample_set (`CountFileSet`): sample set to survey
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
                         black_list=None, sep_plot=False, save_dirc=None):
    """Generate overview plot(s) of unique seqs, total counts and spike-in fractions in the samples

    Args:
        sample_set (`CountFileSet`): sample set to survey
        save_dirc (str): save figure to the directory if not None

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

    if save_dirc:
        fig.savefig(save_dirc, dpi=300)
    plt.show()




def spike_in_peak_plot(sample_set, norm=True, max_dist=15):

    marker_list = ['-o', '->', '-+', '-s']  # different marker for different replicates
    color_list = ['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE']  # different color for different type of samples
    symbol_list = []
    for marker in marker_list:
        symbol_list += [marker for i in range(7)]

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=[24, 8])

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsNormed = counts/counts[0]
        ax[0].plot([i for i in range(maxDist + 1)], countsNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)

    # add binomial distribution guide line
    pList = [0.1, 0.01, 0.001, 0.0005, 0.0001]
    from scipy.stats import binom
    for p in pList:
        rv = binom(21, p)
        pmfs = np.array([rv.pmf(x) for x in range(7)])
        pmfsNormed = pmfs/pmfs[0]
        ax[0].plot([i for i in range(7)], pmfsNormed, color='k', ls = '--', alpha=0.3)
    ax[0].text(s='p=0.1', x=6, y=1e-1, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.01', x=6, y=1e-6, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.001', x=3.8, y=5e-7, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.0001', x=0.8, y=3e-7, ha='left', va='center', fontsize=12)

    ax[0].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-7, 5])
    ax[0].tick_params(labelsize=12)
    ax[0].set_ylabel('Sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsAccumulated = np.array([np.sum(counts[:i+1]) for i in range(maxDist + 1)])
        countsAccumulatedNormed = countsAccumulated/countsAccumulated[0]
        ax[1].plot([i for i in range(maxDist + 1)], countsAccumulatedNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)
    ax[1].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[1].tick_params(labelsize=12)
    ax[1].set_ylim([0.8, 1.5])
    ax[1].set_ylabel('Accumulated sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)
    ax[1].legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
    plt.tight_layout()
    # fig.savefig('/home/yuning/Work/rib


############### TODO


def spike_in_peak_plot(sample_set, norm=True, max_dist=15):

    marker_list = ['-o', '->', '-+', '-s']  # different marker for different replicates
    color_list = ['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE']  # different color for different type of samples
    symbol_list = []
    for marker in marker_list:
        symbol_list += [marker for i in range(7)]

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=[24, 8])

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsNormed = counts/counts[0]
        ax[0].plot([i for i in range(maxDist + 1)], countsNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)

    # add binomial distribution guide line
    pList = [0.1, 0.01, 0.001, 0.0005, 0.0001]
    from scipy.stats import binom
    for p in pList:
        rv = binom(21, p)
        pmfs = np.array([rv.pmf(x) for x in range(7)])
        pmfsNormed = pmfs/pmfs[0]
        ax[0].plot([i for i in range(7)], pmfsNormed, color='k', ls = '--', alpha=0.3)
    ax[0].text(s='p=0.1', x=6, y=1e-1, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.01', x=6, y=1e-6, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.001', x=3.8, y=5e-7, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.0001', x=0.8, y=3e-7, ha='left', va='center', fontsize=12)

    ax[0].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-7, 5])
    ax[0].tick_params(labelsize=12)
    ax[0].set_ylabel('Sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsAccumulated = np.array([np.sum(counts[:i+1]) for i in range(maxDist + 1)])
        countsAccumulatedNormed = countsAccumulated/countsAccumulated[0]
        ax[1].plot([i for i in range(maxDist + 1)], countsAccumulatedNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)
    ax[1].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[1].tick_params(labelsize=12)
    ax[1].set_ylim([0.8, 1.5])
    ax[1].set_ylabel('Accumulated sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)
    ax[1].legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
    plt.tight_layout()
    # fig.savefig('/home/yuning/Work/ribozyme_pred/fig/extStdErr.jpeg', dpi=300)
    plt.show()


















######################### Valid sequence analysis ###############################
def survey_seqs_info(sequence_set):
    sequence_set.seq_info = pd.DataFrame(index = sequence_set.count_table.index)
    input_samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'input']
    reacted_samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'reacted']
    sequence_set.seq_info['occur_in_inputs'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, input_samples] > 0, axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['occur_in_reacteds'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, reacted_samples] > 0, axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['total_counts_in_inputs'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, input_samples], axis=1),
        index=sequence_set.count_table.index
    )
    sequence_set.seq_info['total_counts_in_reacteds'] = pd.Series(
        np.sum(sequence_set.count_table.loc[:, reacted_samples], axis=1),
        index=sequence_set.count_table.index
    )
    return sequence_set

def survey_seq_occurrence(sequence_set, sample_range='reacted', display=True, save_dirc=None):
    if sample_range == 'reacted':
        samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'reacted']
        occurrence = sequence_set.seq_info['occur_in_reacteds'][1:]
        total_counts = sequence_set.seq_info['total_counts_in_reacteds'][1:]
    elif sample_range == 'inputs':
        samples = [sample[0] for sample in sequence_set.sample_info.items() if sample[1]['sample_type'] == 'input']
        occurrence = sequence_set.seq_info['occur_in_inputs'][1:]
        total_counts = sequence_set.seq_info['total_counts_in_inputs'][1:]
    else:
        samples = [sample[0] for sample in sequence_set.sample_info.items()]
        occurrence = sequence_set.seq_info['occur_in_inputs'][1:] + sequence_set.seq_info['occur_in_reacteds'][1:]
        total_counts = sequence_set.seq_info['total_counts_in_inputs'][1:] + sequence_set.seq_info['total_counts_in_reacteds'][1:]
    count_bins = np.bincount(occurrence, minlength=len(samples) + 1)[1:]
    count_bins_weighted = np.bincount(occurrence, minlength=len(samples) + 1, weights=total_counts)[1:]

    if display:
        fig = plt.figure(figsize=[16, 8])
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax11 = fig.add_subplot(gs[0, 0])
        ax11.pie(x=count_bins, labels=[i+1 for i in range(len(samples))], radius=1.2, textprops={'fontsize':12})
        ax12 = fig.add_subplot(gs[0, 1:])
        ax12.bar(height=count_bins, x=[i+1 for i in range(len(samples))])
        ax12.set_xticks([i+1 for i in range(len(samples))])
        ax21 = fig.add_subplot(gs[1, 0])
        ax21.pie(x=count_bins_weighted, labels=[i+1 for i in range(len(samples))], radius=1.2, textprops={'fontsize':12})
        ax22 = fig.add_subplot(gs[1, 1:])
        ax22.bar(height=count_bins_weighted, x=[i+1 for i in range(len(samples))])
        ax22.set_xticks([i + 1 for i in range(len(samples))])
        y_lim = ax11.get_ylim()
        x_lim = ax11.get_xlim()
        ax11.text(s='Unique sequences', x=x_lim[0]*1.5, y=(y_lim[0] + y_lim[1])/2, ha='left', va='center', rotation=90, fontsize=14)
        y_lim = ax21.get_ylim()
        x_lim = ax21.get_xlim()
        ax21.text(s='Total counts', x=x_lim[0]*1.5, y=(y_lim[0] + y_lim[1]) / 2, ha='left', va='center', rotation=90, fontsize=14)
        ax21.text(s='Percentage', x=(x_lim[0] + x_lim[1]) / 2, y=y_lim[0] - (y_lim[1] - y_lim[0]) * 0.1,
                  ha='center', va='top', fontsize=14)
        y_lim = ax22.get_ylim()
        x_lim = ax22.get_xlim()
        ax22.text(s='Number of occurrence', x=(x_lim[0] + x_lim[1]) / 2, y=y_lim[0] - (y_lim[1] - y_lim[0]) * 0.12,
                  ha='center', va='top', fontsize=14)
        plt.tight_layout()
        if save_dirc is not None:
            fig.savefig(dirc=save_dirc, dpi=300, bbox_inches='tight')
        plt.show()

    return count_bins, count_bins_weighted


def get_replicates(sequence_set, key_domain):
    from itertools import groupby

    sample_type = [(sample[0], sample[1]['metadata'][key_domain]) for sample in sequence_set.sample_info.items()]
    sample_type.sort(key=lambda x: x[1])
    groups = {}
    for key, group in groupby(sample_type, key=lambda x: x[1]):
        groups[key] = [x[0] for x in group]
    return groups


def analyze_rep_variability(sequence_set, key_domain, subsample_size=1000, variability='MAD', percentage=True, display=True):
    np.random.seed(23)

    def get_variability(seq_subset, num_rep):
        seq_subset_subset = seq_subset[np.sum(~seq_subset.isnull(), axis=1) == num_rep]
        if variability == 'MAD':
            variability_list = abs(seq_subset_subset.subtract(seq_subset_subset.median(axis=1), axis='index')).median(axis=1)
            if percentage:
                variability_list = variability_list.divide(seq_subset_subset.median(axis=1), axis='index')
        elif variability == 'SD':
            variability_list = seq_subset_subset.std(axis=1, ddof=1)
            if percentage:
                variability_list = variability_list.divide(seq_subset_subset.mean(axis=1), axis='index')
        if len(variability_list) > subsample_size:
            variability_list = np.random.choice(variability_list, size=subsample_size)

        return variability_list

    variability_res = {}
    groups = get_replicates(sequence_set, key_domain)
    for (group_name, group_elems) in groups.items():
        variability_list = []
        for i in range(len(group_elems) - 1):
            num_rep = i + 2
            variability_list.append(
                get_variability(seq_subset=sequence_set.reacted_frac_table.loc[:,group_elems], num_rep=num_rep)
            )
        variability_res[group_name] = variability_list

    if display:
        fig, axes = plt.subplots(1, len(groups), figsize=[3*len(groups), 3], sharey=True)
        plt.subplots_adjust(hspace=0, wspace=0)
        for (ix, (group_name, variability_list)) in enumerate(variability_res.items()):
            axes[ix].violinplot(variability_list, positions=[i + 2 for i in range(len(variability_list))], showmedians=True)
            axes[ix].set_title(group_name, fontsize=14)
            # axes[ix].set_xlabel('Replicates', fontsize=14)
            axes[ix].set_xticks([i + 2 for i in range(len(variability_list))])
            axes[ix].set_xticklabels(['{}\n({})'.format(i + 2, len(variability_list[i])) for i in range(len(variability_list))])
        axes[0].set_ylabel('{}{}'.format('P' if percentage else '', variability), fontsize=14)
        plt.show()
    return variability_res



