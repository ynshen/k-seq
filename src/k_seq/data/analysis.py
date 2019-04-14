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


def sequencing_sample_info_table(sample_set):
    """
    Print out HTML table of basic infos for sequencing samples, including spike-in info if applicable
    return a pd.DataFrame including same info

    :param sample_set: list of SequencingSample objects
    :return: table: pd.DataFrame
    """

    table = pd.DataFrame()
    table['sample type'] = [sample.sample_type for sample in sample_set]
    table['name'] = [sample.name for sample in sample_set]
    table['total counts'] = [sample.total_counts for sample in sample_set]
    table['unique sequences'] = [sample.unique_seqs for sample in sample_set]
    if 'spike_in' in sample_set[0].__dict__.keys():
        table['spike-in amount'] = [sample.spike_in_amount for sample in sample_set]
        table['spike-in counts (dist={})'.format(sample_set[0].quant_factor_max_dist)] = \
            [np.sum(sample.spike_in_counts[0:sample.quant_factor_max_dist + 1]) for sample in sample_set]
        table['spike-in percent'] = [
            np.sum(sample.spike_in_counts[0:sample.quant_factor_max_dist + 1])/sample.total_counts
            for sample in sample_set
        ]
        table['quantification factor'] = [sample.quant_factor for sample in sample_set]
    table_html = table.to_html(
        formatters={
            'total counts': lambda x: '{:,}'.format(x),
            'unique sequences': lambda x: '{:,}'.format(x),
            'spike-in counts (dist{})'.format(sample_set[0].quant_factor_max_dist): lambda x: '{:,}'.format(x),
            'spike-in percent': lambda x: '{:.3f}'.format(x),
            'quantification factor': lambda x:'{:.3e}'.format(x)
        }
    )
    display(HTML(table_html))
    return table


def sequencing_sample_info_plot(sample_set, save_dirc=None):
    """
    An overview plot of unique sequences, total counts and spike-in ratios
    :param sample_set: list of SequencingSample objects
    :param save_dirc: string, directory to save the plot
    :return:
    """

    import plot
    sample_num = len(sample_set)
    fig = plt.figure(figsize=[sample_num * 0.5, 6])
    # Plot bar for total seqs
    ax = fig.add_subplot(111)
    ax.bar(x=[i - 0.2 for i in range(sample_num)], height=[sample.total_counts for sample in sample_set],
           align='center',width=0.4, color='#2C73B4')
    # plot bar for unique seqs
    ax2 = ax.twinx()
    ax2.bar(x=[i + 0.2 for i in range(sample_num)], height=[sample.unique_seqs for sample in sample_set],
            align='center', width=0.4, color='#FC820D')
    # plot scatter for spike-in percentage
    ax3 = ax.twinx()
    ax3.scatter([i for i in range(sample_num)],
                [np.sum(sample.spike_in_counts[0:sample.quant_factor_max_dist + 1])/sample.total_counts
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
    # Aesthetic adjustment
    ax.set_ylabel('Number of total reads in the sample', fontsize=14)
    ax2.set_ylabel('Number of unique sequences in the sample', fontsize=14)
    ax.set_xticks([i for i in range(sample_num)])
    ax.set_xticklabels([sample.name for sample in sample_set], rotation=90)
    plot.set_ticks_size(ax)
    plot.set_ticks_size(ax2)
    lgd = [mpatch.Patch(color='#2C73B4', label='Total counts'),
           mpatch.Patch(color='#FC820D', label='Unique sequences'),
           plt.plot([], [], lw=0, marker='x', color='#B2112A', label='Percent of spike-in')[0]]
    plt.legend(handles=lgd)
    if save_dirc:
        fig.savefig(save_dirc, dpi=300)
    plt.show()


def plot_std_peak_dist(sampleSet, norm=True, maxDist=15):

    markerList = ['-o', '->', '-+', '-s']  # different marker for different replicates
    colorList = ['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE']  # different color for different type of samples
    symbolList = []
    for marker in markerList:
        symbolList += [marker for i in range(7)]

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




































def survey_seq_occurrence(sequence_set, sample_range='reacted', display=True, fig_save_dirc=None, fig_arg=None):
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
        if fig_save_dirc is not None:
            fig.savefig(dirc=fig_save_dirc, dpi=300, bbox_inches='tight')
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



def fitting_check(k, A, xTrue, y, size=100, average=True):
    np.random.seed(23)

    fittingRes = {
        'y_': None,
        'x_': None,
        'k': [],
        'kerr': [],
        'A': [],
        'Aerr': [],
        'kA': [],
        'kAerr': [],
        'mse': [],
        'mseTrue': [],
        'r2': []
    }

    if average:
        y_ = np.mean(y, axis=0)
        x_ = np.mean(xTrue, axis=0)
    else:
        y_ = np.reshape(y, y.shape[0] * y.shape[1])
        x_ = np.reshape(xTrue, xTrue.shape[0] * xTrue.shape[1])

    for epochs in range(size):
        # initGuess= (np.random.random(), np.random.random()*k*100)
        initGuess = (np.random.random(), np.random.random())

        try:
            popt, pcov = curve_fit(func, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]

        if fittingRes['y_'] is None:
            fittingRes['y_'] = y_
        if fittingRes['x_'] is None:
            fittingRes['x_'] = x_
        fittingRes['k'].append(popt[1])
        fittingRes['kerr'].append((popt[1] - k) / k)
        fittingRes['A'].append(popt[0])
        fittingRes['Aerr'].append((popt[0] - A) / A)
        fittingRes['kA'].append(popt[0] * popt[1])
        fittingRes['kAerr'].append((popt[0] * popt[1] - k * A) / (k * A))

        fittingRes['mse'].append(mse(x_, y_, A=popt[0], k=popt[1]))
        fittingRes['mseTrue'].append(mse(x_, y_, A=A, k=k))

        res = y_ - (1 - np.exp(-0.479 * 90 * popt[1] * x_)) * popt[0]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
        fittingRes['r2'].append(1 - ss_res / ss_tot)

    return fittingRes