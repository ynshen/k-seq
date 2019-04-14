"""
This module contains the methods used for k-seq dataset analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from . import pre_processing
from IPython.display import HTML

def sequencing_sample_overview(sample_set,
                               table=True,
                               figures=False,
                               fig_save_dirc=None):
    if table:
        table_to_display = pd.DataFrame()
        table_to_display['name'] = [sample.name for sample in sample_set]
        table_to_display['total counts'] = [sample.name for sample in sample_set]
        table_html = table_to_display.to_html()

        display(HTML(table_html))

    if figures:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatch
        import plot

        sampleNum = len(sampleSet)
        fig = plt.figure(figsize=[sampleNum * 0.5, 6])
        # Plot bar for total seqs
        ax = fig.add_subplot(111)
        ax.bar(x=[i - 0.2 for i in range(sampleNum)], height=[sample.totalSeq for sample in sampleSet], align='center',
               width=0.4, color='#2C73B4')
        # plot bar for unique seqs
        ax2 = ax.twinx()
        ax2.bar(x=[i + 0.2 for i in range(sampleNum)], height=[sample.uniqueSeq for sample in sampleSet],
                align='center', width=0.4, color='#FC820D')
        # plot scatter for spike-in percentage
        ax3 = ax.twinx()
        ax3.scatter([i for i in range(sampleNum)], [sample.stdCounts[0] / sample.totalSeq for sample in sampleSet],
                    color='#B2112A', marker='x')
        ax3.plot([-0.5, sampleNum - 0.5], [0.2, 0.2], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.4, 0.4], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.6, 0.6], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.8, 0.8], '#B2112A', ls='--', alpha=0.3)
        ax3.set_ylim([0, 1])
        ax3.set_yticks([])

        # Aesthetic adjustment
        ax.set_ylabel('Number of total reads in the sample', fontsize=14)
        ax2.set_ylabel('Number of unique sequences in the sample', fontsize=14)
        ax.set_xticks([i for i in range(sampleNum)])
        ax.set_xticklabels([sample.id[:sample.id.find('_counts')] for sample in sampleSet], rotation=90)
        plot.set_ticks_size(ax)
        plot.set_ticks_size(ax2)
        lgd = [mpatch.Patch(color='#2C73B4', label='Total Seqs'), mpatch.Patch(color='#FC820D', label='Unque Seqs'),
               plt.plot([], [], lw=0, marker='x', color='#B2112A', label='Percent of spike-in')[0]]
        plt.legend(handles=lgd)

        if figSaveDirc:
            fig.savefig(figSaveDirc, dpi=300)
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