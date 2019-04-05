"""
This module contains the methods used for k-seq dataset analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def survey_seq_occurrence(sequence_set, sample_range='reacted', display=True, fig_arg=None):
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
        plt.show()
        plt.tight_layout()
    return count_bins, count_bins_weighted

def get_replicates(sequence_set, key_domain):
    from itertools import groupby

    sample_type.sort(key=lambda x: x[1])
    groups = {}
    for key, group in groupby(sample_type, key=lambda x: x[1]):
        groups[key] = [x[0] for x in group]
    return groups


# def analyze_rep_variability(sequence_set, key_domain, subsample_size=100, statistic='MAD', percentage=True, display=True):
#
#     def select_seq(seq_counts, num_rep, subsample_size):
#
#
#     groups = get_replicates(sequence_set, key_domain)
#     for group_name, group_elems in groups.items:
#         rep_tot_num = len(group.elems)







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