
"""
todo:
  - convert spike-in vis to the spike-in normalizer?

"""

from .seq_table import SeqTable
import matplotlib.pyplot as plt
import matplotlib as mpl


def sample_unique_seqs_barplot(seq_table, black_list=None, ax=None, save_fig_to=None, figsize=None, barplot_kwargs=None):
    """Barplot of unique seqs in each sample"""
    import numpy as np

    if barplot_kwargs is None:
        barplot_kwargs = {}
    if hasattr(seq_table, 'table'):
        seq_table = seq_table.table
    if black_list is not None:
        seq_table = seq_table[~seq_table.columns.isin(black_list)]
    uniq_counts = (seq_table > 0).sum(0)
    if ax is None:
        if figsize is None:
            figsize = (len(uniq_counts)/2, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    pos = np.arange(len(uniq_counts))
    ax.bar(pos, uniq_counts, **barplot_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(list(uniq_counts.index), fontsize=12, rotation=90)
    ax.set_ylabel('Number of unique seqs', fontsize=12)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}', ))
    ax.tick_params(axis='both', labelsize=12)

    if fig is not None and save_fig_to is not None:
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return uniq_counts


def sample_total_counts_barplot(seq_table, black_list=None, ax=None, save_fig_to=None, figsize=None, barplot_kwargs=None):
    """Barplot of total counts in each sample"""
    import numpy as np

    if barplot_kwargs is None:
        barplot_kwargs = {}
    if hasattr(seq_table, 'table'):
        seq_table = seq_table.table
    if black_list is not None:
        seq_table = seq_table[~seq_table.columns.isin(black_list)]
    total_counts = seq_table.sum(0)
    if ax is None:
        if figsize is None:
            figsize = (len(total_counts)/2, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    pos = np.arange(len(total_counts))
    ax.bar(pos, total_counts, **barplot_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(list(total_counts.index), fontsize=12, rotation=90)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}', ))
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ax.set_ylabel('Number of total counts', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    if fig is not None and save_fig_to is not None:
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return total_counts


def sample_spike_in_ratio_scatterplot(seq_table, black_list=None, ax=None, save_fig_to=None, figsize=None,
                                      scatter_kwargs=None):
    """Scatter plot of spike in ratio in the pool"""
    import numpy as np
    import pandas as pd

    if scatter_kwargs is None:
        scatter_kwargs = {}
    #
    # if not isinstance(seq_table, SeqTable):
    #     raise TypeError('seq_table needs to be a SeqTable instance')
    samples = seq_table.sample_list
    if black_list is not None:
        samples = [sample for sample in samples if sample not in black_list]

    def get_spike_in_ratio(sample_id):
        sample_meta = seq_table.metadata.samples[sample_id]
        return 1 - (sample_meta['total_counts_no_spike_in'] / sample_meta['total_counts'])

    spike_in_ratio = pd.Series(data=[get_spike_in_ratio(sample_id) for sample_id in samples], index=samples)

    if ax is None:
        if figsize is None:
            figsize = (len(spike_in_ratio)/2, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    pos = np.arange(len(spike_in_ratio))
    ax.scatter(pos, spike_in_ratio, marker='*', **scatter_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(list(spike_in_ratio.index), fontsize=12, rotation=90)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}', ))
    ax.set_ylabel('Spike-in percent', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim([0, ax.get_ylim()[1]])
    yticks = [tick for tick in ax.get_yticks()][:-2]
    # print(yticks)
    ax.set_yticks(yticks)
    # plt.setp(ax.get_yticklabels()[-1], visible=False)
    ax.tick_params(axis='both', labelsize=12)
    if save_fig_to:
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)
    return spike_in_ratio


def sample_overview_plots(seq_table, plot_unique_seq=True, plot_total_counts=True, plot_spike_in_frac=True,
                          black_list=None, save_fig_to=None):
    """Overview plot(s) of unique seqs, total counts and spike-in fractions in the samples

    Args:

        seq_table (`SeqTable`): sample set to survey

        plot_unique_seq (`bool`): plot bar plot for unique sequences if True

        plot_total_counts (`bool`): plot bar plot for total counts if True

        plot_spike_in_frac (`bool`): plot scatter plot for spike in fraction if True

        black_list (list of `str`): list of sample name to exlude from the plots

        sep_plot (`bool`): plot separate plots for unique sequences, total counts and spike_in fractions if True

        fig_save_to (`str`): save figure to the directory if not None

    """
    import numpy as np

    plot_num = np.sum(np.array([plot_unique_seq, plot_total_counts, plot_spike_in_frac]))
    if black_list is None:
        black_list = []
    sample_num = len([sample for sample in seq_table.sample_list if sample not in black_list])

    fig, axes = plt.subplots(plot_num, 1, figsize=[sample_num * 0.5, 3 * plot_num], sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0.01)
    axes_itr = (ax for ax in axes)
    if plot_unique_seq:
        ax = next(axes_itr)
        sample_unique_seqs_barplot(seq_table=seq_table, ax=ax, barplot_kwargs={'color': '#2C73B4'})
    if plot_total_counts:
        ax = next(axes_itr)
        sample_total_counts_barplot(seq_table=seq_table, ax=ax, barplot_kwargs={'color': '#F39730'})
    if plot_spike_in_frac:
        ax = next(axes_itr)
        sample_spike_in_ratio_scatterplot(seq_table=seq_table, ax=ax, scatter_kwargs={'color': '#B2112A'})
    fig.align_ylabels(axes)

    if save_fig_to:
        fig.savefig(save_fig_to, dpi=300)


def sample_rel_abun_hist(seq_table, black_list=None, bins=None, x_log=True, y_log=False,
                         ncol=None, nrow=None, figsize=None, hist_kwargs=None, save_fig_to=None):
    """todo: add pool counts composition curve for straight forward visualization"""
    import numpy as np

    if hist_kwargs is None:
        hist_kwargs = {}
    sample_list = seq_table.sample_list
    if black_list is not None:
        sample_list = [sample for sample in sample_list if sample not in black_list]
    rel_abun = seq_table.table[sample_list] / seq_table.table[sample_list].sum(axis=0)
    if ncol is None:
        ncol = 3
    if nrow is None:
        nrow = int((len(sample_list) - 1) / ncol) + 1
    if figsize is None:
        figsize = [3 * ncol, 2 * nrow]
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.reshape(axes, -1)
    axes_ir = (ax for ax in axes)

    if bins is None:
        if x_log:
            min_dig = int(np.log10(rel_abun[rel_abun > 0].min().min())) - 1
            bins = np.logspace(min_dig, 0, -min_dig * 2)
        else:
            min_dig = rel_abun[rel_abun > 0].min().min()
            bins = np.linspace(min_dig, 1, 20)

    for sample in sample_list:
        ax = next(axes_ir)
        ax.hist(rel_abun[sample], bins=bins, **hist_kwargs)
        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(sample, fontsize=12)

    for ax in axes_ir:
        ax.axis('off')
    fig.text(s='Relative abundance', x=0.5, y=0, ha='center', va='top', fontsize=14)
    plt.tight_layout()


def sample_entropy_scatterplot(seq_table, black_list=None, normalize=False, base=2, figsize=None, scatter_kwargs=None):
    import numpy as np
    import pandas as pd

    def get_entropy(smpl_col):
        valid = smpl_col[smpl_col > 0]
        if pd.api.types.is_sparse(valid):
            valid = valid.sparse.to_dense()
        valid = valid / np.sum(valid)
        if normalize:
            return -np.sum(valid * np.log(valid)) / np.log(len(valid))
        else:
            return -np.sum(valid * np.log(valid)) / np.log(base)

    sample_list = seq_table.sample_list
    if black_list is not None:
        sample_list = [sample for sample in sample_list if sample not in black_list]

    entropy = seq_table.table[sample_list].apply(get_entropy, axis=0)

    if figsize is None:
        figsize = [len(entropy)/2, 4]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pos = np.arange(len(entropy))
    ax.scatter(pos, entropy, marker='x', **scatter_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(sample_list, fontsize=12, rotation=90)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Entropy efficiency' if normalize else f'Entropy (base {base})', fontsize=14)
    ax.set_ylim([0, ax.get_ylim()[1]])
