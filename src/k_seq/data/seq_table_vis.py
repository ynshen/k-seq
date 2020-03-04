
"""
todo:
  - convert spike-in vis to the spike-in normalizer?

"""

from .seq_data import SeqData
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def sample_unique_seqs_barplot(seq_table, black_list=None, ax=None, save_fig_to=None,
                               figsize=None, label_mapper=None, barplot_kwargs=None):
    """Barplot of unique seqs in each sample"""

    if not barplot_kwargs:
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
    if label_mapper:
        if callable(label_mapper):
            label_mapper = {sample: label_mapper(sample) for sample in uniq_counts.index}
    else:
        label_mapper = {sample: sample for sample in uniq_counts.index}

    ax.set_xticklabels(list([label_mapper[sample] for sample in uniq_counts.index]), fontsize=12, rotation=90)
    ax.set_ylabel('Unique seqs', fontsize=14)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}', ))
    ax.tick_params(axis='both', labelsize=12)

    if fig is not None and save_fig_to is not None:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return uniq_counts


def sample_total_counts_barplot(seq_table, black_list=None, ax=None, save_fig_to=None,
                                figsize=None, label_mapper=None, barplot_kwargs=None):
    """Barplot of total counts in each sample"""

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
    if label_mapper:
        if callable(label_mapper):
            label_mapper = {sample: label_mapper(sample) for sample in total_counts.index}
    else:
        label_mapper = {sample: sample for sample in total_counts.index}

    ax.set_xticklabels([label_mapper[sample] for sample in total_counts.index], fontsize=12, rotation=90)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}', ))
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ax.set_ylabel('Total counts', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    if fig is not None and save_fig_to is not None:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return total_counts


def sample_spike_in_ratio_scatterplot(seq_table, black_list=None, ax=None, save_fig_to=None,
                                      figsize=None, label_mapper=None, scatter_kwargs=None):
    """Scatter plot of spike in ratio in the pool"""
    import pandas as pd

    if scatter_kwargs is None:
        scatter_kwargs = {}
    #
    # if not isinstance(seq_table, SeqData):
    #     raise TypeError('seq_table needs to be a SeqData instance')
    samples = seq_table.sample_list
    if black_list is not None:
        samples = [sample for sample in samples if sample not in black_list]

    def get_spike_in_ratio(sample_id):
        spike_in_members = seq_table.spike_in.spike_in_members
        sample_series = seq_table.table[sample_id].sparse.to_dense()

        return sample_series[spike_in_members].sum() / sample_series.sum()

    spike_in_ratio = pd.Series(data=[get_spike_in_ratio(sample_id) for sample_id in samples], index=samples)

    if ax is None:
        if figsize is None:
            figsize = (len(spike_in_ratio)/2, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    pos = np.arange(len(spike_in_ratio))
    ax.scatter(pos, spike_in_ratio, marker='x', s=70, **scatter_kwargs)
    ax.set_xticks(pos)
    if label_mapper:
        if callable(label_mapper):
            label_mapper = {sample: label_mapper(sample) for sample in spike_in_ratio.index}
    else:
        label_mapper = {sample: sample for sample in spike_in_ratio.index}
    ax.set_xticklabels([label_mapper[sample] for sample in spike_in_ratio.index], fontsize=12, rotation=90)
    ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.3f}', ))
    ax.set_ylabel('Spike-in percent', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim([0, ax.get_ylim()[1]])
    yticks = [tick for tick in ax.get_yticks()][:-2]
    ax.set_yticks(yticks)
    # plt.setp(ax.get_yticklabels()[-1], visible=False)
    ax.tick_params(axis='both', labelsize=12)
    if save_fig_to:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)
    return spike_in_ratio


def sample_overview_plots(seq_table, plot_unique_seq=True, plot_total_counts=True, plot_spike_in_frac=True,
                          color_map=None, black_list=None, figsize=None, label_mapper=None, save_fig_to=None):
    """Overview plot(s) of unique seqs, total counts and spike-in fractions in the samples

    Args:

        seq_table (`SeqData`): sample set to survey

        plot_unique_seq (`bool`): plot bar plot for unique sequences if True

        plot_total_counts (`bool`): plot bar plot for total counts if True

        plot_spike_in_frac (`bool`): plot scatter plot for spike in fraction if True

        color_map (dict): {sample_name: color} for all plots

        black_list (list of `str`): list of sample name to exlude from the plots

        sep_plot (`bool`): plot separate plots for unique sequences, total counts and spike_in fractions if True

        label_mapper(dict or callable): alternative labels for samples

        fig_save_to (`str`): save figure to the directory if not None

    """

    plot_num = np.sum(np.array([plot_unique_seq, plot_total_counts, plot_spike_in_frac]))
    if black_list is None:
        black_list = []
    sample_num = len([sample for sample in seq_table.sample_list if sample not in black_list])

    if figsize is None:
        figsize = (sample_num * 0.5, 3 * plot_num)
    fig, axes = plt.subplots(plot_num, 1, figsize=figsize, sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0.01)
    axes_itr = (ax for ax in axes)
    if plot_unique_seq:
        ax = next(axes_itr)
        color = [color_map[sample] for sample in seq_table.sample_list] if color_map else '#2C73B4'
        sample_unique_seqs_barplot(seq_table=seq_table, ax=ax, label_mapper=label_mapper, barplot_kwargs={'color': color})
    if plot_total_counts:
        ax = next(axes_itr)
        color = [color_map[sample] for sample in seq_table.sample_list] if color_map else '#F39730'
        sample_total_counts_barplot(seq_table=seq_table, ax=ax, label_mapper=label_mapper, barplot_kwargs={'color': color})
    if plot_spike_in_frac:
        ax = next(axes_itr)
        color = [color_map[sample] for sample in seq_table.sample_list] if color_map else '#B2112A'
        sample_spike_in_ratio_scatterplot(seq_table=seq_table, ax=ax, label_mapper=label_mapper, scatter_kwargs={'color': color})
    fig.align_ylabels(axes)

    if save_fig_to:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, dpi=300, bbox_inches='tight')


def sample_rel_abun_hist(seq_table, black_list=None, bins=None, x_log=True, y_log=False,
                         ncol=None, nrow=None, figsize=None, hist_kwargs=None, save_fig_to=None):
    """todo: add pool counts composition curve for straight forward visualization"""

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


def sample_entropy_scatterplot(seq_table, black_list=None, normalize=False, base=2, color_map=None, figsize=None, scatter_kwargs=None, save_fig_to=None):
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

    if isinstance(seq_table, pd.DataFrame):
        sample_list = seq_table.columns
    else:
        sample_list = seq_table.sample_list
        seq_table = seq_table.table

    if black_list is not None:
        sample_list = [sample for sample in sample_list if sample not in black_list]

    entropy = seq_table[sample_list].apply(get_entropy, axis=0)

    if figsize is None:
        figsize = [len(entropy)/2, 4]
    if scatter_kwargs is None:
        scatter_kwargs = {}
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pos = np.arange(len(entropy))
    colors = [color_map[sample] for sample in sample_list] if color_map else '#2C73B4'
    ax.scatter(pos, entropy, marker='x', color=colors, **scatter_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(sample_list, fontsize=12, rotation=90)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('Entropy efficiency' if normalize else f'Entropy (base {base})', fontsize=14)
    ax.set_ylim([0, ax.get_ylim()[1]])
    if save_fig_to:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)


def cross_table_compare(base_table, compare_table, samples=None, ax=None, figsize=None, color_map=None,
                        save_fig_to=None):
    plt.style.use('seaborn')

    if samples is None:
        samples = set(base_table.columns) & set(compare_table.columns)
    if figsize is None:
        figsize = (len(samples), 6)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = '#2C73B4' if color_map is None else [color_map[sample] for sample in samples]
    ax.bar(np.arange(len(samples)),
           (compare_table[samples] > 0).sum(axis=0) / (base_table[samples] > 0).sum(axis=0),
           color=colors, width=0.6, label='Unique seq')
    colors = '#F39730' if color_map is None else [color_map[sample] for sample in samples]
    ax.scatter(np.arange(len(samples)),
               compare_table[samples].sum(axis=0) / base_table[samples].sum(axis=0),
               color=colors, marker='x', s=50, label='Total reads')
    ax.set_xticks(np.arange(len(samples)))
    ax.set_xticklabels(samples, fontsize=12, rotation=90)
    ax.set_ylabel('Pass ratio', fontsize=14)
    ax.tick_params('both', labelsize=12)

    if save_fig_to:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)


def rep_variance_scatter(table, grouper, xaxis=None, subsample=None,
                         xlog=True, ylog=True, xlim=None, ylim=None, group_title_pos=None,
                         xlabel=None, ylabel=None, label_map=None,
                         figsize=None, save_fig_to=None):

    table_gen = grouper.get_table(target=table, remove_zero=True)
    if figsize is None:
        figsize = (len(grouper.group) * 3, 3)

    fig, axes = plt.subplots(1, 5, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0.01)
    if xlabel is None:
        xlabel = 'Mean'
    if ylabel is None:
        ylabel = 'Standard Deviation'

    for ix, ((key, subtable), ax) in enumerate(zip(table_gen, axes)):
        if subsample is not None:
            subtable = subtable.sample(subsample, replace=False)

        if xaxis is not None:
            ax.scatter(xaxis[subtable.columns].loc[subtable.index].mean(axis=1), subtable.std(axis=1),
                       s=3, alpha=0.3, zorder=2)
        else:
            ax.scatter(subtable.mean(axis=1), subtable.std(axis=1), s=3, alpha=0.3, zorder=2)

        # plot lines
        line_xlim = ax.get_xlim() if xlim is None else xlim
        xs = np.logspace(np.log10(line_xlim[0]), np.log10(line_xlim[1]), 20)
        ax.plot(xs, xs, '#151515', alpha=0.5, ls='--', zorder=1)
        ax.plot(xs, xs * 0.1, '#151515', alpha=0.3, ls='--', zorder=1)
        ax.plot(xs, xs * 0.01, '#151515', alpha=0.1, ls='--', zorder=1)

        if group_title_pos is None:
            group_title_pos = (ax.get_xlim[0], ax.get_ylim[1])
        if label_map:
            if callable(label_map):
                label = label_map(key)
            else:
                label = label_map[key]
        else:
            label = f'{key:d} $\mu M$'
        ax.text(s=label, x=group_title_pos[0], y=group_title_pos[1], ha='left', va='top', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        if ix > 0:
            xticks = [tick for tick in ax.get_xticks()][2:-1]
            ax.set_xticks(xticks)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
        if xlim is not None:
            ax.set_xlim(xlim)

    fig.text(s=xlabel, x=0.5, y=0, ha='center', va='top', fontsize=12)
    if save_fig_to:
        fig.patch.set_alpha(0)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)
    plt.show()
