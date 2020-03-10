
"""
todo:
  - convert spike-in vis to the spike-in normalizer?

"""

from .seq_data import SeqData, SeqTable
from ..utility.func_tools import FuncToMethod, update_none
from ..utility.plot_tools import savefig, ax_none
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


class SeqTableAnalyzer(FuncToMethod):

    def __init__(self, seq_table):
        super().__init__(
            functions=[
                seq_overview, 
                sample_overview,
                sample_unique_seqs_barplot,
                sample_total_reads_barplot,
                seq_mean_count_detected_samples_scatterplot,
                seq_length_dist
            ],
            seq_table=seq_table
        )

        self.seq_table = seq_table


def seq_overview(seq_table, axis=0):
    """Summarize sample in seq_table, with info of seq length, sample detected, mean, sd
    Returns:
        A `pd.DataFrame` show the summary for sequences
    """
    if axis == 1:
        seq_table = seq_table.transpose()

    return pd.DataFrame.from_dict(
        {'length': seq_table.index.to_series().apply(len),
         'samples detected': (seq_table > 0).sum(axis=1),
         'mean': seq_table.mean(axis=1),
         'sd': seq_table.std(axis=1)},
        orient='columns'
    )


def seq_length_dist(seq_table, axis=0, ax=None, figsize=(6, 3), bins=20, logx=False, logy=False,
                    hist_kwargs=None, save_fig_to=None):
    """Distribution of unique sequences in their length"""

    seqs = seq_table.index.to_series() if axis == 0 else seq_table.columns.to_series()
    seqs_length = seqs.apply(len)

    hist_kwargs = update_none(hist_kwargs, {})
    ax = ax_none(ax, figsize)
    if isinstance(bins, int):
        if logx:
            bins = np.logspace(np.log10(np.min(seqs_length)) - 0.1, np.log10(np.max(seqs_length)) + 0.1, bins + 1)
        else:
            bins = np.linspace(np.min(seqs_length) - 0.1, np.max(seqs_length) + 0.1, bins + 1)

    ax.hist(seqs_length, bins=bins, **hist_kwargs)
    ax.set_xlabel('Sequence length', fontsize=12)
    ax.set_ylabel('Unique sequences', fontsize=12)

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    savefig(save_fig_to)


def sample_overview(seq_table, axis=1):
    """Summarize sequences for a given seq_table, with info of unique seqs, total amount

    Returns:
        A `pd.DataFrame` show the summary for sequences
    """

    if axis == 0:
        seq_table = seq_table.transpose()

    if isinstance(seq_table, SeqTable):
        col_name = f'total amount ({seq_table.unit})'
    else:
        col_name = 'total amount'

    return pd.DataFrame.from_dict(
        {'unique seqs': (seq_table > 0).sum(axis=0),
         col_name: seq_table.sum(axis=0)},
        orient='columns'
    )


def sample_unique_seqs_barplot(seq_table, black_list=None, logy=False,
                               ax=None, save_fig_to=None, figsize=None,
                               x_label=None, y_label='Unique sequences', fontsize=14,
                               label_mapper=None, barplot_kwargs=None):
    """Barplot of unique seqs in each sample"""

    if not barplot_kwargs:
        barplot_kwargs = {}
    if black_list is not None:
        seq_table = seq_table[~seq_table.columns.isin(black_list)]
    uniq_seqs = (seq_table > 0).sum(0)
    if label_mapper:
        uniq_seqs = uniq_seqs.rename(label_mapper)

    barplot_kwargs['ax'] = ax
    barplot_kwargs['figsize'] = figsize
    barplot_kwargs['logy'] = logy

    uniq_seqs.plot(kind='bar', **barplot_kwargs)

    if ax is None:
        ax = plt.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    savefig(save_fig_to)

    return uniq_seqs


def sample_total_reads_barplot(seq_table, black_list=None, logy=False,
                                ax=None, save_fig_to=None, figsize=None,
                                x_label=None, y_label='Total reads', fontsize=14,
                                label_mapper=None, barplot_kwargs=None):
    """Barplot of total counts in each sample"""

    if not barplot_kwargs:
        barplot_kwargs = {}
    if black_list is not None:
        seq_table = seq_table[~seq_table.columns.isin(black_list)]
    total_reads = seq_table.sum(0)
    if label_mapper:
        total_reads = total_reads.rename(label_mapper)

    barplot_kwargs['ax'] = ax
    barplot_kwargs['figsize'] = figsize
    barplot_kwargs['logy'] = logy

    total_reads.plot(kind='bar', **barplot_kwargs)

    if ax is None:
        ax = plt.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    savefig(save_fig_to)

    return total_reads


def seq_mean_count_detected_samples_scatterplot(seq_table, figsize=5,
                                                log_counts=True, subsample=None,
                                                color='#1F77B4',
                                                marker_size=5, scatter_kwargs=None):
    import seaborn as sns
    data = pd.DataFrame({'Mean counts': seq_table.mean(axis=1),
                         'Detected samples': (seq_table > 0).sum(axis=1)})

    if log_counts:
        y = '$\log_{10}$(Mean counts)'
        data_to_plot = pd.DataFrame({y: np.log10(data['Mean counts']),
                                     'Detected samples': data['Detected samples']})
    else:
        data_to_plot = data
        y = 'Mean counts'
    scatter_kwargs = update_none(scatter_kwargs, {})
    if subsample:
        data_to_plot = data_to_plot.sample(subsample)
    sns.jointplot(data=data_to_plot,
                  x='Detected samples', y=y, kind='scatter', height=figsize, ratio=3, space=0.1, color=color,
                  joint_kws={'s': marker_size}.update(scatter_kwargs))

    return data


class SeqDataAnalyzer(FuncToMethod):

    def __init__(self, seq_data):
        super().__init__(
            functions=[
                sample_info
            ],
            seq_data=seq_data
        )
        self.seq_data = seq_data


def sample_info(seq_data):
    """Summarize sample info for a SeqData, with info of total amount and spike-in
    Returns:
        A `pd.DataFrame` show the summary for samples
    """
    info = pd.DataFrame(index=seq_data.samples)
    if hasattr(seq_data, 'grouper') and hasattr(seq_data.grouper, 'input'):
        def get_sample_type(sample):
            if sample in seq_data.grouper.input.group:
                return 'input'
            elif sample in seq_data.grouper.reacted.group:
                return 'reacted'
            else:
                return np.nan

        info['sample type'] = info.index.to_series().apply(get_sample_type)

    if seq_data.x_values is not None:
        info['x values'] = seq_data.x_values

    info = pd.concat([info, seq_data.table.original.analysis.sample_overview()], axis=1)
    info = info.rename(columns={'total amount': 'total reads'})

    if hasattr(seq_data, 'spike_in'):
        info[f'total amount (spike-in, {seq_data.spike_in.unit})'] = seq_data.spike_in.norm_factor * seq_data.spike_in.base_table.sum(axis=0)
        info['spike-in fraction'] = seq_data.spike_in.peak.peak_abun(use_relative=False)[0].sum(
            axis=0) / seq_data.spike_in.base_table.sum(axis=0)

    if hasattr(seq_data, 'sample_total'):
        info[f'total amount (sample total, {seq_data.sample_total.unit})'] = pd.Series(seq_data.sample_total.total_amounts)

    return info


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
