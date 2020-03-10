"""
This module contains project level utility functions
"""
import matplotlib.pyplot as plt


class Presets:
    """Collection of preset colors/markers"""

    @staticmethod
    def _cycle_list(num, prop_list):
        """Generate a list of properties, cycle if num > len(prop_list)"""
        return [prop_list[i % len(prop_list)] for i in range(num)]

    @staticmethod
    def from_list(prop_list):
        from functools import partial
        return partial(Presets._cycle_list, prop_list=prop_list)

    @staticmethod
    def color_cat10(num=5):
        colors = [
            '#1F77B4',
            '#FF7F0E',
            '#2CA02C',
            '#D62728',
            '#9467BD',
            '#8C564B',
            '#E377C2',
            '#7F7F7F',
            '#BCBD22',
            '#17BECF'
        ]
        return Presets._cycle_list(num, colors)

    @staticmethod
    def color_tab10(num=5):
        colors = [
            '#4C78A8',
            '#F58518',
            '#E45756',
            '#72B7B2',
            '#54A24B',
            '#EECA3B',
            '#B279A2',
            '#FF9DA6',
            '#9D755D',
            '#BAB0AC'
        ]
        return Presets._cycle_list(num, colors)

    @staticmethod
    def color_pastel1(num=5):
        colors = [
            "#FBB5AE",
            "#B3CDE3",
            "#CCEBC5",
            "#DECBE4",
            "#FED9A6",
            "#FFFFCC",
            "#E5D8BD",
            "#FDDAEC",
            "#F2F2F2"
        ]
        return Presets._cycle_list(num, colors)

    @staticmethod
    def markers(num=5, with_line=False):
        from math import ceil

        full_marker_list = ['o', '^', 's', '+', 'x', 'D', 'v', '1', 'p', 'H']
        marker_list = []
        for i in range(ceil(num/10)):
            marker_list += full_marker_list
        if with_line:
            return ['-' + marker for marker in marker_list[:num]]
        else:
            return marker_list[:num]


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def blue_header(header):
    print(color.BOLD + color.BLUE + header + color.END)


def barplot(series, ax, label=None, yticklabels=None, barplot_kwargs=None):
    """General barplot for single series"""
    import numpy as np
    import pandas as pd

    pos = np.arange(len(series))
    if label is None and 'label' not in barplot_kwargs.keys() and isinstance(series, pd.Series):
        label = series.name
    if yticklabels is None and isinstance(series, pd.Series):
        yticklabels = series.index
    ax.barplot(pos, series, label=label, **barplot_kwargs)
    ax.set_xticks(pos)
    ax.set_xticklabels(pos, yticklabels, fontsize=12, rotation=90)


def pairplot(data, vars_name=None, vars_lim=None, vars_log=None, figsize=(2, 2), **kwargs):
    """Wrapper over seaborn.pairplot to visualize pairwise correlationw with log option"""
    import numpy as np
    import seaborn as sns

    if vars_name is None:
        vars_name = list(data.columns)
    else:
        data = data[vars_name]

    for var, var_log in zip(vars_name, vars_log):
        if var_log:
            data.loc[:, var] = data[var].apply(np.log10)
            data.rename(columns={var: "$\log_{10}$(%s)" % (var)}, inplace=True)

    return sns.pairplot(data=data, vars=data.columns,
                        markers='o', plot_kws=dict(s=5, edgecolor=None, alpha=0.3),
                        height=figsize[1], aspect=figsize[0] / figsize[1], **kwargs)


def savefig(save_fig_to, dpi=300, alpha=0):
    if save_fig_to is not None:
        fig = plt.gcf()
        fig.patch.set_alpha(alpha)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=dpi)


def ax_none(ax, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    return ax
