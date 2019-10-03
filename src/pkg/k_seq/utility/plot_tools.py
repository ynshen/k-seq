"""
This module contains project level utility functions
"""


class PlotPreset:

    def __init__(self):
        pass

    @staticmethod
    def colors(num=5):
        from math import ceil

        full_color_list = ['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE']
        color_list = []
        for i in range(ceil(num/7)):
            color_list += full_color_list
        return color_list[:num]

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

