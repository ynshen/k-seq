"""
This module contains project level utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def savefig(save_fig_to, dpi=300, alpha=0):
    if save_fig_to is not None:
        fig = plt.gcf()
        fig.patch.set_alpha(alpha)
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=dpi)


def ax_none(ax, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    return ax


def regplot(x, y, ax=None, xlabel=None, ylabel=None, digit=4, equation_loc='best',
            xlog=False, ylog=False,
            kwargs_scatter=None, kwargs_line=None):

    ax = ax_none(ax, figsize=(6, 4))
    if kwargs_scatter is None: kwargs_scatter = {}
    if kwargs_line is None: kwargs_line = {}

    if xlabel is None:
        try:
            xlabel = x.name
        except ArithmeticError:
            pass

    if ylabel is None:
        try:
            ylabel = y.name
        except AttributeError:
            pass

    x = np.array(x)
    y = np.array(y)

    ax.scatter(x, y, **kwargs_scatter)

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(x) if xlog else x,
                                                                   np.log(y) if ylog else y)

    predicted = np.log(x) * slope + intercept if xlog else x * slope + intercept
    if ylog:
        predicted = np.exp(predicted)

    eq_formatter = f"{'ln(y)' if ylog else 'y'} = {{:.{digit}f}} {'ln(x)' if xlog else 'x'} + {{:.{digit}f}}\n" \
                   f"R-squared={{:.{digit}f}}"

    ax.plot(x, predicted, label=eq_formatter.format(slope, intercept, r_value ** 2), **kwargs_line)
    if equation_loc is not None:
        ax.legend(loc=equation_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale('log')
        ax.set_xlim(min(x) / 2, max(x) * 2)
    if ylog:
        ax.set_yscale('log')
        ax.set_ylim(min(y) / 2, max(y) * 2)
    return ax


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


def plot_curve(model, x=None, y=None, param=None, major_param=None, subsample=20,
               x_label=None, y_label=None, x_lim=None, y_lim=None,
               major_curve_kwargs=None, curve_kwargs=None, datapoint_kwargs=None,
               major_curve_label='major curve', curve_label='curves', datapoint_label='data',
               legend=False, legend_loc='upper right', fontsize=12, ax=None,
               x_tick_formatter=None, y_tick_formatter=None, **kwargs):
    """Plot fitting results with
        i) data points (scatter), ii) major curve, iii) a set of curves (e.g. from bootstrap or convergence)

    Args:
        model (callable): kinetic model returns y with first argument as x
        x (list-like): x values of data
        y (list-like): y values of corresponding x data, same length
        param (dict or pd.DataFrame): estimated parameters for model from fitting(s)
        major_param (dict): a major parameter estimated
        subsample (int): maximal num of fitting curves to show
        x_label (str): x axis label name
        y_label (str): y axis label name
        x_lim (2-tuple): lower and upper limit of x axis
        y_lim (2-tuple): lower and upper limit of y axis
        major_curve_kwargs (dict): plot arguments for major curve
        curve_kwargs (dict): plot arguments for other curves
        datapoint_kwargs (dict): scatter plot arguments for data points
        major_curve_label (str): label on legend for major curve
        curve_label (str): label on legend for other curves
        datapoint_label (str): label on legend for data points
        legend (bool): if show legend
        legend_loc (str or 4-tuple): specify the location of legend, default is upper right
        ax (plt.Axes): Axes to plot on
    """
    from yutility import logging
    from .func_tools import update_none

    ax = ax_none(ax, figsize=(4, 3))
    major_curve_kwargs = update_none(major_curve_kwargs, {})
    curve_kwargs = update_none(curve_kwargs, {})
    datapoint_kwargs = update_none(datapoint_kwargs, {})

    # prep plotting kwargs
    major_curve_kwargs = {**{'color': '#4C78A8', 'alpha': 0.5}, **major_curve_kwargs}
    if major_param is None:
        curve_kwargs = {**{'color': '#4C78A8', 'alpha': 0.5}, **curve_kwargs}
    else:
        curve_kwargs = {**{'color': '#AEAEAE', 'alpha': 0.3}, **curve_kwargs}
    datapoint_kwargs = {**{'color': '#E45756', 'alpha': 1, 'zorder': 3, 'marker': 'x'}, **datapoint_kwargs}

    # plot curves
    if x_lim:
        xs = np.linspace(0, x_lim[1] * 0.9, 100)
    else:
        xs = np.linspace(0, np.max(x) * 1.1, 100)

    def add_curve(data, plot_args):
        if isinstance(data, dict):
            y_ = model(xs, **data)
        elif isinstance(data, pd.Series):
            y_ = model(xs, **data.to_dict())
        else:
            logging.error('Unknown parameter input type, should be pd.Series or dict', error_type=TypeError)

        ax.plot(xs, y_, marker=None, **plot_args)

    if param is not None:
        if isinstance(param, dict):
            # single curve
            add_curve(param)
        elif isinstance(param, pd.DataFrame):
            param = param[~np.any(param.isna(), axis=1)]
            if param.shape[0] > subsample:
                param = param.sample(subsample, replace=False)
        param.apply(add_curve, axis=1, plot_args=curve_kwargs)

    # add major curve if applicable
    if major_param is not None:
        add_curve(major_param, plot_args=major_curve_kwargs)

    # add raw data points
    if x is not None and y is not None:
        ax.scatter(x, y, label=datapoint_label, **datapoint_kwargs)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    if legend:
        if major_param is not None:
            ax.plot([0], [0], label=major_curve_label, **major_curve_kwargs)
        if param is not None:
            ax.plot([0], [0], label=curve_label, **curve_kwargs)
        ax.legend(loc=legend_loc, frameon=True, edgecolor=None, framealpha=0.5)
    return ax


def value_to_loc(value, range, resolution, log):
    """Convert actual value to location on heatmap"""

    vmin = range[0]
    vmax = range[1]

    if log:
        return np.log10(value / vmin) / np.log10(vmax / vmin) * resolution
    else:
        return (value - vmin) / (vmax - vmin) * resolution


def format_ticks(ax, axis, tick_num=3, log=False, int_tick_only=False, tick_formatter=None):
    """Manual formatting for figure ticks"""

    if axis == 'x':
        get_lim = ax.get_xlim
        set_ticks = ax.set_xticks
        set_ticklabels = ax.set_xticklabels
    else:
        get_lim = ax.get_ylim
        set_ticks = ax.set_yticks
        set_ticklabels = ax.set_yticklabels

    # set ticks
    lim = get_lim()
    if log:
        tick_ix = np.logspace(np.log(lim[0]), np.log(lim[1]), tick_num, dtype=int if int_tick_only else float)
    else:
        tick_ix = np.linspace(lim[0], lim[1], tick_num, dtype=int if int_tick_only else float)

    set_ticks(tick_ix)

    # set tick labels
    if tick_formatter is None:
        tick_formatter = lambda tick: f'$\mathregular{{10^{{{np.log10(tick):.1f}}}}}$' if log \
            else lambda tick: f'{tick:.2f}'

    set_ticklabels([tick_formatter(tick) for tick in tick_ix])


def plot_loss_heatmap(model, x, y, param, param_name, param1_range, param2_range,
                      param_log=False, resolution=100, subsample=20,
                      fixed_params=None,
                      cost_fn=None, z_log=True,
                      datapoint_color='#E45756', datapoint_label='data', datapoint_kwargs=None,
                      colorbar=True,
                      legend=False, legend_loc='upper left', fontsize=12,
                      ax=None, tick_num=3, x_tick_formatter=None, y_tick_formatter=None):
    """Plot a heatmap to show the energy landscape for cost function, on two params

    Args:
      model (callable): kinetic model with first argument as x. Broadcast should be
        implemented with x as the innest dimension
      x (list-like): x values of data
      y (list-like): y values of corresponding x data, same length
      param (dict or pd.DataFrame): estimated parameters for model from fitting(s)
      param_name (2-tuple of str): name for two params to scan
      scan_range (dict of two tuple): scan range of two parameters:
        {param1:(low, high),
         param2:(low, high)}
        Note: in model output, dim of param1 should always be out of dim param2
      fixed_params (dict): optional. If there is any fixed params, except for the two to scan
      param_log (bool or dict of bool): if the scan is spacing on log scale
      resolution (int or dict of int): resolution for two scan, default 50
      cost_fn (callable): cost function in calculating cost between y_ and y, take (y_, y). Default is mean squared error
      ax (plt.Axes): Axes to plot on
    """
    from functools import partial

    def mse(y_, y):
        return np.mean((y_ - y) ** 2)

    if cost_fn is None:
        cost_fn = mse

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    if isinstance(param_log, bool):
        param_log = (param_log, param_log)

    def get_scan_point(scan_range, resolution=101, log=False):
        """Generate 1d array contains the scanning point of a dimension"""
        if log:
            return np.logspace(np.log10(scan_range[0]), np.log10(scan_range[1]), resolution)
        else:
            return np.linspace(scan_range[0], scan_range[1], resolution)

    param1_list = get_scan_point(param1_range, resolution[0], param_log[0])
    param2_list = get_scan_point(param2_range, resolution[1], param_log[1])

    if not fixed_params:
        fixed_params = {}

    # generate ys_ with shape (param1, param2)
    if isinstance(param, pd.DataFrame):
        param = param[~np.any(param.isna(), axis=1)]
    if param.shape[0] > subsample:
        param = param.sample(n=subsample, replace=False)
    ys_ = model(x, **{param_name[0]: param1_list, param_name[1]: param2_list}, **fixed_params)
    cost = np.apply_along_axis(arr=ys_, axis=-1, func1d=partial(cost_fn, y=np.array(y)))

    # plot heatmap
    ax = ax_none(ax, figsize=(8, 6))
    ax.grid(False)

    if z_log:
        hm = ax.pcolormesh(np.log10(cost), vmax=np.max(np.log10(cost)), cmap='viridis')
    else:
        hm = ax.pcolormesh(cost, vmax=np.max(cost), cmap='viridis')
    ax.set_xlabel(param_name[0], fontsize=fontsize)
    ax.set_ylabel(param_name[1], fontsize=fontsize)

    # plot marker

    if datapoint_kwargs is None:
        datapoint_kwargs = {'marker': 'x', 'alpha': 0.8}
    else:
        datapoint_kwargs = {**{'marker': 'x', 'alpha': 0.8}, **datapoint_kwargs}

    ax.scatter(value_to_loc(param[param_name[0]], param1_range, resolution[0], param_log[0]),
               value_to_loc(param[param_name[1]], param2_range, resolution[1], param_log[1]),
               color=datapoint_color, label=datapoint_label, **datapoint_kwargs)

    # add ticks
    if x_tick_formatter is None:
        if not param_log[0]:
            x_tick_formatter = lambda tick: f'{tick / resolution  * (param1_range[1] - param1_range[0]):.2f}'
        else:
            x_tick_formatter = lambda tick: f'$\mathregular{{10^{{{np.log10(param1_range[0] * (param1_range[1] / param1_range[0]) ** (tick / resolution[0])):.1f}}}}}$'

    format_ticks(ax=ax, axis='x', tick_num=tick_num, log=False, int_tick_only=False,
                 tick_formatter=x_tick_formatter)

    if y_tick_formatter is None:
        if not param_log[1]:
            y_tick_formatter = lambda tick: f'{tick / resolution  * (param2_range[1] - param2_range[0]):.2f}'
        else:
            y_tick_formatter = lambda tick: f'$\mathregular{{10^{{{np.log10(param2_range[0] * (param2_range[1] / param2_range[0]) ** (tick / resolution[1])):.1f}}}}}$'

    format_ticks(ax=ax, axis='y', tick_num=tick_num, log=False, int_tick_only=False,
                 tick_formatter=y_tick_formatter)
    ax.tick_params(labelsize=fontsize)

    ax.set_xlim([-0.5, resolution[0] + 0.5])
    ax.set_ylim([-0.5, resolution[1] + 0.5])

    if colorbar is True:
        fig = plt.gcf()
        cbar = fig.colorbar(hm, fraction=0.045, pad=0.05, ax=ax)
        cbar.set_label(r'$\log_{10}(MSE)$', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    if legend:
        ax.legend(loc=legend_loc, frameon=True, edgecolor=None, framealpha=0.5)

    return ys_
