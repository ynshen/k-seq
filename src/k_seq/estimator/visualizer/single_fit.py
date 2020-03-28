import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ...utility.plot_tools import ax_none
from ...utility.func_tools import update_none
from yutility import logging

__all__ = ['plot_curve', 'plot_loss_heatmap']


def plot_curve(model, x, y, param=None, major_param=None, subsample=20,
               x_label=None, y_label=None, x_lim=None, y_lim=None,
               major_curve_kwargs=None, curve_kwargs=None, datapoint_kwargs=None,
               major_curve_label='major curve', curve_label='curves', datapoint_label='data',
               legend=False, legend_loc='upper right', ax=None):
    """Plot curve of fitting results
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
    ax = ax_none(ax, figsize=(4, 3))
    major_curve_kwargs = update_none(major_curve_kwargs, {})
    curve_kwargs = update_none(curve_kwargs, {})
    datapoint_kwargs = update_none(datapoint_kwargs, {})

    # prep plotting kwargs
    major_curve_kwargs = {**{'color': '#4C78A8', 'alpha': 0.5}, **major_curve_kwargs}
    if major_param is None:
        curve_kwargs = {**{'color': '#4C78A8', 'alpha': 0.5}, **curve_kwargs}
    else:
        curve_kwargs = {**{'color': '#AEAEAE', 'alpha': 0.3}, **datapoint_kwargs}
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
            add_curve(param)
        elif isinstance(param, pd.DataFrame):
            if param.shape[0] > subsample:
                param = param.sample(subsample)
        param.apply(add_curve, axis=1, plot_args=curve_kwargs)

    # add major curve if applicable
    if major_param:
        add_curve(major_param, plot_args=major_curve_kwargs)

    # add raw data points
    ax.scatter(x, y, label=datapoint_label, **datapoint_kwargs)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if x_label:
        ax.set_xlabel(x_label, fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)

    if legend:
        if major_param is not None:
            ax.plot([0], [0], label=major_curve_label, **major_curve_kwargs)
        if param is not None:
            ax.plot([0], [0], label=curve_label, **curve_kwargs)
        ax.legend(loc=legend_loc, frameon=True, edgecolor=None, framealpha=0.5)


def mse(y_, y):
    """Unweighted mean square error between y and y_"""
    return np.mean((y_ - y)**2)


def plot_loss_heatmap(model, x, y, param, param_name, param1_range, param2_range,
                      param_log=False, resolution=100,
                      fixed_params=None,
                      cost_fn=mse, z_log=True,
                      datapoint_color='#E45756', datapoint_label='data', datapoint_kwargs=None,
                      legend=False, legend_loc='upper left',
                      ax=None):
    """Plot a heatmap to show the energy landscape for cost function, on two params

    Args:
      model (callable): kinetic model with first argument as x. Broadcase should be
        implemented with x as the innest dimension
      x (list-like): x values of data
      y (list-like): y values of corresponding x data, same length
      param (dict or pd.DataFrame): estimated parameters for model from fitting(s)
      param_name (2-tuple of str): name for two params to scan
      scan_range (dict of two tuple): scan range of two parameters:
        {param1:(low, high),
         param2:(low, high)}
        Note: in model output, dim of param1 should always be out of dim param2
      fix_params (dict): optional. If there is any fixed params, except for the two to scan
      param_log (bool or dict of bool): if the scan is spacing on log scale
      resolution (int or dict of int): resolution for two scan, default 50
      cost_fn (callable): cost function in calculating cost between y_ and y, take (y_, y)
      ax (plt.Axes): Axes to plot on
    """
    from functools import partial

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
    ys_ = model(x, **{param_name[0]: param1_list, param_name[1]: param2_list}, **fixed_params)
    cost = np.apply_along_axis(arr=ys_, axis=-1, func1d=partial(cost_fn, y=np.array(y)))

    # plot heatmap
    ax = ax_none(ax, figsize=(8, 6))
    ax.grid(False)

    if z_log:
        hm = ax.pcolormesh(np.log10(cost), vmax=np.max(np.log10(cost)), cmap='viridis')
    else:
        hm = ax.pcolormesh(cost, vmax=np.max(cost), cmap='viridis')
    ax.set_xlabel(param_name[0], fontsize=14)
    ax.set_ylabel(param_name[1], fontsize=14)

    # plot marker

    def value_to_loc(value, param_to_scan, log):
        """Convert value for parameter to its location on resolution scale"""

        vmin = param_to_scan[0]
        vmax = param_to_scan[-1]

        if log:
            return np.log10(value / vmin) / np.log10(vmax / vmin) * len(param_to_scan)
        else:
            return (value - vmin) / (vmax - vmin) * len(param_to_scan)

    if datapoint_kwargs is None:
        datapoint_kwargs = {'marker': 'x', 'alpha': 0.8}
    else:
        datapoint_kwargs = {**{'marker': 'x', 'alpha': 0.8}, **datapoint_kwargs}

    ax.scatter(value_to_loc(param[param_name[0]], param1_list, param_log[0]),
               value_to_loc(param[param_name[1]], param2_list, param_log[1]),
               color=datapoint_color, label=datapoint_label, **datapoint_kwargs)

    # add ticks
    tick_ix = np.linspace(0, resolution[0] - 1, 5, dtype=int)
    ax.set_xticks(tick_ix)

    if param_log[0]:
        ax.set_xticklabels([f'$\mathregular{{10^{{{int(np.log10(tick))}}}}}$' for tick in param1_list[tick_ix]])
    else:
        ax.set_xticklabels([f'{tick:.2f}' for tick in param1_list[tick_ix]])

    tick_ix = np.linspace(0, resolution[1] - 1, 5, dtype=int)
    ax.set_yticks(tick_ix)
    if param_log[1]:
        ax.set_yticklabels([f'$\mathregular{{10^{{{int(np.log10(tick))}}}}}$' for tick in param2_list[tick_ix]])
    else:
        ax.set_yticklabels([f'{tick:.2f}' for tick in param2_list[tick_ix]])

    ax.set_xlim([-0.5, resolution[0] - 0.5])
    ax.set_ylim([-0.5, resolution[1] - 0.5])

    fig = plt.gcf()
    cbar = fig.colorbar(hm, fraction=0.045, pad=0.05)
    cbar.set_label(r'$\log_{10}(MSE)$', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    if legend:
        ax.legend(loc=legend_loc, frameon=True, edgecolor=None, framealpha=0.5)

    return ys_


# def fitting_curve_plot(fitting_res, model=None, seq_ix=None, show_data=True, show_bootstrap_curves=50,
#                        legend_off=False, axis_labels=('x_label', 'y_label'), seq_name=None,
#                        ax=None, save_fig_to=None):
#
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     if ax is None:
#         ax_return = False
#     else:
#         ax_return = True
#
#     if seq_ix is None:
#         seq_ix = None
#     if seq_name is None:
#         seq_name = None
#     if model is None:
#         model = None
#
#     model, data_to_plot = parse_fitting_results(fitting_res=fitting_res,
#                                                 model=model,
#                                                 seq_ix=seq_ix,
#                                                 seq_name=seq_name,
#                                                 num_bootstrap_records=show_bootstrap_curves)
#
#     for seq_name, data in data_to_plot.items():
#         if not ax_return:
#             fig = plt.figure(figsize=[8, 6])
#             ax = fig.add_subplot(111)
#
#         if show_data:
#             ax.scatter(data['x_data'], data['y_data'], marker='x', s=15, color='#2C73B4', zorder=2)
#
#         x_series = np.linspace(0, np.max(data['x_data']) * 1.2, 100)
#         if show_bootstrap_curves:
#             for params in data['bs_params'].values:
#                 ax.plot(x_series, model(x_series, *params), color='#AEAEAE', ls='-', lw=2, alpha=0.2, zorder=1)
#
#         ax.plot(x_series, model(x_series, *data['params'].values), color='#F39730', ls='-', lw=3, zorder=3)
#         ax.set_xlim([0, np.max(data['x_data']) * 1.2])
#         ax.set_ylim([0, np.max(data['y_data']) * 1.2])
#
#         if axis_labels is not None:
#             ax.set_xlabel(axis_labels[0], fontsize=12)
#             ax.set_ylabel(axis_labels[1], fontsize=12)
#
#         xlims = ax.get_xlim()
#         ylims = ax.get_ylim()
#         ax.text(s=seq_name,
#                 x=xlims[0] + 0.05 * (xlims[1] - xlims[0]),
#                 y=ylims[1] - 0.05 * (ylims[1] - ylims[0]),
#                 ha='left', va='top',
#                 fontsize=12)
#
#         if not legend_off:
#             from matplotlib.lines import Line2D
#             handles = [ax.scatter([], [], marker='x', s=15, color='#2C73B4', label='Data'),
#                        Line2D([], [], color='#F39730', ls='-', lw=2, label='Fitted line'),
#                        Line2D([], [], color='#AEAEAE', ls='-', lw=2, label='Bootstrapped lines')]
#             labels = ['Data', 'Fitted line', 'Bootstrapped lines']
#             ax.legend(handles, labels, frameon=False, loc='lower right')
#         if save_fig_to:
#             plt.savefig(save_fig_to, bbox_inches='tight', dpi=300)
#     if ax_return:
#         return ax
#     plt.show()


def bootstrap_params_dist_plot(fitting_res, model=None, seq_ix=None, params_to_plot=['k', 'A'],
                               subsample_num=500, seq_name=None, save_fig_to=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if seq_ix is None:
        seq_ix = None
    if seq_name is None:
        seq_name = None
    if model is None:
        model = None

    model, data_to_plot = parse_fitting_results(fitting_res=fitting_res,
                                                model=model,
                                                seq_ix=seq_ix,
                                                seq_name=seq_name,
                                                num_bootstrap_records=subsample_num)

    for seq_name, data in data_to_plot.items():
        jp = sns.jointplot(x=params_to_plot[0], y=params_to_plot[1], data=data['bs_params'],
                           kind='scatter')
        if save_fig_to:
            jp.savefig(save_fit_to, bbox_inches='tight', dpi=300)


def param_value_plot(fitting_res, param, seq_to_show=None, ax=None,
                     line_postfix='', line_color='#2C73B4', line_marker='',
                     show_shade=True, upper_postfix='_97.5%', lower_postfix='_2.5%',
                     shade_color='#2C73B4', shade_alpha=0.3,
                     sort_by=None, y_log=False, save_fig_to=None, **kwargs):
    """Plot of given estimated parameter values across selected sequences"""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .least_squares import BatchFitResults, BatchFitter
    from ..data.seq_data import SeqData

    def get_res_col():
        """parse col name to use"""

        def check_col(param, postfix):
            if (param + postfix) in param_values.columns:
                return param + postfix
            elif postfix in param_values.columns:
                return postfix
            else:
                print(param_values.columns)
                raise IndexError(f"Column '{param + postfix}' or '{postfix}' not found in result")

        line_col = check_col(param, line_postfix)
        if show_shade:
            lower_col = check_col(param, lower_postfix)
            upper_col = check_col(param, upper_postfix)
        else:
            lower_col = None
            upper_col = None
        return line_col, lower_col, upper_col

    def parse_fitting_res():
        """Parse various fitting result input type and get the `pd.DataFrame` output of summary"""

        if isinstance(fitting_res, pd.DataFrame):
            return fitting_res
        elif isinstance(fitting_res, BatchFitter):
            return fitting_res.results.summary()
        elif isinstance(fitting_res, BatchFitResults):
            return fitting_res.summary()
        elif isinstance(fitting_res, SeqData):
            return fitting_res.fitter.results.summary
        else:
            raise Exception('Error: fitting_res has to be an either SeqData or BatchFitting instance')

    param_values = parse_fitting_res()
    line_col, lower_col, upper_col = get_res_col()

    if seq_to_show is not None:
        if isinstance(seq_to_show, list):
            param_values = param_values.loc[seq_to_show]
        elif callable(seq_to_show):
            param_values = param_values[param_values.index.to_series().apply(seq_to_show)]

    if isinstance(sort_by, str):
        if sort_by in param_values.columns:
            param_values = param_values.sort_values(by=sort_by)
        else:
            raise IndexError(f'Unknown column name {sort_by} to sort')
    elif callable(sort_by):
        new_index = param_values.apply(sort_by, axis=1).index
        param_values = param_values.reindex(new_index)
    else:
        if sort_by is not None:
            raise TypeError('Unknown sorting method')

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (12, 6)))

    pos = np.arange(param_values.shape[0])
    ax.plot(pos, param_values[line_col], marker=line_marker, color=line_color, ls='-')
    if show_shade:
        ax.fill_between(pos, y1=param_values[upper_col], y2=param_values[lower_col],
                        facecolor=shade_color, alpha=shade_alpha)
    if y_log:
        ax.set_yscale('log')
    ax.set_xlabel('Sequences', fontsize=14)
    ax.set_ylabel(param, fontsize=14)

    if save_fig_to:
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return ax


# def get_loss(x, y, params, _get_mask=, weights=None):
#     y_ = _get_mask(x, *params)
#     if not weights:
#         weights = np.ones(len(x))
#     return sum(((y_-y) / weights)**2)


# def convergence_test(x, y, _get_mask=func_default, weights=None, param_bounds=([0, 0], [1., np.inf]),
#                      test_size=100, return_verbose=True, key_value='loss',
#                      statistics=None):
#
#     from inspect import signature
#     param_num = len(str(signature(_get_mask)).split(',')) - 1
#     results = {
#         'params': np.zeros((param_num, test_size)),
#         'loss': np.zeros(test_size)
#     }
#
#     for rep in range(test_size):
#         try:
#             init_guess = ([np.random.random() for _ in range(param_num)])
#             if param_bounds:
#                 popt, pcov = curve_fit(_get_mask, x, y, method='trf', bounds=param_bounds, p0=init_guess, sigma=weights)
#             else:
#                 popt, pcov = curve_fit(_get_mask, x, y, method='trf', p0=init_guess, sigma=weights)
#         except RuntimeError:
#             popt = None
#         if popt is not None:
#             results['params'][:, rep] = popt
#             results['loss'][rep] = get_loss(x, y, params=popt, _get_mask=_get_mask, weights=weights)
#     if return_verbose:
#         return results
#     else:
#         if key_value == 'loss':
#             key_stat = results['loss']
#         else:
#             key_stat = results['params'][key_value]
#         if statistics:
#             return statistics(key_stat)
#         else:
#             return (np.max(key_stat) - np.min(key_stat))/np.mean(key_stat)
#
#
#
#
# def fitting_check(k, A, xTrue, y, uniq_seq_num=100, average=True):
#     np.random.seed(23)
#
#     fittingRes = {
#         'y_': None,
#         'x_': None,
#         'k': [],
#         'kerr': [],
#         'A': [],
#         'Aerr': [],
#         'kA': [],
#         'kAerr': [],
#         'mse': [],
#         'mseTrue': [],
#         'r2': []
#     }
#
#     if average:
#         y_ = np.mean(y, axis=0)
#         x_ = np.mean(xTrue, axis=0)
#     else:
#         y_ = np.reshape(y, y.shape[0] * y.shape[1])
#         x_ = np.reshape(xTrue, xTrue.shape[0] * xTrue.shape[1])
#
#     for epochs in range(uniq_seq_num):
#         # initGuess= (np.random.random(), np.random.random()*k*100)
#         initGuess = (np.random.random(), np.random.random())
#
#         try:
#             popt, pcov = curve_fit(_get_mask, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
#         except RuntimeError:
#             popt = [np.nan, np.nan]
#
#         if fittingRes['y_'] is None:
#             fittingRes['y_'] = y_
#         if fittingRes['x_'] is None:
#             fittingRes['x_'] = x_
#         fittingRes['k'].append(popt[1])
#         fittingRes['kerr'].append((popt[1] - k) / k)
#         fittingRes['A'].append(popt[0])
#         fittingRes['Aerr'].append((popt[0] - A) / A)
#         fittingRes['kA'].append(popt[0] * popt[1])
#         fittingRes['kAerr'].append((popt[0] * popt[1] - k * A) / (k * A))
#
#         fittingRes['mse'].append(mse(x_, y_, A=popt[0], k=popt[1]))
#         fittingRes['mseTrue'].append(mse(x_, y_, A=A, k=k))
#
#         res = y_ - (1 - np.exp(-0.479 * 90 * popt[1] * x_)) * popt[0]
#         ss_res = np.sum(res ** 2)
#         ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
#         fittingRes['r2'].append(1 - ss_res / ss_tot)
#
#     return fittingRes