def parse_fitting_results(fitting_res, model=None, seq_ix=None, seq_name=None, num_bootstrap_records=0):
    from .least_square import BatchFitter, SingleFitter
    from ..data.seq_data import SeqData

    def extract_info_from_SingleFitting(single_res):
        data = {
            'x_data': single_res.x_data,
            'y_data': single_res.y_data,
            'params': single_res.point_est.params[:len(single_res.config['parameters'])]
        }
        if num_bootstrap_records is not None:
            if single_res.bootstrap.records.shape[0] > num_bootstrap_records:
                data['bs_params'] = single_res.bootstrap.records.iloc[:, :len(single_res.config['parameters'])].sample(
                    axis=0,
                    n=num_bootstrap_records
                )
            else:
                data['bs_params'] = single_res.bootstrap.records.iloc[:, :len(single_res.config['parameters'])]
        return data

    if num_bootstrap_records == 0:
        num_bootstrap_records = None
    if isinstance(fitting_res, SeqData):
        fitting_res = fitting_res.fitting
    if isinstance(fitting_res, BatchFitter):
        if seq_ix is None:
            raise Exception('Please provide the names of sequences to plot')
        else:
            if isinstance(seq_ix, str):
                seq_ix = [seq_ix]
            if seq_name is None:
                seq_name = seq_ixs
            if model is None:
                model = fitting_res.model
            data_to_plot = {
                name: extract_info_from_SingleFitting(single_res = fitting_res.seq_list[seq_ix])
                for name, seq_ix in zip(seq_name, seq_ix)
            }
    elif isinstance(fitting_res, SingleFitter):
        if seq_name is None:
            seq_name = fitting_res.name
        if model is None:
            model = fitting_res.model
        data_to_plot = {
            seq_name: extract_info_from_SingleFitting(single_res=fitting_res)
        }
    else:
        raise Exception('The input fitting_res should be SeqData, SingleFitting or BatchFitting')

    return model, data_to_plot


def fitting_curve_plot(fitting_res, model=None, seq_ix=None, show_data=True, show_bootstrap_curves=50,
                       legend_off=False, axis_labels=('x_label', 'y_label'), seq_name=None,
                       ax=None, save_fig_to=None):

    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax_return = False
    else:
        ax_return = True

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
                                                num_bootstrap_records=show_bootstrap_curves)

    for seq_name, data in data_to_plot.items():
        if not ax_return:
            fig = plt.figure(figsize=[8, 6])
            ax = fig.add_subplot(111)

        if show_data:
            ax.scatter(data['x_data'], data['y_data'], marker='x', s=15, color='#2C73B4', zorder=2)

        x_series = np.linspace(0, np.max(data['x_data']) * 1.2, 100)
        if show_bootstrap_curves:
            for params in data['bs_params'].values:
                ax.plot(x_series, model(x_series, *params), color='#AEAEAE', ls='-', lw=2, alpha=0.2, zorder=1)

        ax.plot(x_series, model(x_series, *data['params'].values), color='#F39730', ls='-', lw=3, zorder=3)
        ax.set_xlim([0, np.max(data['x_data']) * 1.2])
        ax.set_ylim([0, np.max(data['y_data']) * 1.2])

        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0], fontsize=12)
            ax.set_ylabel(axis_labels[1], fontsize=12)

        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.text(s=seq_name,
                x=xlims[0] + 0.05 * (xlims[1] - xlims[0]),
                y=ylims[1] - 0.05 * (ylims[1] - ylims[0]),
                ha='left', va='top',
                fontsize=12)

        if not legend_off:
            from matplotlib.lines import Line2D
            handles = [ax.scatter([], [], marker='x', s=15, color='#2C73B4', label='Data'),
                       Line2D([], [], color='#F39730', ls='-', lw=2, label='Fitted line'),
                       Line2D([], [], color='#AEAEAE', ls='-', lw=2, label='Bootstrapped lines')]
            labels = ['Data', 'Fitted line', 'Bootstrapped lines']
            ax.legend(handles, labels, frameon=False, loc='lower right')
        if save_fig_to:
            plt.savefig(save_fig_to, bbox_inches='tight', dpi=300)
    if ax_return:
        return ax
    plt.show()


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
    from .least_square import BatchFitResults, BatchFitter
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