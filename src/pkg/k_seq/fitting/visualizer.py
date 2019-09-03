def parse_fitting_results(fitting_res, model=None, seq_ix=None, seq_name=None, num_bootstrap_records=0):
    from .fitting import SingleFitting, BatchFitting
    from ..data.seq_table import SeqTable

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
    if isinstance(fitting_res, SeqTable):
        fitting_res = fitting_res.fitting
    if isinstance(fitting_res, BatchFitting):
        if seq_ix is None:
            raise Exception('Please provide the names of sequences to plot')
        else:
            if isinstance(seq_ix, str):
                seq_ix = [seq_ix]
            if seq_name is None:
                seq_name = seq_ix
            if model is None:
                model = fitting_res.model
            data_to_plot = {
                name: extract_info_from_SingleFitting(single_res = fitting_res.seq_list[seq_ix])
                for name, seq_ix in zip(seq_name, seq_ix)
            }
    elif isinstance(fitting_res, SingleFitting):
        if seq_name is None:
            seq_name = fitting_res.name
        if model is None:
            model = fitting_res.model
        data_to_plot = {
            seq_name: extract_info_from_SingleFitting(single_res=fitting_res)
        }
    else:
        raise Exception('The input fitting_res should be SeqTable, SingleFitting or BatchFitting')

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


def param_value_plot(fitting_res, param, with_ci=True, show_point_est=False, use_mean_sd=False,
                     sort_by='median', y_log=False, seq_filter=None, seq_table=None, save_fig_to=None):

    import matplotlib.pyplot as plt
    import numpy as np
    from .fitting import BatchFitting
    from ..data.seq_table import SeqTable

    def plot_with_shade(ax, x, center, high, low):
        ax.plot(x, center, '.-', color='#2C73B4')
        if (high is not None) and (low is not None):
            ax.fill_between(x, y1=high, y2=low, alpha=0.5, facecolor='#2C73B4')

    cols = [param + post_fix for post_fix in ['_point_est', '_mean', '_std', '_median', '_2.5', '_97.5']]
    if isinstance(fitting_res, BatchFitting):
        param_values = fitting_res.summary[cols]
    elif isinstance(fitting_res, SeqTable):
        param_values = fitting_res.fitting.summary[col]
    else:
        raise Exception('Error: fitting_res has to be an either SeqTable or BatchFitting instance')
    if seq_filter is not None:
        if isinstance(seq_filter, list):
            param_values = param_values.loc[seq_filter]
        elif callable(seq_filter):
            param_values = param_values[[seq_filter(fitting_res.seq_list[seq_ix]) for seq_ix in param_values.index]]

    if isinstance(sort_by, str):
        if sort_by.lower() in ['median', 'point_est', 'mean']:
            param_values = param_values.sort_values(by=param + '_' + sort_by)
        elif sort_by.lower() in ['abun', 'rel_abun', 'relative abundance']:
            if isinstance(fitting_res, SeqTable):
                if seq_table is None:
                    seq_table = fitting_res
            else:
                if seq_table is None:
                    raise Exception('Error: please provide a SeqTable instance for relative abundance')
            param_values['rel_abun'] = seq_table.seq_info['avg_rel_abun_in_inputs'].loc[param_values.index]
            param_values = param_values.sort_values(by='rel_abun')
    elif callable(sort_by):
        sort_fn = lambda x: sort_by(fitting_res.seq_list[x])
        param_values = param_values.reindex(sorted(param_values.index, key=sort_fn))

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_subplot(111)
    pos = np.linspace(0, param_values.shape[0] - 1, param_values.shape[0])
    if use_mean_sd:
        center = param_values[param + '_mean']
        high = param_values[param + '_mean'] - 1.96 * param_values[param + '_std']
        low = param_values[param + '_mean'] + 1.96 * param_values[param + '_std']
    else:
        center = param_values[param + '_median']
        high = param_values[param + '_97.5']
        low = param_values[param + '_2.5']
    if with_ci:
        plot_with_shade(ax=ax, x=pos, center=center, high=high, low=low)
    else:
        plot_with_shade(ax=ax, x=pos, center=center, high=None, low=None)

    if show_point_est:
        ax.scatter(x=pos, y=param_values[param + '_point_est'], marker='x', color='#F39730')

    if y_log:
        ax.set_yscale('log')

    ax.set_xlabel('Sequences', fontsize=14)
    ylabel = 'Estimated ' + param
    if use_mean_sd:
        ylabel += ' (mean)'
    else:
        ylabel += ' (median)'
    if with_ci:
        ylabel += ' with 95% CI'
    ax.set_ylabel(ylabel, fontsize=14)

    if save_fig_to:
        fig.savefig(save_fit_to, bbox_inches='tight', dpi=300)

    plt.show()



# def get_loss(x, y, params, func=, weights=None):
#     y_ = func(x, *params)
#     if not weights:
#         weights = np.ones(len(x))
#     return sum(((y_-y) / weights)**2)


# def convergence_test(x, y, func=func_default, weights=None, param_bounds=([0, 0], [1., np.inf]),
#                      test_size=100, return_verbose=True, key_value='loss',
#                      statistics=None):
#
#     from inspect import signature
#     param_num = len(str(signature(func)).split(',')) - 1
#     results = {
#         'params': np.zeros((param_num, test_size)),
#         'loss': np.zeros(test_size)
#     }
#
#     for rep in range(test_size):
#         try:
#             init_guess = ([np.random.random() for _ in range(param_num)])
#             if param_bounds:
#                 popt, pcov = curve_fit(func, x, y, method='trf', bounds=param_bounds, p0=init_guess, sigma=weights)
#             else:
#                 popt, pcov = curve_fit(func, x, y, method='trf', p0=init_guess, sigma=weights)
#         except RuntimeError:
#             popt = None
#         if popt is not None:
#             results['params'][:, rep] = popt
#             results['loss'][rep] = get_loss(x, y, params=popt, func=func, weights=weights)
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
# def fitting_check(k, A, xTrue, y, size=100, average=True):
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
#     for epochs in range(size):
#         # initGuess= (np.random.random(), np.random.random()*k*100)
#         initGuess = (np.random.random(), np.random.random())
#
#         try:
#             popt, pcov = curve_fit(func, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
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