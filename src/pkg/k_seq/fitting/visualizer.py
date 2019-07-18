def parse_fitting_results(fitting_res, num_bootstrap_records=0):

    def extract_info_from_SingleFitting(single_res, show_bootstrap_curves=None):
        data = {
            'x_data': single_res.x_data,
            'y_data': single_res.y_data,
            'params': single_res.point_est.params[:len(single_res.config['parameters'])].values
        }
        if show_bootstrap_curves is not None:
            data['bs_params'] = single_res.bootstrap.records.iloc[:, :len(single_res.config['parameters'])].sample(
                axis=0,
                n=show_bootstrap_curves
            ).values
        return data

    if show_bootstrap_curves == 0:
        show_bootstrap_curves = None
    if isinstance(fitting_res, SeqTable):
        fitting_res = fitting_res.fitting
    if isinstance(fitting_res, BatchFitting):
        print('input is BatchFitting')
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
                name: extract_info_from_SingleFitting(single_res = fitting_res.seq_list[seq_ix],
                                                      show_bootstrap_curves=show_bootstrap_curves)
                for name, seq_ix in zip(seq_name, seq_ix)
            }
    if isinstance(fitting_res, SingleFitting):
        if seq_name is None:
            seq_name = fitting_res.name
        if model is None:
            model = fitting_res.model
        data_to_plot = {
            seq_name: extract_info_from_SingleFitting(single_res=fitting_res, show_bootstrap_curves=show_bootstrap_curves)
        }

    return model, data_to_plot

def fitting_curve_plot(fitting_res, model=None, seq_ix=None, show_data=True, show_bootstrap_curves=50,
                       legend_off=False, axis_labels=('x_label', 'y_label'), seq_name=None,
                       ax=None, save_fig_to=None):
    from .fitting import SingleFitting, BatchFitting
    from ..data.pre_processing import SeqTable
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax_return = False
    else:
        ax_return = True

    for seq_name, data in data_to_plot.items():
        if not ax_return:
            fig = plt.figure(figsize=[8, 6])
            ax = fig.add_subplot(111)

        if show_data:
            ax.scatter(data['x_data'], data['y_data'], marker='x', s=15, color='#2C73B4', zorder=2)

        x_series = np.linspace(0, np.max(data['x_data']) * 1.2, 100)
        if show_bootstrap_curves:
            for params in data['bs_params']:
                ax.plot(x_series, model(x_series, *params), color='#AEAEAE', ls='-', lw=2, alpha=0.2, zorder=1)

        ax.plot(x_series, model(x_series, *data['params']), color='#F39730', ls='-', lw=3, zorder=3)
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

def


# def fitting_check(k, A, xTrue, y, size=100, average=True):
#
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
# #
# #     return fittingRes


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