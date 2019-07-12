import numpy as np
from scipy.optimize import curve_fit


def plot_fitting_single(seq_res, func, data=True, bootstrap_curve=50,
                        legend_off=False, axis_labels=('x_label', 'y_label'), seq_name=None,
                        ax=None, save_dirc=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=[12, 8])
        ax = fig.add_subplot(111)
        ax_return = False
    else:
        ax_return = True
    if data:
        ax.scatter(seq_res[0]['x_data'], seq_res[0]['y_data'], marker='x', s=15, color='#2C73B4', zorder=2)

    x_series = np.linspace(0, np.max(seq_res[0]['x_data']) * 1.2, 100)

    if bootstrap_curve is not None:
        rand_ix = np.random.choice([i for i in range(len(seq_res[1]))],
                                   replace=False, size=bootstrap_curve)
        for ix in rand_ix:
            ax.plot(x_series, func(x_series, *seq_res[1][ix]),
                    color='#AEAEAE', ls='-', lw=2, alpha=0.2, zorder=1)

    ax.plot(x_series, func(x_series, *seq_res[0]['p50']),
            color='#F39730', ls='-', lw=3, zorder=3)

    ax.set_xlim([0, np.max(seq_res[0]['x_data']) * 1.2])
    ax.set_ylim([0, np.max(seq_res[0]['y_data']) * 1.2])

    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0], fontsize=12)
        ax.set_ylabel(axis_labels[1], fontsize=12)

    if seq_name is not None:
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

    if ax_return:
        return ax
    if save_dirc:
        fig.savefig(save_dirc, bbox_inches='tight', dpi=300)
    plt.show()

def func_default(x, A, k):
    return A * (1 - np.exp(-0.479 * 90 * k * x))

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


def get_loss(x, y, params, func=func_default, weights=None):
    y_ = func(x, *params)
    if not weights:
        weights = np.ones(len(x))
    return sum(((y_-y) / weights)**2)


def convergence_test(x, y, func=func_default, weights=None, param_bounds=([0, 0], [1., np.inf]),
                     test_size=100, return_verbose=True, key_value='loss',
                     statistics=None):

    from inspect import signature
    param_num = len(str(signature(func)).split(',')) - 1
    results = {
        'params': np.zeros((param_num, test_size)),
        'loss': np.zeros(test_size)
    }

    for rep in range(test_size):
        try:
            init_guess = ([np.random.random() for _ in range(param_num)])
            if param_bounds:
                popt, pcov = curve_fit(func, x, y, method='trf', bounds=param_bounds, p0=init_guess, sigma=weights)
            else:
                popt, pcov = curve_fit(func, x, y, method='trf', p0=init_guess, sigma=weights)
        except RuntimeError:
            popt = None
        if popt is not None:
            results['params'][:, rep] = popt
            results['loss'][rep] = get_loss(x, y, params=popt, func=func, weights=weights)
    if return_verbose:
        return results
    else:
        if key_value == 'loss':
            key_stat = results['loss']
        else:
            key_stat = results['params'][key_value]
        if statistics:
            return statistics(key_stat)
        else:
            return (np.max(key_stat) - np.min(key_stat))/np.mean(key_stat)




def fitting_check(k, A, xTrue, y, size=100, average=True):
    np.random.seed(23)

    fittingRes = {
        'y_': None,
        'x_': None,
        'k': [],
        'kerr': [],
        'A': [],
        'Aerr': [],
        'kA': [],
        'kAerr': [],
        'mse': [],
        'mseTrue': [],
        'r2': []
    }

    if average:
        y_ = np.mean(y, axis=0)
        x_ = np.mean(xTrue, axis=0)
    else:
        y_ = np.reshape(y, y.shape[0] * y.shape[1])
        x_ = np.reshape(xTrue, xTrue.shape[0] * xTrue.shape[1])

    for epochs in range(size):
        # initGuess= (np.random.random(), np.random.random()*k*100)
        initGuess = (np.random.random(), np.random.random())

        try:
            popt, pcov = curve_fit(func, x_, y_, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]

        if fittingRes['y_'] is None:
            fittingRes['y_'] = y_
        if fittingRes['x_'] is None:
            fittingRes['x_'] = x_
        fittingRes['k'].append(popt[1])
        fittingRes['kerr'].append((popt[1] - k) / k)
        fittingRes['A'].append(popt[0])
        fittingRes['Aerr'].append((popt[0] - A) / A)
        fittingRes['kA'].append(popt[0] * popt[1])
        fittingRes['kAerr'].append((popt[0] * popt[1] - k * A) / (k * A))

        fittingRes['mse'].append(mse(x_, y_, A=popt[0], k=popt[1]))
        fittingRes['mseTrue'].append(mse(x_, y_, A=A, k=k))

        res = y_ - (1 - np.exp(-0.479 * 90 * popt[1] * x_)) * popt[0]
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_ - np.mean(y_)) ** 2)
        fittingRes['r2'].append(1 - ss_res / ss_tot)

    return fittingRes