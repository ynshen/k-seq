"""
Methods needed for fitting
"""

from scipy.optimize import curve_fit
import numpy as np


def func_default(x, A, k):
    return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BYO degradation adjustment and 90 minutes


def fitting_single(x_data, y_data, func=func_default, weights=None, bounds=None,
                   ci_est=True, bs_depth=500, bs_return_all=True, bs_fix_x=False,
                   missing_data_as_zero=False, y_max=None):
    from inspect import signature

    # get number of args in func
    sig = signature(func)
    param_num = len(sig.parameters) - 1
    # initialize data to fit
    if y_max is not None:
        y_data = np.array([min(yi, y_max) for yi in y_data])
    if missing_data_as_zero:
        y_data[np.isnan(y_data)] = 0
    if not weights:
        weights = np.ones(len(y_data))
    valid = ~np.isnan(y_data)
    x_data = x_data[valid]
    y_data = y_data[valid]
    weights = weights[valid]
    results = {
        'x_data': x_data,
        'y_data': y_data,
        'fitting_weights': weights
    }
    try:
        init_guess = [np.random.random() for _ in range(param_num)]
        results['params'], results['pcov'] = curve_fit(func, xdata=x_data, ydata=y_data, sigma=weights,
                                                       method='trf', bounds=bounds, p0=init_guess)
        y_hat = func(x_data, *results['params'])
        res = y_data - y_hat
        results['pct_res'] = res / y_hat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        results['r2'] = (1 - ss_res / ss_tot)
    except RuntimeError:
        results['params'] = [np.nan for _ in range(param_num)]

    if ci_est:
        if (len(x_data) > 1)and(~np.isnan(results['params'][0])):
            param_list = []
            for _ in range(bs_depth):
                if bs_fix_x:
                    pct_res_resampled = np.random.choice(results['pct_res'], replace=True, size=len(results['pct_res']))  # need to change to generic form
                    y_data_bs = y_hat * (1 + pct_res_resampled)
                    x_data_bs = x_data
                else:
                    indeces = np.linspace(0, len(x_data) - 1, len(x_data))
                    bs_indeces = np.random.choice(a=indeces, size=len(x_data), replace=True)
                    x_data_bs = np.array([x_data[int(i)] for i in bs_indeces])
                    y_data_bs = np.array([y_data[int(i)] for i in bs_indeces])
                try:
                    init_guess = [np.random.random() for _ in range(param_num)]
                    params, pcov = curve_fit(func, xdata=x_data_bs, ydata=y_data_bs,
                                             method='trf', bounds=bounds, p0=init_guess)
                except:
                    params = [np.nan for _ in range(param_num)]
                param_list.append(params)
            if bs_return_all:
                results['bs_params'] = param_list
            results['mean'] = np.nanmean(param_list, axis=0)
            results['sd'] = np.nanstd(param_list, axis=0, ddof=1)
            results['percentiles'] = {
                '2.5': np.percentile(param_list, 2.5, axis=0),
                '50': np.percentile(param_list, 50, axis=0),
                '97.5': np.percentile(param_list, 97.5, axis=0)
            }
        else:
            results['sd'] = np.nan

    return results



# fitting_main is not ready to use
def fitting_main(seqToFit=None, maxFold=None, fitMtd='trf', ciEst=True, func=None):
    import util
    import time
    import multiprocessing as mp

    timeInit = time.time()
    pool = mp.Pool(processes=8)
    if simuSet is None:
        simuSet = util.load_pickle('/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_2_full.pkl')
    simuSet = pool.map(method_5_multi, simuSet)
    timeEnd = time.time()
    print('Process finished in %i s' % (timeEnd - timeInit))

    if not (simuSet is None):
        return simuSet
    else:
        pass
        # util.dump_pickle(simuSet,
        #              '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_1_method_4_parallel_res.pkl',
        #              log='10000 simulated data on [3,3,3,3] using method 5: bootstrap residues for 500 times, parallelized',
        #              overwrite=True)

    return seqToFit