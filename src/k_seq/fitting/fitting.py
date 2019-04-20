"""
Methods needed for fitting
"""

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import multiprocessing as mp


def func_default(x, A, k):
    """
    Default kinetic model used in BYO k-seq fitting:
                    A * (1 - np.exp(- 0.479 * 90 * k * x))
    90: t, reaction time (min)
    0.479: alpha, degradation adjustment parameter

    :param x: predictor for the regression model, here is initial concentration of BYO
    :param A: parameter represents the maximal conversion of reactants
    :param k: parameter represents the apparent kinetic coefficient
    :return: reacted fraction given the independent variable x and parameter (A, k)
    """
    return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BYO degradation adjustment and 90 minutes


def get_args_num(func, exclude_x=True):
    """
    Utility function to get the number of arguments for a callable
    :param func: callable, the function
    :param exclude_x: boolean, return the number of arguments minus 1 if true
    :return: number of arguments of the function func
    """
    from inspect import signature
    sig = signature(func)
    if exclude_x:
        param_num = len(sig.parameters) - 1
    else:
        param_num = len(sig.parameters)
    return param_num


def fitting_single(x_data, y_data, func=func_default, weights=None, bounds=None,
                   bootstrap=True, bs_depth=1000, bs_return_verbose=True, bs_residue=False,
                   missing_data_as_zero=False, y_max=None, **kwargs):

    """
    Core method for fitting. Fit on a single sequence
    :param x_data: list-like. A list of x values of data points to fit
    :param y_data: list-like, same size as x_data. y values of data points to fit, should keep same order as x_data.
                   np.nan is allowed
    :param func: callable. Model function for fitting.
    :param weights: optional. Weights of each data points in fitting, same size as x_data
    :param bounds: a tuple of two tuples.
                   ((lower_bound_0, lower_bound_1, ..., lower_bound_k),
                    (upper_bound_0, upper_bound_1, ..., upper_bound_k))
    :param bootstrap: boolean. If true, use bootstrap to estimate the confidence interval of parameters
    :param bs_depth: int. Number of bootstrap samples to use to estimate the confidence interval
    :param bs_return_verbose: boolean. If true, return the list of parameters for each bootstrap sample
    :param bs_residue: boolean. If true, resample precent residue instead of data points in bootstrap
    :param missing_data_as_zero: boolean. If true, the missing data (np.nan) will be treated as zero
    :param y_max: optional. float if not None. The maximum y value can be accepted
    :param kwargs: no other key arguments, only for completeness
    :return: results: a dictionary of fitting results,
             param_list: a list of parameters from each bootstrap samples. None if bs_return_verbose = False
    """
    param_num = get_args_num(func)
    # regularize data format
    x_data = np.array(list(x_data))
    y_data = np.array(list(y_data))
    if y_max is not None:
        y_data = np.array([min(yi, y_max) for yi in y_data])
    if missing_data_as_zero:
        y_data[np.isnan(y_data)] = 0
    if bounds is None:
        bounds = [[-np.inf for _ in range(param_num)], [np.inf for _ in range(param_num)]]
    if not weights:
        weights = np.ones(len(y_data))
    # only include non np.nan data
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
    except RuntimeError:
        results['params'] = [np.nan for _ in range(param_num)]

    if bootstrap:
        param_list = []
        if (len(x_data) > 1)and(~np.isnan(results['params'][0])):
            for _ in range(bs_depth):
                if bs_residue:
                    pct_res_resampled = np.random.choice(results['pct_res'], replace=True, size=len(results['pct_res']))
                    y_data_bs = y_hat * (1 + pct_res_resampled)
                    x_data_bs = x_data
                else:
                    indices = np.linspace(0, len(x_data) - 1, len(x_data))
                    bs_indeces = np.random.choice(a=indices, size=len(x_data), replace=True)
                    x_data_bs = np.array([x_data[int(i)] for i in bs_indeces])
                    y_data_bs = np.array([y_data[int(i)] for i in bs_indeces])
                try:
                    init_guess = [np.random.random() for _ in range(param_num)]
                    params, pcov = curve_fit(func, xdata=x_data_bs, ydata=y_data_bs,
                                             method='trf', bounds=bounds, p0=init_guess)
                except:
                    params = [np.nan for _ in range(param_num)]
                param_list.append(params)

            results['mean'] = np.nanmean(param_list, axis=0)
            results['sd'] = np.nanstd(param_list, axis=0, ddof=1)
            results['p2.5'] = np.percentile(param_list, 2.5, axis=0)
            results['p50'] = np.percentile(param_list, 50, axis=0)
            results['p97.5'] = np.percentile(param_list, 97.5, axis=0)
        else:
            results['sd'] = np.nan
    if bs_return_verbose:
        return results, param_list
    else:
        return results, None


def fitting_master(seq, **kwargs):
    """
    Master fitting function to convert a iteration from pd.DataFrame.iterrows() to input type for fitting_single
    :param seq: one item from pd.DataFrame.iterrows()
    :param kwargs: all other keyword arguments need to pass to fitting_single()
    :return: return pd.Series object containing the fitting results
    """

    single_res = fitting_single(y_data=list(seq[1]), **kwargs)
    return pd.Series(single_res[0], name=seq[0]), (seq[0], single_res[1])


def fitting_sequence_set(sequence_set, bs_return_verbose=True, parallel_threads=None, inplace=True, **kwargs):
    """
    Method to apply fitting on all sequences in sequence_set
    :param sequence_set:
    :param bs_return_verbose:
    :param parallel_threads:
    :param inplace:
    :param kwargs:
    :return:
    """
    from functools import partial

    partial_func = partial(fitting_master,
                           x_data=sequence_set.reacted_frac_table.col_x_values,
                           bs_return_verbose=bs_return_verbose,
                           **kwargs)

    if parallel_threads:
        pool = mp.Pool(processes=int(parallel_threads))
        results = pool.map(partial_func, sequence_set.reacted_frac_table.iterrows())
    else:
        results = [partial_fun(seq) for seq in sequence_set.reacted_frac_table.iterrows()]

    if inplace:
        sequence_set.fitting_results = pd.DataFrame([res[0] for res in results])
        if bs_return_verbose:
            sequence_set.bs_log = {res[1][0]:res[1][1] for res in results}
    else:
        if bs_return_verbose:
            return pd.DataFrame([res[0] for res in results]), {res[1][0]:res[1][1] for res in results}
        else:
            return pd.DataFrame([res[0] for res in results])
