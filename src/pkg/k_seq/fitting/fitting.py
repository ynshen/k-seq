"""
Methods needed for fitting
"""

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import multiprocessing as mp
from ..data import pre_processing
from ..utility import get_args_params


def byo_model(x, A, k):
    """
    Default kinetic model used in BYO k-seq fitting:
                    A * (1 - np.exp(- 0.479 * 90 * k * x))
    - 90: t, reaction time (min)
    - 0.479: alpha, degradation adjustment parameter for BYO in 90 min
    - k: kinetic coefficient
    - A: maximal conversion the self-aminoacylation ribozyme

    Args:
        x (`float`): predictor, concentration of BYO for each sample, needs have unit mol
        A (`float`)
        k (`float`)

    Returns:
        reacted fraction given the predictor x and parameter (A, k)
    """
    return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BYO degradation adjustment and 90 minutes


class _PointEstimation:

    def __init__(self, single_fitting):
        from scipy.optimize import curve_fit

        try:
            if single_fitting.config['random_init']:
                init_guess = [np.random.random() for _ in single_fitting.config['parameters']]
                params, pcov = curve_fit(single_fitting.model,
                                         xdata=single_fitting.x_data, ydata=single_fitting.y_data,
                                         sigma=single_fitting.weights, method='trf',
                                         bounds=single_fitting.config['bounds'], p0=init_guess)
            else:
                params, pcov = curve_fit(single_fitting.model,
                                         xdata=single_fitting.x_data, ydata=single_fitting.y_data,
                                         sigma=single_fitting.weights, method='trf',
                                         bounds=single_fitting.config['bounds'])
            self.params = pd.Series(data=params, index=get_args_params(single_fitting.model))
            self.pcov = pcov

        except RuntimeError:
            self.params = np.nan
            self.pcov = np.nan

        if single_fitting.metrics is not None:
            for name, fn in single_fitting.metrics.items():
                self.params[name] = self.params.apply(fn, axis=1)


class _Bootstrap:

    @staticmethod
    def bs_sample_generator(single_fitting):
        import numpy as np

        if single_fitting.config['bs_method'] == 'Resample percent residue':
            y_hat = single_fitting.model(single_fitting.x_data, *single_fitting.point_est.params.values())
            pct_res = (single_fitting.y_data - y_hat) / y_hat
            for _ in range(single_fitting.config['bs_depth']):
                pct_res_resampled = np.random.choice(pct_res, replace=True, size=len(pct_res))
                yield single_fitting.x_data, y_hat * (1 + pct_res_resampled)
        else:
            indices = np.linspace(0, len(single_fitting.x_data) - 1, len(single_fitting.x_data), dtype=np.int)
            for _ in range(single_fitting.config['bs_depth']):
                bs_indeces = np.random.choice(a=indices, size=len(single_fitting.x_data), replace=True)
                yield single_fitting.x_data[bs_indeces], single_fitting.y_data[bs_indeces]

    def __init__(self, single_fitting):
        from scipy.optimize import curve_fit

        param_list = pd.DataFrame(index=np.linspace(0, len(single_fitting.config['bs_depth']),
                                                    len(single_fitting.config['bs_depth']) + 1, dtype=np.int),
                                  columns=single_fitting.config['parameters'])

        for ix, (x_data, y_data) in enumerate(bs_sample_generator(single_fitting)):
            try:
                if single_fitting.config['random_init']:
                    init_guess = [np.random.random() for _ in single_fitting.config['parameters']]
                    params, _ = curve_fit(single_fitting.model,
                                             xdata=x_data, ydata=y_data,
                                             method='trf', bounds=single_fitting.config['bounds'], p0=init_guess)
                else:
                    params, _ = curve_fit(single_fitting.model,
                                             xdata=x_data, ydata=y_data,
                                             method='trf', bounds=single_fitting.config['bounds'])
            except:
                params = np.repeat(np.nan, len(config['parameters']))
            param_list.loc[ix] = params

        if single_fitting.metrics is not None:
            for name, fn in single_fitting.metrics.items():
                param_list[name] = param_list.apply(fn, axis=1)



                results['mean'] = np.nanmean(param_list, axis=0)
                results['sd'] = np.nanstd(param_list, axis=0, ddof=1)
                results['p2.5'] = np.nanpercentile(param_list, 2.5, axis=0)
                results['p50'] = np.nanpercentile(param_list, 50, axis=0)
                results['p97.5'] = np.nanpercentile(param_list, 97.5, axis=0)
            else:
                results['sd'] = np.nan


        if bs_return_verbose > 0:
            return results, param_list
        else:
            return results, None



class SingleFitting:

    def __init__(self, x_data, y_data, func, weights=None, bounds=None, bootstrap_depth=0, bs_return_size=None,
                 resample_pct_res=False, missing_data_as_zero=False, random_init=True, **kwargs):

        if missing_data_as_zero:
            y_data[np.isnan(y_data)] = 0
        parameters = get_args_params(func)
        if bounds is None:
            bounds = [[-np.inf for _ in func], [np.inf for _ in func]]
        if weights is None:
            weights = np.ones(len(y_data))
        # only include non np.nan data
        valid = ~np.isnan(y_data)
        self.model = func
        self.x_data = x_data[valid]
        self.y_data = y_data[valid]
        self.weights = weights[valid]
        self.config = {
            'parameters': get_args_params(func, exclude_x=True),
            'bounds': bounds,
            'missing_data_as_zero': missing_data_as_zero,
            'random_init': random_init
        }
        if bootstrap_depth > 0 and len(self.x_data) > 1:
            self.config['bootstrap'] = True
            self.config['bs_depth'] = bootstrap_depth
            if bs_return_size is None:
                self.config['bs_return_size'] = bootstrap_depth
            else:
                self.config['bs_return_size'] = bs_return_size
            if resample_pct_res:
                self.config['bs_method'] = 'Resample percent residues'
            else:
                self.config['bs_method'] = 'Resample data points'
        else:
            self.config['bootstrap'] = False

    def fitting(self):
        self.point_est = _PointEstimation(self)
        if self.config['bootstrap']:
            if np.isnan(self.point_est.params):
                self.config['bs_method'] = 'Resample data points'  # if the point estimation is not valid, can only resample data points
            self.bootstrap = _Bootstrap(self)
        else:
            self.bootstrap = np.nan

    @classmethod
    def from_raw_data(cls, x_data, y_data, func, weights=None, bounds=None, bootstrap_depth=0):
        pass


    @classmethod
    def from_SeqTable(cls,):
        pass






def fitting_single(x_data, y_data, func=byo_model, weights=None, bounds=None,
                   bootstrap=True, bs_depth=1000, bs_res_return_size=None, bs_residue=False,
                   missing_data_as_zero=False, y_max=None, random_init=True, **kwargs):

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




def fitting_master(seq, **kwargs):
    """
    Master fitting function to convert a iteration from pd.DataFrame.iterrows() to input type for fitting_single
    :param seq: one item from pd.DataFrame.iterrows()
    :param kwargs: all other keyword arguments need to pass to fitting_single()
    :return: return pd.Series object containing the fitting results
    """

    single_res = fitting_single(y_data=list(seq[1]), **kwargs)
    return pd.Series(single_res[0], name=seq[0]), (seq[0], single_res[1])


def fitting_sequence_set(sequence_set, bs_return_verbose=100, parallel_threads=None, inplace=True, **kwargs):
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

    if isinstance(sequence_set, pd.DataFrame):
        reacted_frac_table = sequence_set
    else:
        reacted_frac_table = sequence_set.reacted_frac_table

    if parallel_threads:
        pool = mp.Pool(processes=int(parallel_threads))
        results = pool.map(partial_func, sequence_set.reacted_frac_table.iterrows())
    else:
        results = [partial_func(seq) for seq in reacted_frac_table.iterrows()]

    if inplace:
        sequence_set.fitting_results = pd.DataFrame([res[0] for res in results])
        if bs_return_verbose:
            sequence_set.bs_log = {res[1][0]:res[1][1] for res in results}
    else:
        if bs_return_verbose:
            return pd.DataFrame([res[0] for res in results]), {res[1][0]:res[1][1] for res in results}
        else:
            return pd.DataFrame([res[0] for res in results])





# TODO: make a full script to run the whole