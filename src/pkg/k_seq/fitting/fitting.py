"""
Methods needed for fitting
"""


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
    import numpy as np

    return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BYO degradation adjustment and 90 minutes


class _PointEstimation:

    def __init__(self, single_fitting):
        from scipy.optimize import curve_fit
        import numpy as np
        import pandas as pd

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
            self.params = pd.Series(data=params, index=single_fitting.config['parameters'])
            self.pcov = pcov

            if hasattr(single_fitting, 'metrics'):
                for name, fn in single_fitting.metrics.items():
                    self.params[name] = fn(self.params)
        except RuntimeError:
            self.params = np.nan
            self.pcov = np.nan


def _bs_sample_generator(single_fitting):
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


class _Bootstrap:

    def __init__(self, single_fitting):
        from scipy.optimize import curve_fit
        import numpy as np
        import pandas as pd

        param_list = pd.DataFrame(index=np.linspace(0, single_fitting.config['bs_depth'] - 1,
                                                    single_fitting.config['bs_depth'], dtype=np.int),
                                  columns=single_fitting.config['parameters'],
                                  dtype=np.float64)

        for ix, (x_data, y_data) in enumerate(_bs_sample_generator(single_fitting)):
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
                params = np.repeat(np.nan, len(single_fitting.config['parameters']))
            param_list.loc[ix] = params

        if hasattr(single_fitting, 'metrics'):
            for name, fn in single_fitting.metrics.items():
                param_list[name] = param_list.apply(fn, axis=1)

        self.summary = param_list.describe(percentiles=[0.025, 0.5, 0.975], include='all')
        if single_fitting.config['bs_return_size'] == 0:
            self.records = None
        elif single_fitting.config['bs_return_size'] < single_fitting.config['bs_depth']:
            self.records = param_list.sample(n=single_fitting.config['bs_return_size'], replace=False, axis=0)
        else:
            self.records = param_list


class SingleFitting:

    def __init__(self, x_data, y_data, model, name=None, weights=None, bounds=None, bootstrap_depth=0, bs_return_size=None,
                 resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None, **kwargs):
        import numpy as np
        import pandas as pd
        from ..utility import get_args_params

        if len(x_data) != len(y_data):
            raise Exception('Error: sizes of x and y data do not match')
        if name is not None:
            self.name = name
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        if missing_data_as_zero:
            y_data[np.isnan(y_data)] = 0

        # only include non np.nan data
        valid = ~np.isnan(y_data)
        self.model = model
        self.x_data = x_data[valid]
        self.y_data = y_data[valid]
        if weights is None:
            weights = np.ones(len(y_data))
        self.weights = weights[valid]
        self.config = {
            'parameters': get_args_params(model, exclude_x=True),
            'missing_data_as_zero': missing_data_as_zero,
            'random_init': random_init
        }
        if bounds is None:
            self.config['bounds'] = [[-np.inf for _ in self.config['parameters']],
                                     [np.inf for _ in self.config['parameters']]]
        else:
            self.config['bounds'] = bounds
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
        if metrics is not None:
            self.metrics = metrics

    def fitting(self):
        import numpy as np
        import pandas as pd

        self.point_est = _PointEstimation(self)
        if self.config['bootstrap']:
            if isinstance(self.point_est.params, pd.Series):
                self.config['bs_method'] = 'Resample data points'  # if the point estimation is not valid, can only resample data points
            self.bootstrap = _Bootstrap(self)
        else:
            self.bootstrap = np.nan

    @classmethod
    def from_SeqTable(cls, seq_table, seq, model, weights=None, bounds=None, bootstrap_depth=0, bs_return_size=None,
                      resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None, **kwargs):
        import numpy as np

        x_data = np.array(seq_table.x_values(with_col_name=False))
        y_data = np.array(seq_table.reacted_frac_table.loc[seq])
        return cls(x_data=x_data, y_data=y_data, name=seq, model=model,
                   weights=weights, bounds=bounds,
                   bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size, resample_pct_res=resample_pct_res,
                   missing_data_as_zero=missing_data_as_zero, random_init=random_init, metrics=metrics)
    @property
    def summary(self):
        import numpy as np
        import pandas as pd

        stats = self.config['parameters']
        if hasattr(self, 'metrics'):
            stats += tuple(self.metrics.keys())
        res = {}
        if self.point_est is not np.nan:
            res.update({stat + '_point_est': self.point_est.params[stat] for stat in stats})
        if self.config['bootstrap']:
            for stat in stats:
                res.update({
                    stat + '_mean': self.bootstrap.summary[stat]['mean'],
                    stat + '_std': self.bootstrap.summary[stat]['std'],
                    stat + '_2.5': self.bootstrap.summary[stat]['2.5%'],
                    stat + '_median': self.bootstrap.summary[stat]['50%'],
                    stat + '_97.5': self.bootstrap.summary[stat]['97.5%'],
            })
        if hasattr(self, 'name'):
            return pd.Series(data=list(res.values()), index=list(res.keys()), name=self.name)
        else:
            return pd.Series(data=list(res.values()), index=list(res.keys()))


class BatchFitting:

    def __init__(self, seq_to_fit, x_values, model, weights=None, bounds=None, bootstrap_depth=0, bs_return_size=None,
                 resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None, **kwargs):
        from ..utility import get_args_params
        import pandas as pd
        import numpy as np
        from datetime import datetime

        self.model = model
        self.config = {
            'parameters': get_args_params(model, exclude_x=True),
            'missing_data_as_zero': missing_data_as_zero,
            'random_init': random_init,
            'fitting_set_create_time': datetime.now()
        }
        if weights is not None:
            self.config['weights'] = weights
        else:
            weights = None
        if bounds is not None:
            self.config['bounds'] = bounds
        else:
            bounds = None

        if bootstrap_depth > 0:
            self.config['bootstrap'] = True
            self.config['bs_depth'] = bootstrap_depth
            if bs_return_size is None:
                self.config['bs_return_size'] = bootstrap_depth
            elif bs_return_size > bootstrap_depth:
                self.config['bs_return_size'] = bootstrap_depth
            else:
                self.config['bs_return_size'] = bs_return_size
            if resample_pct_res:
                self.config['bs_method'] = 'Resample percent residues'
            else:
                self.config['bs_method'] = 'Resample data points'
        else:
            self.config['bootstrap'] = False
        if bs_return_size is None:
            bs_return_size = None
        if metrics is not None:
            self.metrics = metrics
        else:
            metrics = None

        if isinstance(seq_to_fit, list) or isinstance(seq_to_fit, np.ndarray):
            if len(np.array(x_values).shape) == 1:
                self.seq_list = [
                    SingleFitting(x_data=x_values, y_data=y_values, model=model, name=ix,
                                  weights=weights, bounds=bounds,
                                  bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                                  resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                                  random_init=random_init, metrics=metrics)
                    for ix, y_values in enumerate(seq_to_fit)
            ]
            else:
                self.seq_list = [
                    SingleFitting(x_data=x_values[ix], y_data=y_values, model=model, name=ix,
                                  weights=weights, bounds=bounds,
                                  bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                                  resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                                  random_init=random_init, metrics=metrics)
                    for ix, y_values in enumerate(seq_to_fit)
                ]
        elif isinstance(seq_to_fit, dict):
            if len(np.array(x_values).shape) == 1:
                self.seq_list = [
                    SingleFitting(x_data=x_values, y_data=y_values, model=model, name=id,
                                  weights=weights, bounds=bounds,
                                  bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                                  resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                                  random_init=random_init, metrics=metrics)
                    for id, y_values in seq_to_fit.items()
            ]
            else:
                self.seq_list = [
                    SingleFitting(x_data=x_values[ix], y_data=y_values, model=model, name=id,
                                  weights=weights, bounds=bounds,
                                  bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                                  resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                                  random_init=random_init, metrics=metrics)
                    for ix, (id, y_values) in enumerate(seq_to_fit.items())
                ]

    def fitting(self, parallel_cores=1):
        if parallel_cores > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=int(parallel_cores))
            work_fn = lambda seq_fitting: seq_fitting.fitting()
            pool.map(work_fn, self.seq_list)
        else:
            for seq_fitting in self.seq_list:
                seq_fitting.fitting()


    @classmethod
    def from_SeqTable(cls, seq_table, model, seq_to_fit=None, weights=None, bounds=None, bootstrap_depth=0, bs_return_size=None,
                      resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None, **kwargs):
        if seq_to_fit is None:
            seq_to_fit = seq_table.reacted_frac_table.index
        if weights is None:
            weights = None
        if bounds is None:
            bounds = None
        if bs_return_size is None:
            bs_return_size = None
        if metrics is None:
            metrics = None
        return cls(seq_to_fit = {seq: seq_table.reacted_frac_table.loc[seq] for seq in seq_to_fit},
                   x_values=seq_table.x_values(with_col_name=False), model=model,
                   weights=weights, bounds=bounds,
                   bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                   resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                   random_init=random_init, metrics=metrics, **kwargs)



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