"""
This sub-module contains the classic fitting each sequence individually to the kinetic model,
  using absolute amount or reacted fraction

Several functions are included:
  - point estimation using `scipy.optimize.curve_fit`
  - option to exclude zero in fitting
  - option to initialize values
  - weighted fitting depends on the customized weights
  - confidence interval estimation using bootstrap

"""


class Bootstrap:
    """Class to perform bootstrap in fitting"""

    def __repr__(self):
        return f"Bootstrap method using {self.method} (n = {self.bootstrap_num})"

    def __init__(self, fitter, bootstrap_num, return_num, method, **kwargs):

        implemented_methods = {
            'pct_res': 'pct_res',
            'resample percent residues': 'pct_res',
            'resample data points': 'data',
            'data': 'data',
            'stratified': 'stratified',
        }
        if method in implemented_methods.keys():
            self.method = method
            if method == 'stratified':
                try:
                    grouper = kwargs['grouper']
                    from ..data.grouper import Group
                    if isinstance(grouper, Group):
                        grouper = grouper.group
                    if isinstance(grouper, dict):
                        self.grouper = grouper
                    else:
                        raise TypeError('Unsupported grouper type for stratified bootstrap')
                except KeyError:
                    raise Exception('Please indicate grouper when using stratified bootstrapping')
        else:
            raise NotImplementedError(f'Bootstrap method {method} is not implemented')

        self.fitter = fitter
        self.bootstrap_num = bootstrap_num
        self.return_num = return_num
        self.record = None

    def _percent_residue(self):
        import numpy as np
        try:
            y_hat = self.fitter.model(self.fitter.x_data, *self.fitter.results.point_estimation.params.values())
        except AttributeError:
            params = self.fitter._fit()['params']
            y_hat = self.fitter.model(self.fitter.x_data, *params)
        pct_res = (self.fitter.y_data - y_hat) / y_hat
        for _ in range(self.bootstrap_num):
            pct_res_resample = np.random.choice(pct_res, size=len(pct_res), replace=True)
            yield self.fitter.x_data, y_hat * (1 + pct_res_resample)

    def _data(self):
        import numpy as np
        indices = np.arange(len(self.fitter.x_data))
        for _ in range(self.bootstrap_num):
            indices_resample = np.random.choice(indices, size=len(indices), replace=True)
            yield self.fitter.x_data[indices_resample], self.fitter.y_data[indices_resample]

    def _stratified(self):
        import numpy as np
        for _ in range(self.bootstrap_num):
            ix_resample = []
            for member_ix in self.grouper.values():
                ix_resample += np.random.choice(member_ix, size=len(member_ix), replace=True)
            yield self.fitter.x_data[ix_resample], self.fitter.y_data[ix_resample]

    def _bs_sample_generator(self):
        if self.method == 'pct_res':
            return self._percent_residue()
        elif self.method == 'data':
            return self._data()
        elif self.method == 'stratified':
            return self._stratified()
        else:
            return None

    def run(self):
        """Perform bootstrap"""
        import numpy as np
        import pandas as pd

        bs_sample_gen = self._bs_sample_generator()
        ix_list = pd.Series(np.arange(self.bootstrap_num))

        def fitting_runner(_):
            x_data, y_data = next(bs_sample_gen)
            result = self.fitter._fit(x_data=x_data, y_data=y_data)
            res_series = pd.Series(data=result['params'], index=self.fitter.parameters)
            if result['metrics'] is not None:
                for key, value in results['metrics'].items():
                    res_series[key] = value
            res_series['x_data'] = x_data
            res_series['y_data'] = y_data
            return res_series

        results = ix_list.apply(fitting_runner)
        self.summary = results.describe(percentiles=[0.025, 0.5, 0.975], include='all')
        self.fitter.results.uncertainty = self
        if self.return_num == self.bootstrap_num:
            self.records = results
        else:
            self.records = results.sample(n=self.return_num, replace=False, axis=0)


class FitResults:

    def __init__(self):
        self.point_estimation = None
        self.uncertainty = None

    def to_series(self):
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


# noinspection PyUnresolvedReferences
class SingleFitter:
    """Class to fit a single kinetic curve"""

    def __init__(self, x_data, y_data, model, name=None, weights=None, bounds=None, opt_method='trf',
                 bootstrap_num=0, bs_return_num=None, bs_method='pct_res',
                 exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None, **kwargs):
        """

        Args:
            x_data (list, np.ndarray, pd.Series): list object will convert to np.ndarray
            y_data (list, np.ndarray, pd.Series):
            model:
            name:
            weights:
            bounds:
            opt_method:
            bootstrap_num:
            bs_return_num:
            bs_method:
            exclude_zero:
            init_guess:
            metrics:
            rnd_seed:
            **kwargs:
        """
        import numpy as np
        from ..utility.func_tools import DictToAttr, get_func_params

        if len(x_data) != len(y_data):
            raise ValueError('Shapes of x and y do not match')

        self.model = model
        if name is not None:
            self.name = name
        self.parameters = get_func_params(model, exclude_x=True),
        self.config = DictToAttr({
            'opt_method': opt_method,
            'exclude_zero': exclude_zero,
            'init_guess': init_guess,
            'rnd_seed': rnd_seed
        })

        if isinstance(x_data, list):
            x_data = np.array(x_data)
        if isinstance(y_data, list):
            y_data = np.array(y_data)
        if exclude_zero is True:
            mask = y_data != 0
        else:
            mask = np.repeat(True, x_data.shape[0])

        self.x_data = x_data[mask]
        self.y_data = y_data[mask]
        if weights is None:
            weights = np.ones(len(y_data))
        self.weights = weights[mask]

        if bounds is None:
            self.config.bounds = [np.repeat(-np.inf, len(self.parameters)),
                                  np.repeat(np.inf, len(self.parameters))]
        else:
            self.config.bounds = bounds

        if bootstrap_num > 0 and len(self.x_data) > 1:
            if bs_return_num is None:
                bs_return_num = 0
            elif bs_return_num < 0:
                bs_return_num = bootstrap_num
            else:
                bs_return_num = bs_return_num
            self.config.bootstrap = _Bootstrap(fitter=self, bootstrap_num=bootstrap_num, return_num=bs_return_num,
                                               method=bs_method, **kwargs)
        else:
            self.config.bootstrap = None

        self.metrics = metrics
        self.results = None
        from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot
        from ..utility.func_tools import FuncToMethod
        self.visualizer = FuncToMethod(obj=self, functions=[fitting_curve_plot,
                                                            bootstrap_params_dist_plot])

    def _fit(self, model=None, x_data=None, y_data=None, weights=None, parameters=None, bounds=None,
             metrics=None, init_guess=None, opt_method=None):
        """perform fitting"""
        from scipy.optimize import curve_fit
        import numpy as np
        import pandas as pd
        import warnings

        if model is None:
            model = self.models
            parameters = self.parameters
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if weights is None:
            weights = self.weights
        if parameters is None:
            from ..utility.func_tools import get_func_params
            parameters = get_func_params(model, exclude_x=True)
        if bounds is None:
            bounds = self.bounds
        if metrics is None:
            metrics = self.metrics
        if init_guess is None:
            init_guess = self.config.init_guess
        if opt_method is None:
            opt_method = self.config.opt_method

        try:
            if init_guess is None:
                # use a random guess
                init_guess = [np.random.random() for _ in parameters]
            params, pcov = curve_fit(f=model,
                                     xdata=x_data, ydata=y_data,
                                     sigma=weights, method=opt_method,
                                     bounds=bounds, p0=init_guess)
            if metrics is not None:
                metrics = {name: fn(params) for name, fn in metrics.items()}
            return {
                'params': params,
                'pcov': pcov,
                'metrics': metrics
            }
        except RuntimeError:
            warnings.warn(
                f"RuntimeError for fitting model {self.model} on {self.name if self.name is not None else '*'} when\n"
                f'\tx = {self.x_data}\n'
                f'\ty={self.y_data}\n'
            )
            return {
                'params': None,
                'pcov': None,
                'metrics': None
            }

    def fit(self):
        """Wrapper on _fit method"""

        import numpy as np
        import pandas as pd

        point_est = self._fit()
        if self.bootstrap is None:
            pass
        else:
            # do bootstrap
            self.bootstrap.run()

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

    @classmethod
    def from_files(cls, model, x_col_name, y_col_name, path_to_file=None, path_to_x=None, path_to_y=None,
                   name=None, weights=None, bounds=None,
                   bootstrap_depth=0, bs_return_size=None, resample_pct_res=False, missing_data_as_zero=False,
                   random_init=True, metrics=None, **kwargs):
        """Fit data directly from files"""
        from ..data.io import read_table_files

        if path_to_x is not None and path_to_y is not None:
            x_data = read_table_files(file_path=path_to_x, col_name=x_col_name)
            y_data = read_table_files(file_path=path_to_y, col_name=y_col_name)
        elif path_to_file is not None:
            x_data = read_table_files(file_path=path_to_file, col_name=x_col_name)
            y_data = read_table_files(file_path=path_to_file, col_name=y_col_name)

        return cls(x_data=x_data, y_data=y_data, name=name, model=model, weights=weights, bounds=bounds,
                   bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size, resample_pct_res=resample_pct_res,
                   missing_data_as_zero=missing_data_as_zero, random_init=random_init, metrics=metrics)


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

        from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot, param_value_plot
        from ..utility import FunctionWrapper
        self.visualizer = FunctionWrapper(data=self,
                                          functions=[
                                              fitting_curve_plot,
                                              bootstrap_params_dist_plot,
                                              param_value_plot
                                          ])

    def fitting(self, parallel_cores=1):
        if parallel_cores > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=int(parallel_cores))
            self.seq_list = pool.map(_work_fn, self.seq_list)
        else:
            for seq_fitting in self.seq_list:
                seq_fitting.fitting()
        self.seq_list = {seq_fitting.name: seq_fitting for seq_fitting in self.seq_list}

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
                   x_values=seq_table.x_values, model=model,
                   weights=weights, bounds=bounds,
                   bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
                   resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
                   random_init=random_init, metrics=metrics, **kwargs)

    @property
    def summary(self):
        import pandas as pd
        series = [fitter.summary for fitter in self.seq_list.values()]
        return pd.concat(series, axis=1).transpose()


def _work_fn(fitter):
    fitter.fitting()
    return fitter
