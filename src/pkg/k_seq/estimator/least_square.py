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

    def __init__(self, fitter, bootstrap_num, return_num, method, **kwargs):
        from scipy.optimize import curve_fit
        import numpy as np
        import pandas as pd

        self.fitter = fitter
        self.bootstrap_num = bootstrap_num
        self.return_num = return_num
        implemented_methods_map = {
            'pct_res': 'pct_res',
            'resample percent residues': 'pct_res',
            'resample data points': 'data',
            'data': 'data',
            'stratified': 'stratified',
        }
        if method in implemented_methods_map.keys():
            self.method = method
        else:
            raise NotImplementedError(f'Bootstrap method {method} is not implemented')



    @staticmethod
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

    def run(self):
        """Perform bootstrap"""
        pass

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


class FitResults:

    pass
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

# noinspection PyUnresolvedReferences
class SingleFitter:
    """Class to fit a single kinetic curve"""

    def __init__(self, x_data, y_data, model, name=None, weights=None, bounds=None, opt_method='trf',
                 bootstrap_num=0, bs_return_num=None, bs_method='pct_res',
                 exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None, **kwargs):
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

        x_data = np.array(x_data)
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
            self.config.bootstrap = _Bootstrap(fitter=self, bootstrap_num=bootstrap_num,
                                               return_num=0 if bs_return_num is None else bs_return_num,
                                               method = bs_method)
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

    def fit(self, model=None, x_data=None, y_data=None, weights=None, parameters=None, bounds=None, metric=None, init_guess=None, method=None):
        """Wrapper on _fit method"""

        import numpy as np
        import pandas as pd

        self.results = point_est = self._fit(model, x_data, y_data, weights=None, parameters=None, bounds=None,
             metrics=None, init_guess=None, method='trf')
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
