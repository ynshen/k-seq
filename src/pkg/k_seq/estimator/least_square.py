"""
This sub-module contains the classic fitting each sequence individually to the kinetic model,
  using absolute amount or reacted fraction

Several functions are included:
  - point estimation using `scipy.optimize.curve_fit`
  - option to exclude zero in fitting
  - option to initialize values
  - weighted fitting depends on the customized weights
  - confidence interval estimation using bootstrap

todo: fitting seems slower, read again to improve the performance
"""

from . import EstimatorType


# noinspection PyUnresolvedReferences
class SingleFitter(EstimatorType):
    """Class to fit a single kinetic model for one sequence

    Attributes:

    """

    def __repr__(self):
        return f"Single fitter for {self.name}"\
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __str__(self):
        return f"Single fitter for {self.name}"

    def __init__(self, x_data, y_data, model, name=None, parameters=None, weights=None, bounds=None, opt_method='trf',
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
        from ..utility.func_tools import AttrScope, get_func_params

        super().__init__()

        if len(x_data) != len(y_data):
            raise ValueError('Shapes of x and y do not match')

        self.model = model
        self.name = name
        if parameters is None:
            self.parameters = get_func_params(model, exclude_x=True)
        else:
            self.parameters = list(parameters)
        self.config = AttrScope({
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
            self.bootstrap = Bootstrap(fitter=self, bootstrap_num=bootstrap_num, return_num=bs_return_num,
                                               method=bs_method, **kwargs)
        else:
            self.bootstrap = None

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
            model = self.model
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
            bounds = self.config.bounds
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
                metrics_res = {name: fn(params) for name, fn in metrics.items()}
            else:
                metrics_res = None
        except RuntimeError:
            warnings.warn(
                f"RuntimeError for fitting model {self.model} on {self.name if self.name is not None else '*'} when\n"
                f'\tx = {self.x_data}\n'
                f'\ty={self.y_data}\n'
            )
            params = np.full(fill_value=np.nan, shape=len(parameters))
            pcov = np.full(fill_value=np.nan, shape=(len(parameters), len(parameters)))
            if metrics is not None:
                metrics_res = {name: np.nan for name, fn in metrics.items()}
            else:
                metrics_res = None
        return {
            'params': params,
            'pcov': pcov,
            'metrics': metrics_res
        }

    def fit(self):
        """Wrapper on _fit method"""

        import numpy as np
        import pandas as pd
        from ..utility.func_tools import AttrScope

        if self.config.rnd_seed is not None:
            np.random.seed(self.config.rnd_seed)

        self.results = FitResults(fitter=self)
        point_est = self._fit()
        params = {key: value for key, value in zip(self.parameters, point_est['params'])}
        if point_est['metrics'] is not None:
            params.update(point_est['metrics'])
        self.results.point_estimation.params = pd.Series(params)
        self.results.point_estimation.pcov = pd.DataFrame(data=point_est['pcov'],
                                                          index=self.parameters, columns=self.parameters)
        if self.bootstrap is None:
            pass
        else:
            self.bootstrap.run()

    def summary(self, stats_included=None):
        return self.results.to_series(stats_included=stats_included)

    @classmethod
    def from_table(cls, table, seq, model, x_data, weights=None, bounds=None, bootstrap_num=0, bs_return_num=None, bs_method='pct_res', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None, **kwargs):
        """Get data from a row of `pd.DataFrame` table. `SeqTable` is not supported due to multiple tables contained"""
        import numpy as np
        import pandas as pd

        if isinstance(x_data, (list, np.ndarray)):
            x_data = pd.Series(x_data, index=table.columns)
        elif isinstance(x_data, pd.Series):
            x_data = x_data[table.columns]

        y_data = table.loc[seq]
        return cls(x_data=x_data, y_data=y_data, model=model, name=seq,
                   weights=weights, bounds=bounds,
                   bootstrap_num=bootstrap_num, bs_return_num=bs_return_num, bs_method=bs_method,
                   exclude_zero=exclude_zero, init_guess=init_guess, metrics=metrics, rnd_seed=rnd_seed, **kwargs)

    def load_results(self, json_o_path):
        self.results = FitResults(fitter=self)
        self.results.read_json(json_o_path)

    # @classmethod
    # def from_files(cls, model, x_col_name, y_col_name, path_to_file=None, path_to_x=None, path_to_y=None,
    #                name=None, weights=None, bounds=None,
    #                bootstrap_depth=0, bs_return_size=None, resample_pct_res=False, missing_data_as_zero=False,
    #                random_init=True, metrics=None, **kwargs):
    #     """[Unimplemented] Fit data directly from files"""
    #     from ..data.io import read_table_files
    #
    #     if path_to_x is not None and path_to_y is not None:
    #         x_data = read_table_files(file_path=path_to_x, col_name=x_col_name)
    #         y_data = read_table_files(file_path=path_to_y, col_name=y_col_name)
    #     elif path_to_file is not None:
    #         x_data = read_table_files(file_path=path_to_file, col_name=x_col_name)
    #         y_data = read_table_files(file_path=path_to_file, col_name=y_col_name)
    #
    #     return cls(x_data=x_data, y_data=y_data, name=name, model=model, weights=weights, bounds=bounds,
    #                bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size, resample_pct_res=resample_pct_res,
    #                missing_data_as_zero=missing_data_as_zero, random_init=random_init, metrics=metrics)


class FitResults:
    """A class to store, format, and visualize fitting fitting results for single fitter

    Attributes:

         - fitter (`EstimatorBase` instance): fitter used to generate this fitting result

         - point_estimation (`AttrScope`): a scope stores point estimation results

         - uncertainty (`AttrScope`): a scope stores uncertainty estimation results

    """

    def __repr__(self):
        return f"Fitting results for {self.fitter} " \
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __init__(self, fitter):
        """
        Args:

            fitter (a `EstimatorBase`): fitter used to generate this fitting result

        """
        from ..utility.func_tools import AttrScope

        self.fitter = fitter
        self.point_estimation = AttrScope(keys=['params', 'pcov'])
        self.uncertainty = AttrScope(keys=['summary', 'record'])

    def to_series(self, stats_included=None):
        import pandas as pd

        allowed_stats = ['mean', 'std', '2.5%', '50%', '97.5%']
        if stats_included is None:
            stats_included = allowed_stats

        res = self.point_estimation.params.to_dict()
        if self.uncertainty is not None:
            from ..utility.func_tools import dict_flatten
            res.update(dict_flatten(self.uncertainty.summary.loc[stats_included].to_dict()))

        if self.fitter.name is not None:
            return pd.Series(res, name=self.fitter.name)
        else:
            return pd.Series(res)

    def to_json(self, path=None):
        """Convert results into a json file contains
            - fitter: str representation of fitter project
            - point_estimation
            - uncertainty
        """
        import json

        data_to_dump = {
            'fitter': str(self.fitter),
            'point_estimation': {
                'params': self.point_estimation.params.to_json(),
                'pcov': self.point_estimation.pcov.to_json()
            },
            'uncertainty': {
                'summary': self.uncertainty.summary.to_json(),
                'record': self.uncertainty.record.to_json()
            }
        }
        if path is None:
            return json.dumps(data_to_dump)
        else:
            with open(path, 'w') as handle:
                json.dump(data_to_dump, fp=handle)

    def load_json(self, json_o_path):
        """load fitting results from json"""

        import pandas as pd
        import json
        try:
            # first consider it is a json string
            json_data = json.loads(json_o_path)
        except json.JSONDecodeError:
            try:
                with open(json_o_path, 'r') as handle:
                    json_data = json.load(handle)
            except:
                raise TypeError(f'Can not parse json record for {self.__repr__()}')
        if 'point_estimation' in json_data.keys():
            self.point_estimation.params = pd.read_json(json_data['point_estimation']['params'])
            self.point_estimation.pcov = pd.read_json(json_data['point_estimation']['pcov'])
        if 'uncertainty' in json_data.keys():
            self.uncertainty.summary = pd.read_json(json_data['uncertainty']['summary'])
            self.uncertainty.record = pd.read_json(json_data['uncertainty']['record'])


class Bootstrap:
    """Class to perform bootstrap in fitting and store results to `FitResult`
    Three types of bootstrap supported:

      - `pct_res`: resample the percent residue (from data property)

      - `data`: resample data points

      - `stratified`: resample within group, grouper is needed

    Attributes:

        - fitter (`EstimatorBase` type): fitter used for estimation

        - method (`str`): a string indicate the bootstrap method

        - bootstrap_num (`int`): number of bootstraps

        - return_num (`int`): number of bootstraps to store

        - summary (describe of `pd.DataFrame`): a summary of estimated statistics of parameters
          a proxy to `self.fitter.results.uncertainty.summary`

        - record (`pd.DataFrame`): dataframe storing part of original fitting results
          a proxy to `self.fitter.results.uncertainty.record`

        - grouper (`dict` or `Grouper`): required if using `stratified` bootstrap, `Grouper` will be converted to `dict`
    """

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
        self.summary = None
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
                ix_resample += list(np.random.choice(member_ix, size=len(member_ix), replace=True))
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
                for key, value in result['metrics'].items():
                    res_series[key] = value
            res_series['x_data'] = x_data
            res_series['y_data'] = y_data
            return res_series

        results = ix_list.apply(fitting_runner)
        self.fitter.results.uncertainty.summary = results.describe(percentiles=[0.025, 0.5, 0.975], include=np.number)
        self.summary = self.fitter.results.uncertainty.summary
        if self.return_num == self.bootstrap_num:
            self.fitter.results.uncertainty.record = results
        else:
            self.fitter.results.uncertainty.record = results.sample(n=self.return_num, replace=False, axis=0)
        self.record = self.fitter.results.uncertainty.record

    def to_json(self, path=None):
        """Export bootstrap results as json format"""
        import json
        data_to_json = {
            'fitter': str(self.fitter),
            'bootstrap_num': self.bootstrap_num,
            'return_num': self.return_num,
            'grouper': self.grouper if hasattr(self, 'grouepr') else None,
            'method': self.method,
            'record': self.record.to_json(),
            'summary': self.summary.to_json()
        }
        if path is None:
            return json.dumps(data_to_json)
        else:
            with open(path, 'w') as handle:
                return json.dump(data_to_json, fp=path)

    def load_json(self, json_or_path):
        """Load results from json format
        NOTICE: we did not check fitter type and always assume the record matches this instance
        """
        import pandas as pd
        import json
        try:
            # first consider it is a json string
            json_data = json.loads(json_or_path)
        except json.JSONDecodeError:
            try:
                with open(json_or_path, 'r') as handle:
                    json_data = json.load(handle)
            except:
                raise TypeError(f'Can not parse json record for {self.__repr__()}')

        # assume fitter is always correct
        self.bootstrap_num = json_data.pop('bootstrap_num')
        self.return_num = json_data.pop('return_num')
        self.grouper = json_data.pop('grouper', None)
        self.method = json_data.pop('method')
        self.record = pd.read_json(json_data.pop('record'))
        self.summary = pd.read_json(json_data.pop('summary'))


class BatchFitResults:

    def __init__(self, fitter_list, master_fitter):
        self.fitter = master_fitter
        self.results = {fitter[0]: fitter[1] for fitter in fitter_list}
        self._summary = None

    def summary(self):
        if self._summary is None:
            import pandas as pd
            self._summary = pd.DataFrame.from_dict({seq: fitter.summary() for seq, fitter in self.results.items()}, orient='index')
        return self._summary

    def to_dataframe(self, seq_list=None):
        if seq_list is None:
            return self.summary()
        else:
            return self._summary.loc[seq_list]


class BatchFitter:

    def __init__(self, table, x_values, model, weights=None, bounds=None, seq_to_fit=None, bootstrap_num=0, bs_return_num=None, bs_method='pct_res',
                 opt_method='trf', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None, **kwargs):
        from ..utility.func_tools import get_func_params, AttrScope
        import pandas as pd
        import numpy as np

        self.model = model
        self.parameters = get_func_params(model, exclude_x=True)
        self.config = AttrScope({
            'seq_to_fit': seq_to_fit,
        })
        # prep fitting params once to save time for each single fitting
        self.x_values = x_values[table.columns.values]
        if weights is None:
            weights = np.ones(len(self.x_values))
        if bounds is None:
            bounds = [np.repeat(-np.inf, len(self.parameters)),
                      np.repeat(np.inf, len(self.parameters))]
        if len(x_values) <= 1:
            bootstrap_num = 0
        if bs_return_num is None:
            bs_return_num = 0
        elif bs_return_num < 0:
            bs_return_num = bootstrap_num
        else:
            pass
        if bs_return_num > 0:
            self.config.bootstrap = True

        # contains parameters should pass to the single fitter
        self.fit_params = AttrScope({
            'x_data': self.x_values,
            'model': self.model,
            'parameters': self.parameters,
            'weights': weights,
            'bounds': bounds,
            'opt_method': opt_method,
            'bootstrap_num': bootstrap_num,
            'bs_return_num': bs_return_num,
            'bs_method': bs_method,
            'exclude_zero': exclude_zero,
            'init_guess': init_guess,
            'metrics': metrics,
            'rnd_seed': rnd_seed
        })

        if isinstance(table, str):
            from pathlib import Path
            table_path = Path(table)
            if table_path.is_file():
                try:
                    table = pd.read_pickle(table_path)
                except:
                    raise TypeError(f'{table_path} is not pickled pd.DataFrame object')
            else:
                raise FileNotFoundError(f'{table_path} is not a valid file')
        elif not isinstance(table, pd.DataFrame):
            raise TypeError('Table should be a pd.DataFrame')
        else:
            pass

        self.table = table
        if seq_to_fit is None:
            self.seq_list = self.table.index.values
        else:
            if isinstance(seq_to_fit, (list, np.ndarray, pd.Series)):
                self.seq_list = list(seq_to_fit)
            else:
                raise TypeError('Unknown seq_to_fit type, is it list-like?')

        # from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot, param_value_plot
        # from ..utility import FunctionWrapper
        # self.visualizer = FunctionWrapper(data=self,
        #                                   functions=[
        #                                       fitting_curve_plot,
        #                                       bootstrap_params_dist_plot,
        #                                       param_value_plot
        #                                   ])

    def worker_generator(self):
        for seq in self.seq_list:
            try:
                yield SingleFitter.from_table(table=self.table, seq=seq, **self.fit_params.__dict__)
            except:
                raise Exception(f'Can not create fitting worker for {seq}')

    def fit(self, deduplicate=False, parallel_cores=1):
        if deduplicate:
            self._hash()
        if parallel_cores > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=int(parallel_cores))
            self.results = BatchFitResults(
                fitter_list=pool.map(_work_fn, self.worker_generator()),
                master_fitter=self
            )
        else:
            # single thread
            self.results = BatchFitResults(
                fitter_list=[_work_fn(fitter) for fitter in self.worker_generator()],
                master_fitter=self
            )
        if deduplicate:
            self._hash_inv()

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

    def summary(self, save_to=None):
        if save_to is None:
            return self.results.summary()
        else:
            self.results.summary().to_csv(save_to)

    def _hash(self):
        """De-duplicate rows to avoid unnecessary fitting"""

        def hash_series(row):
            return hash(tuple(row))

        self._table_dup = self.table.copy()
        self.table = self.table.loc[self.seq_list]
        print(f'Shrink rows in table by removing duplicates: {self.table.shape[0]} -->', end='')
        self._seq_to_hash = self.table.apply(hash_series, axis=1).to_dict()
        self.table = self.table[~self.table.duplicated(keep='first')]
        self.table.rename(index=self._seq_to_hash, inplace=True)
        self._seq_list_dup = self.seq_list.copy()
        self.seq_list = [self._seq_to_hash[seq] for seq in self.seq_list]
        print(self.table.shape[0])

    def _hash_inv(self):
        """Recover the hashed results"""
        import pandas as pd
        print('Recovering original table...')
        self.table = self._table_dup.copy()
        del self._table_dup
        self.seq_list = self._seq_list_dup.copy()
        del self._seq_list_dup
        self.results.results = {seq: self.results.results[seq_hash] for seq, seq_hash in self._seq_to_hash.items()}


def _work_fn(worker):
    worker.fit()
    return worker.name, worker
