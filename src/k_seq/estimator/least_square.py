"""
This sub-module contains the classic fitting each sequence individually to the kinetic model,
  using absolute amount or reacted fraction

Several functions are included:
  - point estimation using `scipy.optimize.curve_fit`
  - option to exclude zero in fitting
  - option to initialize values
  - weighted fitting depends on the customized weights
  - confidence interval estimation using bootstrap

todo: creating all the single fitters for BYO-doped will cost 20 min along - time consuming
"""
from ..estimator import EstimatorType
from ..utility.func_tools import DocHelper
import logging

doc_helper = DocHelper(
    x_data=('list', 'list of x values in fitting'),
    y_data=('list, pd.Series', 'y values in fitting'),
    model=('callable', 'model to fit'),
    parameters=('list', 'Optional. List of parameter names, extracted from model if None'),
    name=('str', "Optional. Fitter's name"),
    sigma=('list, pd.Series, or pd.DataFrame', 'Optional, same size as y_data/y_data_batch.'
                                               'Sigma (variance) for data points for weighted fitting'),
    bounds=('2 by m `list` ', 'Optional, [[lower bounds], [higher bounds]] for each parameter'),
    opt_method=('`str`', "Optimization methods in `scipy.optimize`. Default 'trf'"),
    bootstrap_num=('`int`', 'Number of bootstrap to perform, 0 means no bootstrap'),
    bs_record_num=('`int`', 'Number of bootstrap results to store. Negative number means store all results.'
                            'Not recommended due to memory consumption'),
    bs_method=('`str`', "Bootstrap method, choose from 'pct_res' (resample percent residue),"
                        "'data' (resample data), or 'stratified' (resample within replicates)"),
    exclude_zero=('`bool`', "If exclude zero/missing data in fitting. Default False."),
    init_guess=('list of `float` or generator', "Initial guess estimate parameters, random value from 0 to 1 "
                                                "will be use if None"),
    metrics=('`dict` of `callable`', "Optional. Extra metric/parameters to calculate for each estimation"),
    rnd_seed=('`int`', "random seed used in fitting for reproducibility"),
    curve_fit_params=('dict', 'other keyword parameters to pass to `scipy.optimze.curve_fit`'),
    grouper=('dict or Grouper', 'Indicate the grouping of samples'),
    seq_to_fit=('list of str', 'Optional. List of sequences used in batch fitting')
)


class SingleFitter(EstimatorType):
    __doc__ = """A wrapper over `scipy.optimize.curve_fit` to fit a model for a single dataset
    Can do point estimation or bootstrap for empirical CI estimation

    Attributes:
        {attr}
        bootstrap (Bootstrap): proxy to the bootstrap object
        
        results (FitResult): proxy to the FirResult object
        
        config (AttrScope): name space for fitting, contains
        {config}
        
        bootstrap_config (AttrScope): name space for bootstrap, contains
        {bs_config}
            
    """.format(
        attr=doc_helper.get(['x_data', 'y_data', 'model', 'parameter', 'silent', 'name']),
        config=doc_helper.get(['opt_method', 'exclude_zero', 'init_guess', 'rnd_seed', 'sigma',
                               'bounds', 'metric', 'curve_fit_params'], indent=8),
        bs_config=doc_helper.get(['bootstrap_num', 'bs_record_num', 'bs_method'])
    )

    def __repr__(self):
        return f"Single fitter for {self.name}"\
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __str__(self):
        return f"Single fitter for {self.name}"

    def __init__(self, x_data, y_data, model, name=None, parameters=None, sigma=None, bounds=None, init_guess=None,
                 opt_method='trf', exclude_zero=False, metrics=None, rnd_seed=None, grouper=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', curve_fit_params=None, silent=False):
        """Initialize a `SingleFitter` instance
        
        Args:
            {}
        """.format(doc_helper.get(self.__init__))

        import numpy as np
        from ..utility.func_tools import AttrScope, get_func_params

        super().__init__()

        if len(x_data) != len(y_data):
            raise ValueError('Shapes of x and y do not match')

        self.model = model
        self.name = name
        self.parameters = get_func_params(model, exclude_x=True) if parameters is None else list(parameters)
        self.config = AttrScope(
            opt_method=opt_method,
            exclude_zero=exclude_zero,
            init_guess=init_guess,
            rnd_seed=rnd_seed,
            curve_fit_params={} if curve_fit_params is None else curve_fit_params
        )

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
        if sigma is None:
            self.config.sigma = np.ones(len(self.y_data))
        elif isinstance(sigma, list):
            self.config.sigma = np.array(sigma)[mask]
        else:
            self.config.sigma = sigma[mask]

        if bounds is None:
            self.config.bounds = (-np.inf, np.inf)
        else:
            self.config.bounds = bounds

        self.bootstrap_config = AttrScope(
            bootstrap_num=bootstrap_num,
            bs_record_num=bs_record_num,
            bs_method=bs_method
        )

        if bootstrap_num > 0 and len(self.x_data) > 1:
            if bs_record_num is None:
                bs_record_num = 0
            self.bootstrap = Bootstrap(fitter=self, bootstrap_num=bootstrap_num, bs_record_num=bs_record_num,
                                       bs_method=bs_method, grouper=grouper)
        else:
            self.bootstrap = None

        self.config.metrics = metrics
        self.results = FitResults(fitter=self)
        # TODO: check visualizers work
        from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot
        from ..utility.func_tools import FuncToMethod
        self.visualizer = FuncToMethod(obj=self, functions=[fitting_curve_plot,
                                                            bootstrap_params_dist_plot])
        self.silent = silent
        if not silent:
            logging.info(f"{self.__repr__()} initiated")

    def _fit(self, model=None, x_data=None, y_data=None, sigma=None, parameters=None, bounds=None,
             metrics=None, init_guess=None, opt_method=None, curve_fit_params=None):
        """Core function perform fitting"""
        from scipy.optimize import curve_fit
        import numpy as np

        if model is None:
            model = self.model
            parameters = self.parameters
        if parameters is None:
            from ..utility.func_tools import get_func_params
            parameters = get_func_params(model, exclude_x=True)
        if x_data is None:
            x_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if sigma is None:
            sigma = self.config.sigma
        if bounds is None:
            bounds = self.config.bounds
        if metrics is None:
            metrics = self.config.metrics
        if init_guess is None:
            init_guess = self.config.init_guess
        if opt_method is None:
            opt_method = self.config.opt_method
        if curve_fit_params is None:
            curve_fit_params = self.config.curve_fit_params

        try:
            if init_guess is None:
                # use a random guess form (0, 1)
                init_guess = [np.random.random() for _ in parameters]
            if curve_fit_params is None:
                curve_fit_params = {}
            params, pcov = curve_fit(f=model,
                                     xdata=x_data, ydata=y_data,
                                     sigma=sigma, method=opt_method,
                                     bounds=bounds, p0=init_guess, **curve_fit_params)
            if metrics is not None:
                metrics_res = {name: fn(params) for name, fn in metrics.items()}
            else:
                metrics_res = None
        except RuntimeError:
            logging.warning(
                f"RuntimeError on \n"
                f'\tx = {self.x_data}\n'
                f'\ty={self.y_data}\n'
                f'\tsigma={self.config.sigma}'
            )
            params = np.full(fill_value=np.nan, shape=len(parameters))
            pcov = np.full(fill_value=np.nan, shape=(len(parameters), len(parameters)))
            if metrics is not None:
                metrics_res = {name: np.nan for name, fn in metrics.items()}
            else:
                metrics_res = None
        except ValueError:
            logging.warning(
                f"ValueError on \n"
                f'\tx = {self.x_data}\n'
                f'\ty={self.y_data}\n'
                f'\tsigma={self.config.sigma}'
            )
            params = np.full(fill_value=np.nan, shape=len(parameters))
            pcov = np.full(fill_value=np.nan, shape=(len(parameters), len(parameters)))
            if metrics is not None:
                metrics_res = {name: np.nan for name, fn in metrics.items()}
            else:
                metrics_res = None
        except:
            logging.warning(
                f"Other error observed on\n"
                f'\tx = {self.x_data}\n'
                f'\ty={self.y_data}\n'
                f'\tsigma={self.config.sigma}'
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
        """Run fitting, configuration are from the object"""

        import numpy as np
        import pandas as pd

        if self.config.rnd_seed is not None:
            np.random.seed(self.config.rnd_seed)

        point_est = self._fit()
        params = {key: value for key, value in zip(self.parameters, point_est['params'])}
        if point_est['metrics'] is not None:
            params.update(point_est['metrics'])
        self.results.point_estimation.params = pd.Series(params)
        self.results.point_estimation.pcov = pd.DataFrame(data=point_est['pcov'],
                                                          index=self.parameters,
                                                          columns=self.parameters)
        if not self.silent:
            logging.info(f'Point estimation for {self.__repr__()} finished')
        if self.bootstrap is None:
            if not self.silent:
                logging.info('Bootstrap not conducted')
            pass
        else:
            if not self.silent:
                logging.info(f"Bootstrap using {self.bootstrap_config.bs_method} for "
                             f"{self.bootstrap_config.bootstrap_num} and "
                             f"save {self.bootstrap_config.bs_record_num} records")
            self.bootstrap.run()

    def summary(self):
        """Return a pd.series as fitting summary"""
        return self.results.to_series()

    def to_json(self, save_to_file=None, return_dict=False):
        """Save the configuration needed to re-initialization the object to JSON
        Notice: except for model as model object is usually not json-able
        """
        config_dict = {
            **{
                'x_data': self.x_data,
                'y_data': self.y_data,
                'model': self.model,
                'name': self.name,
                'parameters': self.parameters,
                'silent': self.silent,
            },
            **self.config.__dict__,
            **self.bootstrap_config.__dict__
        }
        if return_dict:
            return config_dict
        else:
            import json
            if save_to_file is None:
                return json.dumps(config_dict)
            else:
                from pathlib import Path
                path = Path(save_to_file)
                if path.suffix == '.json':
                    # its a named file
                    if not path.parent.exists():
                        path.parent.mkdir(parents=True)
                    with open(path, 'w') as handle:
                        json.dump(config_dict, handle)
                elif path.suffix == '':
                    # its a path
                    if not path.exists():
                        path.mkdir(parents=True)
                    with open(str(path) + '/config.json', 'w') as handle:
                        json.dump(config_dict, handle)
                else:
                    raise NameError('Unrecognized saving path')

    @classmethod
    def from_json(cls, file_path, model):
        """Load a fitter from saved json file
        Args:

            file_path (str): path to saved json file

            model (callable): as callable is not json-able, need to reassign
        """

        import json
        with open(file_path, 'r') as handle:
            config_dict = json.load(handle)
        return cls(model=model, **config_dict)

    # @classmethod
    # def from_table(cls, table, seq, model, x_data, sigma=None, bounds=None, bootstrap_num=0, bs_record_num=0,
    #                bs_method='pct_res', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None, **kwargs):
    #     """Get data from a row of `pd.DataFrame` table. `SeqTable` is not supported due to multiple tables contained"""
    #     import numpy as np
    #     import pandas as pd
    #
    #     if isinstance(x_data, (list, np.ndarray)):
    #         x_data = pd.Series(x_data, index=table.columns)
    #     elif isinstance(x_data, pd.Series):
    #         x_data = x_data[table.columns]
    #
    #     y_data = table.loc[seq]
    #     return cls(x_data=x_data, y_data=y_data, model=model, name=seq,
    #                weights=weights, bounds=bounds,
    #                bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method,
    #                exclude_zero=exclude_zero, init_guess=init_guess, metrics=metrics, rnd_seed=rnd_seed, **kwargs)

    # def load_results(self, json_o_path):
    #     self.results = FitResults(fitter=self)
    #     self.results.load_json(json_o_path)

    ######################### Not necessary in k-seq implementation ####################
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
    # todo: export a dictionary to pass results through multiprocesser
    """A class to store, format, fitting fitting results for single fitter

    Attributes:

         fitter (EstimatorType): proxy to the fitter

         point_estimation (AttrScope): a scope stores point estimation results, includes

             params (pd.Series): stores the parameter estimation, with extra metrics calculation

             pcov (pd.DataFrame): covariance matrix for estimated parameter

         uncertainty (AttrScope): a scope stores uncertainty estimation results, includes
             summary

             record

    """

    def __repr__(self):
        return f"Fitting results for {self.fitter} " \
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __init__(self, fitter):
        """
        Args:

            fitter (EstimatorType): fitter used to generate this fitting result

        """
        from ..utility.func_tools import AttrScope

        self.fitter = fitter
        self.point_estimation = AttrScope(keys=['params', 'pcov'])
        self.uncertainty = AttrScope(keys=['summary', 'record'])

    def to_series(self):
        """Convert `point_estimation.params` and `uncertainty.summary` to a series include flattened info"""
        import pandas as pd

        allowed_stats = ['mean', 'std', '2.5%', '50%', '97.5%']

        res = self.point_estimation.params.to_dict()
        if self.uncertainty.summary is not None:
            from ..utility.func_tools import dict_flatten
            res.update(dict_flatten(self.uncertainty.summary.loc[allowed_stats].to_dict()))

        if self.fitter.name is not None:
            return pd.Series(res, name=self.fitter.name)
        else:
            return pd.Series(res)

    def to_dict(self, include_pcov=False, include_record=True):
        """Pass results as dictionary"""
        results = {
            'point_estimation': {
                'params': self.point_estimation.params
            },
            'uncertainty': {
                'summary': self.uncertainty.summary
            }
        }
        if include_pcov:
            results['point_estimation']['pcov'] = self.point_estimation.pcov
        if include_record:
            results['uncertainty']['record'] = self.uncertainty.record
        return results

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
            if json_data['point_estimation']['params'] is not None:
                self.point_estimation.params = pd.read_json(json_data['point_estimation']['params'], typ='series')
            if json_data['point_estimation']['pcov'] is not None:
                self.point_estimation.pcov = pd.read_json(json_data['point_estimation']['pcov'])
        if 'uncertainty' in json_data.keys():
            if json_data['uncertainty']['summary'] is not None:
                self.uncertainty.summary = pd.read_json(json_data['uncertainty']['summary'])
            if json_data['uncertainty']['record'] is not None:
                self.uncertainty.record = pd.read_json(json_data['uncertainty']['record'])


class Bootstrap:
    """Class to perform bootstrap during fitting and store results to `FitResult`
    Three types of bootstrap supported:

      - `pct_res`: resample the percent residue (from data property)

      - `data`: resample data points

      - `stratified`: resample within group, `grouper` is needed

    Attributes:
    
        fitter (`EstimatorBase` type): fitter used for estimation
        
    {}
    """.format(doc_helper.get(['bs_method', 'bootstrap_num', 'bs_record_num', 'grouper']))

    def __repr__(self):
        return f"Bootstrap method using {self.bs_method} (n = {self.bootstrap_num})"

    def __init__(self, fitter, bootstrap_num, bs_record_num, bs_method, grouper=None):
        """
        Args:
            fitter (EstimatorType): the fitter generates the results
        {}
        """.format(doc_helper.get(['bootstrap_num', 'bs_record_num', 'bs_method', 'grouper']))

        implemented_methods = {
            'pct_res': 'pct_res',
            'resample percent residues': 'pct_res',
            'resample data points': 'data',
            'data': 'data',
            'stratified': 'stratified',
        }

        if bs_method in implemented_methods.keys():
            self.bs_method = bs_method
            if bs_method == 'stratified':
                try:
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
            raise NotImplementedError(f'Bootstrap method {bs_method} is not implemented')

        self.fitter = fitter
        self.bootstrap_num = bootstrap_num
        self.bs_record_num = bs_record_num

    def _percent_residue(self):
        """Bootstrap percent residue"""
        import numpy as np
        try:
            y_hat = self.fitter.model(
                self.fitter.x_data, *self.fitter.results.point_estimation.params[self.fitter.parameters].values
            )
        except AttributeError:
            # if could not find point estimation, do another fit
            params = self.fitter._fit()['params']
            y_hat = self.fitter.model(self.fitter.x_data, *params)

        pct_res = (self.fitter.y_data - y_hat) / y_hat
        for _ in range(self.bootstrap_num):
            pct_res_resample = np.random.choice(pct_res, size=len(pct_res), replace=True)
            yield self.fitter.x_data, y_hat * (1 + pct_res_resample)

    def _data(self):
        """Apply data based bootstrap"""
        import numpy as np
        indices = np.arange(len(self.fitter.x_data))
        for _ in range(self.bootstrap_num):
            indices_resample = np.random.choice(indices, size=len(indices), replace=True)
            yield self.fitter.x_data[indices_resample], self.fitter.y_data[indices_resample]

    def _stratified(self):
        """Apply stratified bootstrap, need grouper assigned
        x_data and y_data needs to be `Series` or the grouper key should be index
        {}
        """
        import numpy as np
        for _ in range(self.bootstrap_num):
            ix_resample = []
            for member_ix in self.grouper.values():
                ix_resample += list(np.random.choice(member_ix, size=len(member_ix), replace=True))
            yield self.fitter.x_data[ix_resample], self.fitter.y_data[ix_resample]

    def _bs_sample_generator(self):
        if self.bs_method == 'pct_res':
            return self._percent_residue()
        elif self.bs_method == 'data':
            return self._data()
        elif self.bs_method == 'stratified':
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
        if (self.bs_record_num < 0) or (self.bs_record_num >= self.bootstrap_num):
            self.fitter.results.uncertainty.record = results
        else:
            self.fitter.results.uncertainty.record = results.sample(n=self.bs_record_num, replace=False, axis=0)


class BatchFitResults:
    """Store, convert, and visualize BatchFitter results
    Only save results (detached from fitter), corresponding fitter should be found by sequence

    Attributes:
        fitter: proxy to the `BatchFitter`
        bs_record (dict of pd.DataFrame): {seq: `SingleFitter.results.uncertainty.record`}
        summary (`pd.DataFrame`): summarized results with each sequence as index

    Methods:
        summary_to_csv: export summary dataframe as csv file
        to_pickle: save results as pickled dictionary
        from_pickle: load bootstrapping results from picked dictionary


    """

    def __init__(self, fitter):
        """Parse results from fitter_list which the first is """
        self.fitter = fitter
        self.bs_record = None
        self.summary = None

    def summary_to_csv(self, path):
        """Save summary table as csv file"""
        self.summary.to_csv(path)

    def to_pickle(self, path=None):
        """Serialize results as a pickled `dict`"""

        import pickle
        results = {
            'bs_record': self.bs_record,
            'summary': self.summary
        }
        if path is None:
            return pickle.dumps(results)
        else:
            from ..utility.file_tools import dump_pickle
            dump_pickle(obj=results, path=path)

    @classmethod
    def from_pickle(cls, fitter, path):
        """Create a `BatchFitResults` instance with results loaded from pickle"""
        inst = cls(fitter=fitter)
        from ..utility.file_tools import read_pickle
        results = read_pickle(path=path)
        inst.bs_record = results['bs_record']
        inst.summary = results['summary']
        return inst

    def to_json(self, path=None):
        """Save fitting results as a JSON file
        Structure:

          - fitter: a string representation of the fitter
          - record: {seq: JSON of single fitter results}

        """
        import json
        data_to_json = {
            'bs_record': {seq: record.to_json() for seq, record in self.bs_record},
            'summary': self.summary.to_json()
        }

        if path is None:
            return json.dumps(data_to_json)
        else:
            json.dump(data_to_json, path)

    @classmethod
    def from_json(cls, fitter, json_o_path):
        """Load results from JSON"""

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
                raise TypeError("Invalid JSON input")
        inst = cls(fitter=fitter)
        inst.summary = pd.read_json(json_data['summary'])
        inst.bs_record = {key: pd.read_json(record) for key, record in json_data['bs_record']}

        return inst


class BatchFitter(EstimatorType):
    """Fitter for least squared batch fitting

    Attributes:
        y_data_batch (`pd.DataFrame`): a dataframe table containing y values to fit (e.g. reacted fraction in each sample)
    {attr}
        keep_single_fitters (bool): if each single fitter is saved, default False to save storage
        note (str): note about this fitting job
        fitters (None or dict): a dictionary of SingleFitters if `self.config.keep_single_fitters` is True
        seq_to_fit (list of str): list of seq to fit for this job
        results (BatchFitResult): fitting results
        fit_params (AttrScope): parameters pass to each fitting, should be same for each sequence, includes:
    {fit_params}
    """.format(attr=doc_helper.get(['model', 'x_data', 'parameters'], indent=4),
               fit_params=doc_helper.get(['x_data', 'model', 'parameters', 'bounds', 'init_guess', 'opt_method',
                                          'exclude_zero', 'metrics', 'rnd_seed', 'bootstrap_num', 'bs_record_num',
                                          'bs_method', 'curve_fit_params', 'silent']))

    def __repr__(self):
        return 'Least-squared BatchFitter at'\
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __str__(self):
        return 'Least squared BatchFitter'

    def __init__(self, y_data_batch, x_data, model, sigma=None, bounds=None, seq_to_fit=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', grouper=None,
                 opt_method='trf', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None,
                 curve_fit_params=None, keep_single_fitters=False, note=None, silent=False):
        """
        Args:
            y_data_batch (pd.DataFrame or str): a set of y_data to fit form rows of y_data_batch, can be a string
                indicate the path to a pickled pd.DataFrame record
        {args}
            keep_single_fitters (bool): if keep all single fitters in the object
            note (str): Optional notes for the fitter
            results: a proxy to BatchFitResults
        """.format(args=doc_helper.get(['x_data', 'model', 'bounds', 'sigma', 'bootstrap_num', 'bs_record_num',
                                        'bs_method', 'grouper', 'opt_method', 'exclude_zero', 'init_guess',
                                        'metrics', 'rnd_seed', 'curve_fit_params', 'seq_to_fit'], indent=4))

        from ..utility.func_tools import get_func_params, AttrScope
        import pandas as pd
        import numpy as np
        super().__init__()

        logging.info('Creating the BatchFitter...')

        self.model = model
        self.parameters = get_func_params(model, exclude_x=True)
        self.note = note,
        self.keep_single_fitters = keep_single_fitters
        self.fitters = None

        if isinstance(y_data_batch, str):
            from pathlib import Path
            table_path = Path(y_data_batch)
            if table_path.is_file():
                try:
                    y_data_batch = pd.read_pickle(table_path)
                except:
                    raise TypeError(f'{table_path} is not pickled `pd.DataFrame` object')
            else:
                raise FileNotFoundError(f'{table_path} is not a valid file')
        elif not isinstance(y_data_batch, pd.DataFrame):
            raise TypeError('Table should be a `pd.DataFrame`')
        else:
            pass

        self.y_data_batch = y_data_batch

        if seq_to_fit is not None:
            if isinstance(seq_to_fit, (list, np.ndarray, pd.Series)):
                self.seq_list = list(seq_to_fit)
            else:
                raise TypeError('Unknown seq_to_fit type, is it list-like?')
        self.seq_to_fit = seq_to_fit

        # prep fitting params once to save time for each single fitting
        if isinstance(x_data, pd.Series):
            self.x_data = x_data[y_data_batch.columns.values]
        elif len(x_data) != y_data_batch.shape[1]:
            raise ValueError('x_data length and table column number does not match')
        else:
            self.x_data = np.array(x_data)

        if sigma is not None:
            if np.shape(sigma) != np.shape(self.y_data_batch):
                raise ValueError('Shape of sigma does not match the shape of y_data_batch')
        self.sigma = sigma

        if bounds is None:
            bounds = (-np.inf, np.inf)

        if len(x_data) <= 1:
            logging.warning("Number of data points less than 2, no bootstrap will be performed")
            bootstrap_num = 0
        self.bootstrap = bootstrap_num > 0

        # contains parameters should pass to the single fitter
        self.fit_params = AttrScope(
            x_data=self.x_data,
            model=self.model,
            parameters=self.parameters,
            bounds=bounds,
            opt_method=opt_method,
            bootstrap_num=bootstrap_num,
            bs_record_num=bs_record_num,
            bs_method=bs_method,
            exclude_zero=exclude_zero,
            init_guess=init_guess,
            metrics=metrics,
            rnd_seed=rnd_seed,
            grouper=grouper if bs_method == 'stratified' else None,
            curve_fit_params=curve_fit_params,
            silent=True
        )

        self.results = BatchFitResults(fitter=self)

        # TODO: recover the visualizer
        # from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot, param_value_plot
        # from ..utility import FunctionWrapper
        # self.visualizer = FunctionWrapper(data=self,
        #                                   functions=[
        #                                       fitting_curve_plot,
        #                                       bootstrap_params_dist_plot,
        #                                       param_value_plot
        #                                   ])
        logging.info('BatchFitter created')

    def worker_generator(self):
        if self.seq_to_fit is None:
            seq_list = self.y_data_batch.index.values
        else:
            seq_list = self.seq_to_fit
        for seq in seq_list:
            try:
                yield SingleFitter(
                    name=seq,
                    y_data=self.y_data_batch.loc[seq],
                    sigma=None if self.sigma is None else self.sigma.loc[seq],
                    **self.fit_params.__dict__
                )
            except:
                raise Exception(f'Can not create fitting worker for {seq}')

    def fit(self, deduplicate=False, parallel_cores=1, dry_run=False):
        """Run the estimation
        Args:
            deduplicate (bool): hash the y_data_batch to deduplicate before fitting if True
            parallel_cores (int): number of parallel cores to use. Default 1
            dry_run (bool): only create the worker without fitting if True
        """
        import pandas as pd

        logging.info('Batch fitting starting...')

        if deduplicate:
            self._hash()

        if dry_run:
            workers = [worker for worker in self.worker_generator()]
        else:
            if parallel_cores > 1:
                import multiprocessing as mp
                pool = mp.Pool(processes=int(parallel_cores))
                workers = pool.map(_work_fn, self.worker_generator())
            else:
                # single thread
                workers = [_work_fn(fitter) for fitter in self.worker_generator()]

        if self.keep_single_fitters:
            self.fitters = {worker.name: worker for worker in workers}

        if self.bootstrap:
            self.results.bs_record = {worker.name: worker.results.uncertainty.record for worker in workers}
        self.results.summary = pd.DataFrame({worker.name: worker.summary() for worker in workers}).transpose()

        if deduplicate:
            self._hash_inv()
        logging.info('Fitting finished')

    # @classmethod
    # def from_SeqTable(cls, seq_table, model, seq_to_fit=None, weights=None, bounds=None, bootstrap_depth=0,
    #                   bs_return_size=None,
    #                   resample_pct_res=False, missing_data_as_zero=False, random_init=True, metrics=None, **kwargs):
    #     """I did not really use this """
    #     raise NotImplementedError('This method is not implemented yet')
    #     if seq_to_fit is None:
    #         seq_to_fit = seq_table.reacted_frac_table.index
    #     if weights is None:
    #         weights = None
    #     if bounds is None:
    #         bounds = None
    #     if bs_return_size is None:
    #         bs_return_size = None
    #     if metrics is None:
    #         metrics = None
    #     return cls(seq_to_fit = {seq: seq_table.reacted_frac_table.loc[seq] for seq in seq_to_fit},
    #                x_data=seq_table.x_data, model=model,
    #                weights=weights, bounds=bounds,
    #                bootstrap_depth=bootstrap_depth, bs_return_size=bs_return_size,
    #                resample_pct_res=resample_pct_res, missing_data_as_zero=missing_data_as_zero,
    #                random_init=random_init, metrics=metrics, **kwargs)

    def summary(self, save_to=None):
        if save_to is None:
            return self.results.summary
        else:
            self.results.summary.to_csv(save_to)

    def _hash(self):
        """De-duplicate rows before fitting"""
        import pandas as pd

        def hash_series(row):
            return hash(tuple(row))

        self._y_data_batch_dup = self.y_data_batch.copy()
        if self.seq_to_fit is not None:
            self._seq_to_fit_dup = self.seq_to_fit.copy()
            self.y_data_batch = self.y_data_batch.loc[self.seq_to_fit]
        self._seq_to_hash = self.y_data_batch.apply(hash_series, axis=1).to_dict()
        if isinstance(self.sigma, pd.DataFrame):
            self._sigma_dup = self.sigma.copy()
            self.sigma.rename(index=self._seq_to_hash, inplace=True)
        self.y_data_batch = self.y_data_batch[~self.y_data_batch.duplicated(keep='first')]
        self.y_data_batch.rename(index=self._seq_to_hash, inplace=True)
        if self.seq_to_fit is not None:
            self.seq_to_fit = [self._seq_to_hash[seq] for seq in self.seq_to_fit]
        logging.info('Shrink rows in table by removing duplicates: '
                     f'{self._y_data_batch_dup.shape[0]} --> {self.y_data_batch.shape[0]}')

    def _hash_inv(self):
        """Recover the hashed results"""
        import pandas as pd

        logging.info('Recovering original table from hash...')

        def get_summary(seq):
            return self.results.summary.loc[self._seq_to_hash[seq]]

        self.results.summary = pd.Series(data=list(self._seq_to_hash.keys()),
                                         index=list(self._seq_to_hash.keys())).apply(get_summary)
        if self.results.bs_record is not None:
            self.results.bs_record = {seq: self.results.bs_record[seq_hash]
                                      for seq, seq_hash in self._seq_to_hash.items()}
        self.y_data_batch = self._y_data_batch_dup.copy()
        del self._y_data_batch_dup
        if hasattr(self, '_sigma_dup'):
            self.sigma = self._sigma_dup.copy()
            del self._sigma_dup
        if hasattr(self, '_seq_to_fit'):
            self.seq_to_fit = self._seq_to_fit_dup.copy()
            del self._seq_to_fit_dup

    def save_model(self, model_path, result_path=None, table_path=None):
        """Save models to disk as pickled dictionary,
        except for y_data_batch, and sigma which is large data
        """
        from ..utility.file_tools import dump_pickle

        dump_pickle(
            obj={
                **{'parameters': self.parameters,
                   'keep_single_fitters': self.keep_single_fitters,
                   'note': self.note,
                   'seq_to_fit': self.seq_to_fit},
                **self.fit_params.__dict__
            },
            path=model_path
        )
        if result_path is not None:
            self.results.to_pickle(path=result_path)
        if table_path is not None:
            dump_pickle(obj=self.y_data_batch, path=table_path)

    def save_results(self, result_path):
        """Save results to disk as pickled dict"""
        self.results.to_pickle(result_path)

    @classmethod
    def load_model(cls, model_path, y_data_batch, sigma=None, result_path=None):
        """Create a model from picked file on disk

        Args:
            model_path (str): path to picked model configuration file
            y_data_batch (pd.DataFrame or str): y_data table for fitting
            sigma (pd.DataFrame or str): optional sigma table for fitting
            result_path (str): path to fitting results
        """

        from ..utility.file_tools import read_pickle

        model_config = read_pickle(model_path)
        if isinstance(y_data_batch, str):
            y_data_batch = read_pickle(y_data_batch)
        if sigma is not None:
            if isinstance(sigma, str):
                sigma = read_pickle(sigma)
        inst = cls(y_data_batch=y_data_batch, sigma=sigma, **model_config)
        if result_path is not None:
            results = read_pickle(result_path)
            inst.results.bs_record = results['bs_record']
            inst.results.summary = results['summary']
        return inst


def _work_fn(worker):
    worker.fit()
    return worker


