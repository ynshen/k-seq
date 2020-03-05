"""Least-squares fitting for sequence individually with given kinetic model,

Several functions are included:
  - point estimation using `scipy.optimize.curve_fit`
  - option to exclude zero in fitting
  - option to initialize values
  - weighted fitting depends on the customized weights
  - confidence interval estimation using bootstrap
"""

from ..estimator import EstimatorType
from doc_helper import DocHelper
from ..utility.file_tools import read_json, dump_json, check_dir
from yutility import logging
import pandas as pd
import numpy as np

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
    curve_fit_kwargs=('dict', 'other keyword parameters to pass to `scipy.optimize.curve_fit`'),
    grouper=('dict or Grouper', 'Indicate the grouping of samples'),
    seq_to_fit=('list of str', 'Optional. List of sequences used in batch fitting')
)


@doc_helper.compose("""Use `scipy.optimize.curve_fit` to fit a model for a sequence time/concentration series
It can conduct point estimation, uncertainty estimation for bootstrap, empirical CI estimation

Attributes:
    <<x_data, y_data, model, parameter, silent, name>>
    bootstrap (Bootstrap): proxy to the bootstrap object
    results (FitResult): proxy to the FitResult object
    config (AttrScope): name space for fitting, contains
    <<opt_method, exclude_zero, init_guess, rnd_seed, sigma, bounds, metric, curve_fit_kwargs>>
    bootstrap_config (AttrScope): name space for bootstrap, contains
    <<bootstrap_num, bs_record_num, bs_method>>""")
class SingleFitter(EstimatorType):

    def __repr__(self):
        return f"Single fitter for {self.name}"\
               f"<{self.__class__.__module__}{self.__class__.__name__} at {hex(id(self))}>"

    def __init__(self, x_data, y_data, model, name=None, parameters=None, sigma=None, bounds=None, init_guess=None,
                 opt_method='trf', exclude_zero=False, metrics=None, rnd_seed=None, grouper=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', curve_fit_kwargs=None,
                 conv_reps=0, init_range=None,
                 save_to=None, overwrite=False, silent=False):
        """Initialize a `SingleFitter` instance
        
        Args:
            {}
            save_to (str): save results to the given path when fitting finishes, if not None
            overwrite (bool): if overwrite the results directory if already exists. Estimation will be skipped if False,
              default False
            silent (bool):
        """.format(doc_helper.get(self.__init__))

        from ..utility.func_tools import AttrScope, get_func_params

        super().__init__()

        if len(x_data) != len(y_data):
            raise ValueError('Shapes of x and y do not match')

        self.model = model
        self.name = name
        self.parameters = list(get_func_params(model, exclude_x=True)) if parameters is None else list(parameters)
        self.config = AttrScope(
            opt_method=opt_method,
            exclude_zero=exclude_zero,
            init_guess=init_guess,
            rnd_seed=rnd_seed,
            curve_fit_kwargs={} if curve_fit_kwargs is None else curve_fit_kwargs
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

        if bootstrap_num > 0 and len(self.x_data) > 1:
            if bs_record_num is None:
                bs_record_num = 0
            self.bootstrap = Bootstrap(fitter=self, bootstrap_num=bootstrap_num, bs_record_num=bs_record_num,
                                       bs_method=bs_method, grouper=grouper)
        else:
            self.bootstrap = None
        self.config.add(
            bootstrap_num=bootstrap_num,
            bs_record_num=bs_record_num,
            bs_method=bs_method
        )

        if conv_reps > 0:
            self.converge_tester = ConvergenceTester(reps=conv_reps, fitter=self, init_range=init_range)
        else:
            self.converge_tester = None
        self.config.add(
            conv_reps=conv_reps,
            init_range=init_range
        )
        self.config.metrics = metrics
        self.results = FitResults(fitter=self)
        self.save_to = save_to
        self.overwrite = overwrite
        self.silent = silent
        if not silent:
            logging.info(f"{self.__repr__()} initiated")

    def _fit(self, model=None, x_data=None, y_data=None, sigma=None, bounds="unspecified",
             metrics=None, init_guess=None, curve_fit_kwargs=None, **kwargs):
        """Core function performing fitting using `scipy.optimize.curve_fit`,
        Arguments will be inferred from instance's attributes if not provided

        Args:
        {fit_param}

        Returns: A dictionary contains least-squares fitting results
          - params: pd.Series of estimated parameter
          - pcov: `pd.Dataframe` of covariance matrix
          - metrics: None or pd.Series of calculated metrics
        """.format(fit_param=doc_helper.get(self._fit))

        from scipy.optimize import curve_fit
        from ..utility.func_tools import update_none
        model = update_none(model, self.model)
        from ..utility.func_tools import get_func_params
        parameters = get_func_params(model, exclude_x=True)
        x_data = update_none(x_data, self.x_data)
        y_data = update_none(y_data, self.y_data)
        sigma = update_none(sigma, self.config.sigma)
        if bounds == "unspecified":
            bounds = self.config.bounds
        if bounds is None:
            bounds = (-np.inf, np.inf)
        metrics = update_none(metrics, self.config.metrics)
        init_guess = update_none(init_guess, self.config.init_guess)
        curve_fit_kwargs = update_none(curve_fit_kwargs, self.config.curve_fit_kwargs)

        try:
            if not init_guess:
                # by default, use a random guess form (0, 1)
                init_guess = [np.random.random() for _ in parameters]
            if curve_fit_kwargs is None:
                curve_fit_kwargs = {}
            params, pcov = curve_fit(f=model, xdata=x_data, ydata=y_data,
                                     sigma=sigma, bounds=bounds, p0=init_guess, **curve_fit_kwargs)
            if metrics:
                metrics_res = pd.Series({name: fn(params) for name, fn in metrics.items()})
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
                metrics_res = pd.Series({name: np.nan for name, fn in metrics.items()})
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
                metrics_res = pd.Series({name: np.nan for name, fn in metrics.items()})
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
                metrics_res = pd.Series({name: np.nan for name, fn in metrics.items()})
            else:
                metrics_res = None

        return {
            'params': pd.Series(data=params, index=parameters),
            'pcov': pd.DataFrame(data=pcov, index=parameters, columns=parameters),
            'metrics': metrics_res
        }

    def point_estimate(self, **kwargs):
        """Single point estimation on the given data
        Keyword Args accepted for fitting:
        {fit_param}
        
        Fitting results will be saved to `self.results.point_estimation`
        """.format(fit_param=doc_helper.get(self._fit))

        results = self._fit(**kwargs)
        if not self.silent:
            logging.info(f'Point estimation for {self.__repr__()} finished')
        return results

    def run_bootstrap(self, bs_record_num=-1, **kwargs):
        """Use bootstrap to estimation uncertainty
        Args:
            {bootstrap_args}
            
        Returns:
            summary
            results: subsample if 0 <= bs_record_num <= bootstrap_num 
        """.format(bootstrap_args=doc_helper.get(['bs_method', 'bootstrap_num', 'grouper', 'bs_record_num'], indent=4))

        if self.bootstrap is None:
            self.bootstrap = Bootstrap(fitter=self, **kwargs)

        if 'bs_method' in kwargs.keys():
            self.bootstrap.bs_method = kwargs['bs_method']
        if 'bootstrap_num' in kwargs.keys():
            self.bootstrap.bootstrap_num = kwargs['bootstrap_num']
        if 'grouper' in kwargs.keys():
            self.bootstrap.grouper = kwargs['grouper']

        summary, results = self.bootstrap.run()
        if 0 <= bs_record_num <= results.shape[0]:
            results = results.sample(n=bs_record_num, replace=False, axis=0)

        if self.bootstrap is None:
            if not self.silent:
                logging.info('Bootstrap not conducted')
        else:
            if not self.silent:
                logging.info(f"Bootstrap using {self.bootstrap.bs_method} for "
                             f"{self.bootstrap.bootstrap_num} and "
                             f"save {self.bootstrap.bs_record_num} records")
        return summary, results

    def convergence_test(self, **kwargs):
        """Empirically estimate convergence by repeated fittings,
        Wrapper over `ConvergenceTester`
        Keyword Args:
            reps (int): number of repeated fitting to conduct, default 10
            init_range (list of 2-tuples): range of parameters to initialize fitting, default (0, 1)
            param (list of string): list of parameter to report. Report all parameter if None
            show_sd (bool): if report standard deviation for parameters in summary
            show_range (bool): if report range for parameters in summary

        Returns:
              summary, results
        """
        self.converge_tester = ConvergenceTester(fitter=self, **kwargs)
        return self.converge_tester.run()

    def fit(self, point_estimate=True, bootstrap=False, convergence_test=False, **kwargs):
        """Run fitting, configuration are from the object
        Args:
            point_estimate (bool): if do point estimation, default True
            bootstrap (bool): if do bootstrap, default False
            convergence_test (bool): if do convergence test, default False
        """

        save_to = kwargs.pop('save_to', self.save_to)
        overwrite = kwargs.pop('overwrite', self.overwrite)
        from pathlib import Path
        if save_to and (overwrite is False) and Path(save_to).exists():
            # result stream to hard drive, check if the result exists
            from json import JSONDecodeError
            try:
                # don't do fitting if can saved result is readable
                logging.info(f'Try to recover info from {save_to}...')
                self.results = FitResults.from_json(save_to, fitter=self)
                logging.info('Found, skip fitting...')
                return None
            except JSONDecodeError:
                # still do the fitting
                logging.info('Can not parse JSON file, continue fitting...')
        logging.info('Perform fitting...')
        rnd_seed = kwargs.pop('rnd_seed', self.config.rnd_seed)
        if rnd_seed:
            np.random.seed(rnd_seed)

        if point_estimate:
            results = self.point_estimate(**kwargs)
            self.results.point_estimation.params = results['params']
            if results['metrics'] is not None:
                self.results.point_estimation.params.append(results['metrics'])
            self.results.point_estimation.pcov = results['pcov']

        if bootstrap and (len(self.x_data) >= 2):
            self.results.uncertainty.summary, self.results.uncertainty.records = self.run_bootstrap(**kwargs)

        if convergence_test:
            self.results.convergence.summary, self.results.convergence.records = self.convergence_test(**kwargs)

        if self.save_to:
            # stream to disk as JSON file
            from pathlib import Path
            check_dir(Path(self.save_to).parent)
            self.results.to_json(self.save_to)

    def summary(self):
        """Return a pd.series as fitting summary with flatten info"""
        return self.results.to_series()

    def to_dict(self):
        """Save fitter configuration as a dictionary

        Returns:
            Dict of configurations for the Fitter
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
        }
        return config_dict

    def to_json(self, save_to_file=None):
        """Save the fitter configuration as a json file, except for model as model object is usually not json-able
        """

        config_dict = self.to_dict()
        _ = config_dict.pop('model', None)

        if save_to_file:
            from pathlib import Path
            path = Path(save_to_file)
            if path.suffix == '.json':
                # its a named json file
                check_dir(path.parent)
                dump_json(obj=config_dict, path=path, indent=2)
            elif path.suffix == '':
                # its a directory
                check_dir(path)
                dump_json(obj=config_dict, path=str(path) + '/config.json', indent=2)
            else:
                raise NameError('Unrecognized saving path')
        else:
            return dump_json(config_dict, indent=0)

    @classmethod
    def from_json(cls, file_path, model):
        """create a fitter from saved json file

        Args:
            file_path (str): path to saved json file
            model (callable): as callable is not json-able, need to reassign
        """
        config_dict = read_json(file_path)
        return cls(model=model, **config_dict)


class FitResults:
    """A class to store, process, and visualize fitting results for single fitter

    Attributes:

         fitter (EstimatorType): proxy to the fitter

         point_estimation (AttrScope): a scope stores point estimation results, includes
             params (pd.Series): stores the parameter estimation, with extra metrics calculation
             pcov (pd.DataFrame): covariance matrix for estimated parameter

         uncertainty (AttrScope): a scope stores uncertainty estimation results, includes
             summary (pd.DataFrame): summary of each parameter or metric from records
             records (pd.DataFrame): records for stored bootstrapping results

         convergence (AttrScope): a scope stores convergence test results, includes
             summary (pd.DataFrame): summary for each parameter or metric from records
             records (pd.DataFrame): records for repeated fitting results
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
        self.uncertainty = AttrScope(keys=['summary', 'records'])
        self.convergence = AttrScope(keys=['summary', 'records'])

        # TODO: update to make visualizer work
        from .visualizer import fitting_curve_plot, bootstrap_params_dist_plot
        from ..utility.func_tools import FuncToMethod
        self.visualizer = FuncToMethod(obj=self, functions=[fitting_curve_plot,
                                                            bootstrap_params_dist_plot])

    def to_series(self):
        """Convert point_estimation, uncertainty (if possible), and convergence (if possible) to a series include
        flattened info:
        e.g. columns will include [param1, param2, param1_mean, param1_std, param1_2.5%, ..., param1_range]
        """
        from ..utility.func_tools import dict_flatten

        res = self.point_estimation.params
        if self.uncertainty.summary is not None:
            # uncertainty estimation results exists
            res = res.append(pd.Series(dict_flatten(self.uncertainty.summary.to_dict())))

        if self.convergence.summary is not None:
            # uncertainty estimation results exists
            res = res.append(pd.Series(dict_flatten(self.convergence.summary.to_dict())))

        if self.fitter is not None:
            res.name = self.fitter.name
        return res

    def to_json(self, path=None):
        """Convert results into a json string/file contains
            {
              point_estimation: { params: jsonfy(pd.Series)
                                  pcov: jsonfy(pd.DataFrame) }
              uncertainty: { summary: jsonfy(pd.DataFrame)
                             records: jsonfy(pd.DataFrame) }
              convergence: { summary: jsonfy(pd.DataFrame)
                             records: jsonfy(pd.DataFrame) }
            }
        """

        def jsonfy(target):
            try:
                return target.to_json()
            except:
                return None

        data_to_dump = {
            'point_estimation': {
                'params': jsonfy(self.point_estimation.params),
                'pcov': jsonfy(self.point_estimation.pcov)
            },
            'uncertainty': {
                'summary': jsonfy(self.uncertainty.summary),
                'records': jsonfy(self.uncertainty.records)
            },
            'convergence': {
                'summary': jsonfy(self.convergence.summary),
                'records': jsonfy(self.convergence.records)
            }
        }
        if path is None:
            return dump_json(data_to_dump)
        else:
            dump_json(data_to_dump, path=path)

    @classmethod
    def from_json(cls, json_path, fitter=None):
        """load fitting results from json records
        Note: no fitter info if fitter is None
        """

        json_data = read_json(json_path)
        results = cls(fitter=fitter)

        if 'point_estimation' in json_data.keys():
            if json_data['point_estimation']['params'] is not None:
                results.point_estimation.params = pd.read_json(json_data['point_estimation']['params'], typ='series')
            if json_data['point_estimation']['pcov'] is not None:
                results.point_estimation.pcov = pd.read_json(json_data['point_estimation']['pcov'])
        if 'uncertainty' in json_data.keys():
            if json_data['uncertainty']['summary'] is not None:
                results.uncertainty.summary = pd.read_json(json_data['uncertainty']['summary'])
            if 'record' in json_data['uncertainty'].keys():
                label = 'record'
            else:
                label = 'records'
            if json_data['uncertainty'][label] is not None:
                results.uncertainty.records = pd.read_json(json_data['uncertainty'][label])
        if 'convergence' in json_data.keys():
            if json_data['convergence']['summary'] is not None:
                results.uncertainty.summary = pd.read_json(json_data['convergence']['summary'])
            if 'record' in json_data['convergence'].keys():
                label = 'record'
            else:
                label = 'records'
            if json_data['convergence'][label] is not None:
                results.uncertainty.records = pd.read_json(json_data['convergence'][label])
        return results


class ConvergenceTester:
    """Apply repeated fitting on `SingleFitter` with perturbed initial value for empirical convergence test
    Store the convergence test results as these are separate tests from estimation

    Attributes:
        reps (int): number of repeated fittings conducted
        fitter (SingleFitter): proxy to the associated Fitter
        init_range (list of 2-tuple): a list of two tuple range (min, max) with same length as model parameters.
              All parameters are initialized from (0, 1) with random uniform draw

    Methods:
        run: run converge test and return a summary and full records
    """

    def __init__(self, fitter, reps=10, init_range=None):
        """Apply convergence test to given fitter

        Args:
            fitter (SingleFitter): the target single fitter
            reps (int): number of repeated fitting, default 10
            init_range (list of 2-tuple): a list of two tuple range (min, max) with same length as model parameters.
              All parameters are initialized from (0, 1) with random uniform draw
        """
        self.reps = reps
        self.fitter = fitter
        self.init_range = init_range

    def _get_summary(self, records, param=None, show_sd=True, show_range=True, **kwargs):
        """Utility to summarize multiple fitting result"""

        from ..utility.func_tools import dict_flatten
        report_data = records
        if param:
            report_data = report_data[param]
        report_data = report_data.describe()
        stats = ['mean']
        if show_sd:
            stats.append('std')
        if show_range:
            report_data.loc['range'] = report_data.loc['max'] - report_data.loc['min']
            stats.append('range')

        def add_prefix(name):
            """Prefix 'conv_' is added to convergence test results"""
            return 'conv_' + name

        return pd.Series(dict_flatten(report_data.loc[stats].to_dict()), name=self.fitter.name).rename(add_prefix)

    def run(self, **kwargs):
        """Run convergence test, report a summary and full records

        Keyword Args:
            param (list): list of parameter/metric estimated to report. e.g. ['A', 'kA'].
              All values are reported if None
            show_sd (bool): if report the standard deviation for reported params
            show_range (bool): if report the range (max - min) for reported params

        Returns:
            summary: A pd.Series contains the `mean`, `sd`, `range` for each reported parameter
            records: A pd.Dataframe contains the all records
        """

        if not self.init_range:
            init_range = [(0, 1) for _ in self.fitter.parameters]
        conv_test_res = [
            self.fitter.point_estimate(init_guess=[np.random.uniform(low, high) for (low, high) in init_range])
            for _ in range(self.reps)
        ]

        def results_to_series(result):
            if result['metrics'] is not None:
                return result['params'].append(pd.Series(result['metrics']))
            else:
                return result['params']

        records = pd.DataFrame([results_to_series(result) for result in conv_test_res])

        return self._get_summary(records, **kwargs), records



