"""
This sub-module contains the classic least-squares fitting for each sequence individually to given kinetic model,


Several functions are included:
  - point estimation using `scipy.optimize.curve_fit`
  - option to exclude zero in fitting
  - option to initialize values
  - weighted fitting depends on the customized weights
  - confidence interval estimation using bootstrap
"""

from ..estimator import EstimatorType
from ..utility.doc_helper import DocHelper
from ..utility.file_tools import read_json, dump_json, check_dir
from ..utility.log import logging
import pandas as pd
import numpy as np

doc_helper = DocHelper(
    x_data=('list', 'list of x values in fitting'),
    y_data=('list, pd.Series', 'y values in fitting'),
    model=('callable', 'model to fit'),
    parameters=('list', 'Optional. List of parameter names, extracted from model if None'),
    name=('str', "Optional. Fitter's name"),
    sigma=('list, pd.Series, or pd.DataFrame', 'Optional, same uniq_seq_num as y_data/y_data_batch.'
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


class SingleFitter(EstimatorType):
    """`scipy.optimize.curve_fit` to fit a model for a single dataset
    Can do point estimation or bootstrap for empirical CI estimation

    Attributes:
        {attr}
        bootstrap (Bootstrap): proxy to the bootstrap object
        results (FitResult): proxy to the FitResult object
        config (AttrScope): name space for fitting, contains
        {config}
        bootstrap_config (AttrScope): name space for bootstrap, contains
        {bs_config}
            
    """.format(
        attr=doc_helper.get(['x_data', 'y_data', 'model', 'parameter', 'silent', 'name']),
        config=doc_helper.get(['opt_method', 'exclude_zero', 'init_guess', 'rnd_seed', 'sigma',
                               'bounds', 'metric', 'curve_fit_kwargs'], indent=8),
        bs_config=doc_helper.get(['bootstrap_num', 'bs_record_num', 'bs_method'], indent=8)
    )

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
            }
        """

        def jsonfy(target):
            try:
                return target.dump_json()
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
            fitter (SingleFitter): the name single fitter
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


class Bootstrap:
    """Class to perform bootstrap for fitting uncertainty estimation
    Three types of bootstrap supported:
      - `pct_res`: resample the percent residue, based on the assumption that variance are proportional to the mean
         (from data property)
      - `data`: directly resample data points
      - `stratified`: resample within groups, `grouper` is needed

    Attributes:
        fitter (`EstimatorBase` type): proxy to the associated fitter
        
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
        self.fitter = fitter
        self.bootstrap_num = bootstrap_num
        self.bs_record_num = bs_record_num
    
    @property
    def bs_method(self):
        return self._bs_method
    
    @bs_method.setter
    def bs_method(self, bs_method):
        implemented_methods = {
            'pct_res': 'pct_res',
            'resample percent residues': 'pct_res',
            'resample data points': 'data',
            'data': 'data',
            'stratified': 'stratified',
        }
        if bs_method in implemented_methods.keys():
            self._bs_method = bs_method
        else:
            raise NotImplementedError(f'Bootstrap method {bs_method} is not implemented')

    def _percent_residue(self):
        """Bootstrap percent residue"""
        try:
            y_hat = self.fitter.model(
                self.fitter.x_data, **self.fitter.results.point_estimation.params[self.fitter.parameters].to_dict()
            )
        except AttributeError:
            # if could not find point estimation, do another fit
            params = self.fitter.point_estimate()['params'][self.fitter.parameters]
            y_hat = self.fitter.model(self.fitter.x_data, **params.to_dict())

        pct_res = (self.fitter.y_data - y_hat) / y_hat
        for _ in range(self.bootstrap_num):
            pct_res_resample = np.random.choice(pct_res, size=len(pct_res), replace=True)
            yield self.fitter.x_data, y_hat * (1 + pct_res_resample)

    def _data(self):
        """Apply data based bootstrap"""
        indices = np.arange(len(self.fitter.x_data))
        for _ in range(self.bootstrap_num):
            indices_resample = np.random.choice(indices, size=len(indices), replace=True)
            yield self.fitter.x_data[indices_resample], self.fitter.y_data[indices_resample]

    def _stratified(self):
        """Apply stratified bootstrap, need grouper assigned
        x_data and y_data needs to be `Series` or the grouper key should be index
        {}
        """
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
        """Perform bootstrap with arguments indicated in instance attributes
        Returns
           summary, results
        """

        bs_sample_gen = self._bs_sample_generator()
        ix_list = pd.Series(np.arange(self.bootstrap_num))

        def fitting_runner(_):
            x_data, y_data = next(bs_sample_gen)
            result = self.fitter.point_estimate(x_data=x_data, y_data=y_data)
            res_series = pd.Series(data=result['params'], index=self.fitter.parameters)
            if result['metrics'] is not None:
                for key, value in result['metrics'].items():
                    res_series[key] = value
            res_series['x_data'] = x_data
            res_series['y_data'] = y_data
            return res_series

        results = ix_list.apply(fitting_runner)
        summary = results.describe(percentiles=[0.025, 0.5, 0.975], include=np.number)
        allowed_stats = ['mean', 'std', '2.5%', '50%', '97.5%']
        from ..utility.func_tools import dict_flatten
        summary = pd.Series(dict_flatten(summary.loc[allowed_stats].to_dict()))
        return summary, results


def _read_seq_json(json_path):
    """Read single fitting results from json file and return a summarized pd.Series"""
    fit_res = FitResults.from_json(json_path)
    return fit_res.to_series()


def _read_work_fn(seq):
    """Work function to read JSON results for each sequence"""

    res = _read_seq_json(seq[1])
    res.name = seq[0]
    return res


class BatchFitResults:
    """Parse, store, and visualize BatchFitter results
    Only save results (detached from each fitter), corresponding fitter should be found by sequence

    Attributes:
        fitter: proxy to the `BatchFitter`
        summary (`pd.DataFrame`): summarized results with each sequence as index
        bs_record (dict of pd.DataFrame): {seq: `SingleFitter.results.uncertainty.records`}
        conv_record (dict of pd.DataFrame): {seq: `SingleFitter.results.convergence.records}

    Methods:
        summary_to_csv: export summary dataframe as csv file
        dump_json: preferred format to save results
        to_pickle: save results as pickled dictionary
        from_pickle: load bootstrapping results from picked dictionary
        from_folder: link results to a saved folder
        from_record: overall method to infer either load `BatchFitResults` from pickled or a folder
    """

    def __init__(self, fitter=None, result_path=None):
        """Init a BatchFitResults instance
        Args:
            fitter (`BatchFitter`): corresponding fitter
            result_path (str): optional, path to saved results
        """
        self.fitter = fitter
        self.bs_record = None
        self.conv_record = None
        self.summary = None
        self._result_path = result_path
        self._sep_files = None
        if result_path:
            self.parse_saved_results()

        # TODO: add visualization here

    @staticmethod
    def generate_summary(result_folder_path, n_core=1, save_to=None):
        """Generate a summary csv file from given result folder. This could be used if summary was not successfully
        generated during fitting

        Result folder should have a structure of:
          - seqs
            - [seq name or hash].json
            - [if hash] seq_to_hash.json

        Args:
            result_folder_path (str): path to the root of `results` folder
            n_core (int): number of threads to process in parallel. Default 1
            save_to (str): save CSV file to local path

        Returns:
            pd.DataFrame of summary
        """
        from pathlib import Path
        from ..utility.file_tools import get_file_list

        if Path(result_folder_path).joinpath('seqs').exists():
            seq_root = Path(result_folder_path).joinpath('seqs')
        else:
            seq_root = Path(result_folder_path)

        file_list = get_file_list(str(seq_root), full_path=False)
        if 'seq_to_hash.json' in file_list:
            from k_seq.utility.file_tools import read_json
            file_list = read_json(seq_root.joinpath('seq_to_hash.json'))
            file_list = ((key, seq_root.joinpath(f'{hash_}.json')) for key, hash_ in file_list.items())
        else:
            file_list = ((seq, seq_root.joinpath(f'{seq}.json')) for seq in file_list)

        if n_core > 1:
            import multiprocessing as mp
            pool = mp.Pool(n_core)
            result = pool.map(_read_work_fn, file_list)
        else:
            result = [_read_work_fn(seq) for seq in file_list]
        result = pd.DataFrame(result)
        if save_to is not None:
            result.to_csv(save_to)
        return result

    def parse_saved_results(self):
        """Load/link data from `self.result_path`
        TODO: Need to set internal trigger of how to load results
        """

        from pathlib import Path
        result_path = Path(self.result_path)
        if result_path.is_dir():
            if result_path.joinpath('results.pkl').exists():
                self.sep_files = False
                result_path = result_path.joinpath('results.pkl')
            elif result_path.joinpath('results').is_dir():
                self.sep_files = True
                result_path = result_path.joinpath('results/')
            elif result_path.joinpath('summary.pkl').exists():
                self.sep_files = True
                result_path = result_path
            else:
                raise ValueError('Can not parse result_path.')
        else:
            self.sep_files = False
            result_path = result_path
        from ..utility.file_tools import read_pickle
        if self.sep_files:
            # record the results root
            self.result_path = result_path
            self.summary = read_pickle(result_path.joinpath('summary.pkl'))
        else:
            results = read_pickle(result_path)
            self.summary = results['summary']
            if 'bs_record' in results.keys():
                self.bs_record = results['bs_record']

    @property
    def bs_record(self):
        if self.sep_files is True:
            print('Bootstrap records are saved as separate files, use `get_bs_record` instead')
        else:
            return self._bs_record

    @bs_record.setter
    def bs_record(self, value):
        self._bs_record = value
        self.sep_files = False

    def get_bs_record(self, seq=None):
        """Load bootstrap records for seq from files
        Args:
            seq (str or a list of str): a sequence or a list of sequence

        Returns:
            a pd.DataFrame of bootstraprecordsif seq is str
            a dict of pd.DataFrame contains bootstrap records if seq is a list of str
        """
        import pandas as pd
        import numpy as np
        from ..utility.file_tools import read_pickle

        if isinstance(seq, (list, pd.Series, np.ndarray)):
            return {seq: read_pickle(str(self.result_path) + '/' + s + '.pkl') for s in seq}
        else:
            return read_pickle(str(self.result_path) + '/' + seq + '.pkl')

    def summary_to_csv(self, path):
        """Save summary table as csv file"""
        self.summary.to_csv(path)

    def to_pickle(self, output_dir, bs_record=True, sep_files=True):
        """Save fitting results as a pickled dict, notice: `dump_json` is preferred
        Args:
             output_dir (str): path to saved results, should be the parent of name location
             bs_record (bool): if output bs_record as well
             sep_files (bool): if save bs_records as separate files
                 If True:
                     |path/results/
                         |- summary.pkl
                         |- seqs
                             |- seq1.pkl
                             |- seq2.pkl
                              ...
                if False:
                     save to path/results.pkl contains
                     {
                         summary: pd.DataFrame
                         bs_records: {
                            seq1 (pd.DataFrame)
                            seq2 (pd.DataFrame)
                            ...
                       }
                     }
        """
        from ..utility.file_tools import dump_pickle

        if sep_files:
            check_dir(f'{output_dir}/results/')
            dump_pickle(obj=self.summary, path=f'{output_dir}/results/summary.pkl')
            if bs_record and self.bs_record is not None:
                [dump_pickle(obj=record, path=f'{output_dir}/results/{seq}.pkl')
                 for seq, record in self.bs_record.items()]
        else:
            check_dir(output_dir)
            data_to_dump = {'summary': self.summary}
            if bs_record and self.bs_record is not None:
                data_to_dump['bs_record'] = self.bs_record
            dump_pickle(obj=data_to_dump, path=output_dir + '/results.pkl')

    @classmethod
    def from_pickle(cls, path_to_pickle, fitter=None):
        """Create a `BatchFitResults` instance with results loaded from pickle
        Notice:
            this will take a very long time if the pickle is large
        """
        return cls(fitter=fitter, result_path=path_to_pickle)

    def to_json(self, output_dir, bs_record=True, sep_files=True):
        """Serialize results as json format
        Args:
             output_dir (str): path to save results, should be the parent of name location
             bs_record (bool): if output bs_record as well
             sep_files (bool): if save bs_records as separate files
                 If True:
                     |path/results/
                         |- summary.json
                         |- seqs
                             |- seq1.json
                             |- seq2.json
                              ...
                if False:
                     save to path/results.json contains
                     {
                         summary: pd.DataFrame.json
                         bs_records: {
                            seq1 (pd.DataFrame.json)
                            seq2 (pd.DataFrame.json)
                            ...
                       }
                     }
        """
        check_dir(output_dir)
        if sep_files:
            check_dir(f'{output_dir}/results/')
            dump_json(obj=self.summary.dump_json(), path=f'{output_dir}/results/summary.json')
            if bs_record and self.bs_record is not None:
                check_dir(f'{output_dir}/seqs')
                for seq, record in self.bs_record.items():
                    dump_json(obj=record.dump_json(), path=f"{output_dir}/results/seqs/{seq}.json")
        else:
            data_to_json = {'summary': self.summary.dump_json()}
            if bs_record and self.bs_record is not None:
                data_to_json['bs_record'] = {seq: record.dump_json() for seq, record in self.bs_record}
            dump_json(obj=data_to_json, path=f"{output_dir}/results.json")

    @classmethod
    def from_json(cls, fitter, json_o_path):
        """Load results from JSON
        TODO: parse JSON results
        """

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


def _work_fn(worker, point_estimate, bootstrap, convergence_test):
    """Utility work function to parallelize workers"""
    worker.fit(point_estimate=point_estimate, bootstrap=bootstrap, convergence_test=convergence_test)
    return worker


class BatchFitter(EstimatorType):
    """Fitter for least squared batch fitting

    Attributes:
        y_data_batch (pd.DataFrame): a table containing y values to fit (e.g. reacted fraction in each sample)
    {attr}
        note (str): note about this fitting job
        fitters (None or dict): a dictionary of SingleFitters if `self.config.keep_single_fitters` is True
        seq_to_fit (list of str): list of seq to fit for this job
        results (BatchFitResult): fitting results
        fit_params (AttrScope): parameters pass to each fitting, should be same for each sequence, includes:
    {fit_params}
    """.format(attr=doc_helper.get(['model', 'x_data', 'parameters'], indent=4),
               fit_params=doc_helper.get(['x_data', 'model', 'parameters', 'bounds', 'init_guess', 'opt_method',
                                          'exclude_zero', 'metrics', 'rnd_seed', 'bootstrap_num', 'bs_record_num',
                                          'bs_method', 'curve_fit_kwargs', 'silent']))

    def __repr__(self):
        from ..utility.func_tools import get_object_hex
        return f'Least-squared BatchFitter at {get_object_hex(self)}'

    def __init__(self, y_data_batch, x_data, model, sigma=None, bounds=None, seq_to_fit=None,
                 bootstrap_num=0, bs_record_num=0, bs_method='pct_res', grouper=None,
                 conv_reps=0, init_range=None,
                 opt_method='trf', exclude_zero=False, init_guess=None, metrics=None, rnd_seed=None,
                 curve_fit_kwargs=None, note=None, result_path=None):
        """
        Args:
            y_data_batch (pd.DataFrame or str): a set of y_data to fit form rows of y_data_batch, can be a string
                indicate the path to a pickled pd.DataFrame record
        {args}
            note (str): Optional notes for the fitter
            results: a proxy to BatchFitResults
        """.format(args=doc_helper.get(['x_data', 'model', 'bounds', 'sigma', 'bootstrap_num', 'bs_record_num',
                                        'bs_method', 'grouper', 'opt_method', 'exclude_zero', 'init_guess',
                                        'metrics', 'rnd_seed', 'curve_fit_kwargs', 'seq_to_fit'], indent=4))

        from ..utility.func_tools import get_func_params, AttrScope
        super().__init__()

        logging.info('Creating the BatchFitter...')

        self.model = model
        self.parameters = get_func_params(model, exclude_x=True)
        self.note = note

        # parse y_data_batch
        from ..utility.file_tools import table_object_to_dataframe
        self.y_data_batch = table_object_to_dataframe(y_data_batch)

        # process seq_to_fit
        if seq_to_fit is not None:
            if isinstance(seq_to_fit, (list, np.ndarray, pd.Series)):
                self.seq_list = list(seq_to_fit)
            else:
                raise TypeError('Unknown seq_to_fit type, is it list-like?')
        self.seq_to_fit = seq_to_fit

        # prep fitting params shared by all fittings
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
            logging.warning("Number of data points less than 2, bootstrap will not be performed")
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
            conv_reps=conv_reps,
            init_range=init_range,
            exclude_zero=exclude_zero,
            init_guess=init_guess,
            metrics=metrics,
            rnd_seed=rnd_seed,
            grouper=grouper if bs_method == 'stratified' else None,
            curve_fit_kwargs=curve_fit_kwargs,
            silent=True
        )

        self.results = BatchFitResults(fitter=self, result_path=result_path)

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

    # def generate_summary(self, result_folder_path):
    #     """Generate a summary csv file from given result folder. This could be used if summary was not successfully
    #     generated during fitting

        # Result folder should have a structure of:
        #   - seqs
        #     - [seq name or hash].json
        #     - [if hash] seq_to_hash.json

        # Args:
        #     result_folder_path (str): path to the root of `results` folder

        # Returns:
        #     pd.DataFrame of summary
        # """
        # from pathlib import Path
        # from ..utility.file_tools import get_file_list
        # if Path(result_folder_path).append('seqs').exists():

    def worker_generator(self, stream_to_disk=None, overwrite=False):
        """Return a generator of worker for each sequence"""

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
                    save_to=None if stream_to_disk is None else f"{stream_to_disk}/seqs/{seq}.json",
                    overwrite=overwrite,
                    **self.fit_params.__dict__
                )
            except:
                raise Exception(f'Can not create fitting worker for {seq}')

    def fit(self, deduplicate=False, parallel_cores=1,
            point_estimate=True, bootstrap=False, convergence_test=False,
            stream_to_disk=None, overwrite=False):
        """Run the estimation
        Args:
            deduplicate (bool): hash the y_data_batch to deduplicate before fitting if True
            parallel_cores (int): number of parallel cores to use. Default 1
            point_estimate (bool): if do point estimation, default True
            bootstrap (bool): if do bootstrap uncertainty estimation, default False
            convergence_test (bool): if do convergence test, default False
            stream_to_disk (str): Directly stream fitting results to disk if output path is given
                will create a folder with name of seq/hash with pickled dict of fitting results
            overwrite (bool): if overwrite existing results when stream to disk. Default False.
        """

        logging.info('Batch fitting starting...')

        if deduplicate:
            self._hash()
            if stream_to_disk:
                check_dir(stream_to_disk + '/seqs/')
                dump_json(obj=self._seq_to_hash, path=f"{stream_to_disk}/seqs/seq_to_hash.json")

        from functools import partial
        work_fn = partial(_work_fn, point_estimate=point_estimate,
                          bootstrap=bootstrap, convergence_test=convergence_test)
        worker_generator = self.worker_generator(stream_to_disk=stream_to_disk, overwrite=overwrite)
        if parallel_cores > 1:
            import multiprocessing as mp
            pool = mp.Pool(processes=int(parallel_cores))
            logging.info('Use multiprocessing to fit in {} parallel threads...'.format(parallel_cores))
            workers = pool.map(work_fn, worker_generator)
        else:
            # single thread
            logging.info('Fitting in a single thread...')
            workers = [work_fn(fitter) for fitter in worker_generator]

        # record results
        if self.bootstrap:
            self.results.bs_record = {worker.name: worker.results.uncertainty.records for worker in workers}
        if convergence_test:
            self.results.conv_record = {worker.name: worker.results.convergence.records for worker in workers}
        self.results.summary = pd.DataFrame({worker.name: worker.summary() for worker in workers}).transpose()

        if deduplicate:
            self._hash_inv()
        logging.info('Fitting finished')

    def summary(self, save_to=None):
        if save_to is None:
            return self.results.summary
        else:
            self.results.summary.to_csv(save_to)

    def _hash(self):
        """De-duplicate rows before fitting"""

        def hash_series(row):
            return hash(tuple(row))

        self._y_data_batch_dup = self.y_data_batch.copy()
        if self.seq_to_fit is not None:
            # filter the seq to fit
            self._seq_to_fit_dup = self.seq_to_fit.copy()
            self.y_data_batch = self.y_data_batch.loc[self.seq_to_fit]
        # find seq to hash mapping
        self._seq_to_hash = self.y_data_batch.apply(hash_series, axis=1).to_dict()
        # only keep the first instance of each hash
        self.y_data_batch = self.y_data_batch[~self.y_data_batch.duplicated(keep='first')]
        if isinstance(self.sigma, pd.DataFrame):
            # only accept sigma as an pd.DataFrame
            self._sigma_dup = self.sigma.copy()
            # filter sigma table for only the first instance of each hash
            self.sigma = self.sigma.loc[self.y_data_batch.index]
            # convert seq --> hash
            self.sigma.rename(index=self._seq_to_hash, inplace=True)
        # convert seq --> hash
        self.y_data_batch.rename(index=self._seq_to_hash, inplace=True)
        if self.seq_to_fit is not None:
            self.seq_to_fit = [self._seq_to_hash[seq] for seq in self.seq_to_fit]
        logging.info('Shrink rows in table by removing duplicates: '
                     f'{self._y_data_batch_dup.shape[0]} --> {self.y_data_batch.shape[0]}')

    def _hash_inv(self):
        """Recover the hashed results"""

        logging.info('Recovering original table from hash...')

        def get_summary(seq):
            return self.results.summary.loc[self._seq_to_hash[seq]]

        # map hash --> seq for results summary
        self.results.summary = pd.Series(data=list(self._seq_to_hash.keys()),
                                         index=list(self._seq_to_hash.keys())).apply(get_summary)
        # map hash --> seq for bs_record
        if self.results.bs_record is not None:
            self.results.bs_record = {seq: self.results.bs_record[seq_hash]
                                      for seq, seq_hash in self._seq_to_hash.items()}
        # recover the original y_data_batch
        self.y_data_batch = self._y_data_batch_dup.copy()
        del self._y_data_batch_dup
        # recover the original sigma if exists
        if hasattr(self, '_sigma_dup'):
            self.sigma = self._sigma_dup.copy()
            del self._sigma_dup
        # recover the original seq_to_fit if exists
        if hasattr(self, '_seq_to_fit'):
            self.seq_to_fit = self._seq_to_fit_dup.copy()
            del self._seq_to_fit_dup

    def save_model(self, output_dir, results=True, bs_results=True, sep_files=True, tables=True):
        """Save model to a given directory
        model_config will be saved as a pickled dictionary to recover the model
            - except for `y_data_batch` and `sigma` which are too large

        Args:
            output_dir (str): path to save the model, create if the path does not exist
            results (bool): if save estimation results to `results` as well, to be load by `BatchFitResults`,
                Default True
            bs_results (bool): if save bootstrap results
            sep_files (bool): if save the record of bootstrap as separate files in a subfolder `results/seqs/`
                Default True
            tables (bool): if save table (y_data_batch, sigma) in the folder. Default True
        """
        from ..utility.file_tools import dump_pickle

        check_dir(output_dir)
        dump_pickle(
            obj={
                **{'parameters': self.parameters,
                   'note': self.note,
                   'seq_to_fit': self.seq_to_fit},
                **self.fit_params.__dict__
            },
            path=str(output_dir) + '/model_config.pkl'
        )
        if results:
            self.save_results(result_path=str(output_dir), bs_results=bs_results, sep_files=sep_files)
        if tables is not None:
            dump_pickle(obj=self.y_data_batch, path=str(output_dir) + '/y_data.pkl')
            if self.sigma is not None:
                dump_pickle(obj=self.sigma, path=str(output_dir) + '/sigma.pkl')

    def save_results(self, result_path, bs_results=True, sep_files=True, use_pickle=False):
        """Save results to disk as JSON or pickle
        JSON is preferred for speed, readability, compatibility, and security
        """
        if use_pickle:
            self.results.to_pickle(result_path, bs_record=bs_results, sep_files=sep_files)
        else:
            self.results.to_json(result_path, bs_record=bs_results, sep_files=sep_files)

    @classmethod
    def load_model(cls, model_path, y_data_batch=None, sigma=None, result_path=None):
        """Create a model from pickled config file

        Args:
            model_path (str): path to picked model configuration file or the saved folder
            y_data_batch (pd.DataFrame or str): y_data table for fitting
            sigma (pd.DataFrame or str): optional sigma table for fitting
            result_path (str): path to fitting results

        Returns:
            a BatchFitter instance
        """

        from ..utility.file_tools import read_pickle
        from pathlib import Path

        config_file = model_path if Path(model_path).is_file() else model_path + '/model_config.pkl'
        model_config = read_pickle(config_file)
        if y_data_batch is None:
            # try infer from the folder
            y_data_batch = read_pickle(model_path + '/y_data.pkl')
        else:
            if isinstance(y_data_batch, str):
                y_data_batch = read_pickle(y_data_batch)
        if sigma is not None:
            if isinstance(sigma, str):
                sigma = read_pickle(sigma)
        return cls(y_data_batch=y_data_batch, sigma=sigma, result_path=result_path, **model_config)


def load_estimation_results(point_est_csv=None, seqtable_path=None, bootstrap_csv=None,
                            **kwargs):
    """Collect estimation results (summary.csv files)and compose a table
    As

    Args:
        seq_table (str): path to pickled `SeqData` or `pd.DataFrame` object,
            will import 'input_counts'/, 'mean_counts'
        point_est_csv (str): optional, path to reported csv file from point estimation
        seqtable_path (str): optional. path to original seqTable object for count info
        bootstrap_csv (str): optional. path to csv file from bootstrap
        kwargs: optional keyword argument of callable to calculate extra columns, apply on results dataframe row-wise

    Returns:
        a pd.DataFrame contains composed results from provided information

    """

    point_est_res = pd.read_csv(point_est_csv, index_col=0)
    est_res = point_est_res[point_est_res.columns]
    seq_list = est_res.index.values

    if seqtable_path:
        # add counts in input pool
        from ..utility import file_tools
        seq_table = file_tools.read_pickle(seqtable_path)
        if seq_table.grouper and hasattr(seq_table.grouper, 'input'):
            est_res['input_counts'] = seq_table.table[seq_table.grouper.input.group].loc[seq_list].mean(axis=1)
        est_res['mean_counts'] = seq_table.table.loc[seq_list].mean(axis=1)
        est_res['min_counts'] = seq_table.table.loc[seq_list].min(axis=1)

        if hasattr(seq_table, 'pool_peaks'):
            # has doped pool, add dist to center
            from ..data import landscape
            mega_peak = landscape.Peak.from_peak_list(seq_table.pool_peaks)
            est_res['dist_to_center'] = mega_peak.dist_to_center

    if bootstrap_csv:
        bootstrap_res = pd.read_csv(bootstrap_csv, index_col=0)
        # add bootstrap results
        est_res[['kA_mean', 'kA_std', 'kA_2.5%', 'kA_50%', 'kA_97.5%']] = bootstrap_res[
            ['kA_mean', 'kA_std', 'kA_2.5%', 'kA_50%', 'kA_97.5%']]
        est_res['A_range'] = bootstrap_res['A_97.5%'] - bootstrap_res['A_2.5%']

    if kwargs:
        for key, func in kwargs.items():
            if callable(func):
                est_res[key] = est_res.apply(func, axis=1)
            else:
                logging.error(f'Keyword argument {key} is not a function', error_type=TypeError)
    return est_res
