"""Uncertainty estimation through bootstrap"""
from .least_squares import doc_helper
from yutility import logging
import numpy as np
import pandas as pd


@doc_helper.compose("""Perform bootstrap for fitting uncertainty estimation

Three types of bootstrap supported:
  - `pct_res`: resample the percent residue, based on the assumption that variance are proportional to the mean
     (from data property)
  - `data`: directly resample data points
  - `stratified`: resample within groups, `grouper` is required

Attributes:
    estimator (`EstimatorBase` type): accessor to the associated estimator
    <<bs_method, bootstrap_num, bs_record_num, bs_stats, grouper, record_full>>
""")
class Bootstrap:

    def __repr__(self):
        return f"Bootstrap method using {self.bs_method} (n = {self.bootstrap_num})"

    @doc_helper.compose("""
    Args:
        estimator (Estimator): the estimator generates the results
    <<bootstrap_num, bs_record_num, bs_method, bs_stats, grouper, record_full>>
    """)
    def __init__(self, estimator, bootstrap_num, bs_record_num, bs_method, grouper=None, bs_stats=None,
                 record_full=False):

        self.bs_method = bs_method
        if bs_method == 'stratified':
            from ..data.grouper import Grouper
            if isinstance(grouper, Grouper):
                grouper = grouper.group
            if isinstance(grouper, dict):
                self.grouper = grouper
            else:
                logging.error('Unsupported grouper type for stratified bootstrap', error_type=TypeError)
        self.estimator = estimator
        self.bootstrap_num = bootstrap_num
        self.bs_record_num = bs_record_num
        self.bs_stats = bs_stats
        self.record_full = record_full

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
            logging.error(f'Bootstrap method {bs_method} is not implemented', error_type=NotImplementedError)

    def _percent_residue(self):
        """Bootstrap percent residue"""
        try:
            y_hat = self.estimator.model(
                self.estimator.x_data,
                **self.estimator.results.point_estimation.params[self.estimator.parameters].to_dict()
            )
        except AttributeError:
            # if could not find point estimation, do another fit
            params = self.estimator.point_estimate()['params'][self.estimator.parameters]
            y_hat = self.estimator.model(self.estimator.x_data, **params.to_dict())

        pct_res = (self.estimator.y_data - y_hat) / y_hat
        for _ in range(self.bootstrap_num):
            pct_res_resample = np.random.choice(pct_res, size=len(pct_res), replace=True)
            yield self.estimator.x_data, y_hat * (1 + pct_res_resample)

    def _data(self):
        """Apply data based bootstrap"""
        indices = np.arange(len(self.estimator.x_data))
        for _ in range(self.bootstrap_num):
            indices_resample = np.random.choice(indices, size=len(indices), replace=True)
            yield self.estimator.x_data[indices_resample], self.estimator.y_data[indices_resample]

    def _stratified(self):
        """Apply stratified bootstrap, `grouper` is required
        `x_data` and `y_data` needs to be `Series` or the grouper key should be index
        """
        for _ in range(self.bootstrap_num):
            ix_resample = []
            for member_ix in self.grouper.values():
                ix_resample += list(np.random.choice(member_ix, size=len(member_ix), replace=True))
            yield self.estimator.x_data[ix_resample], self.estimator.y_data[ix_resample]

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

        Returns:
           summary (pd.Series): summarized results for each parameter and metrics from bootstrap
           records (pd.DataFrame): records of bootstrapped results, each row is a bootstrapped result
        """

        bs_sample_gen = self._bs_sample_generator()
        ix_list = pd.Series(np.arange(self.bootstrap_num))

        record_full = self.record_full

        def fitting_runner(_):
            x_data, y_data = next(bs_sample_gen)
            result = self.estimator.point_estimate(x_data=x_data, y_data=y_data)
            res_series = pd.Series(data=result['params'], index=self.estimator.parameters)
            if result['metrics'] is not None:
                for key, value in result['metrics'].items():
                    res_series[key] = value
            if record_full:
                res_series['x_data'] = x_data
                res_series['y_data'] = y_data
            return res_series

        records = ix_list.apply(fitting_runner)
        summary = records.describe(percentiles=[0.025, 0.5, 0.975], include=np.number)
        allowed_stats = ['mean', 'std', '2.5%', '50%', '97.5%']
        from ..utility.func_tools import dict_flatten
        summary = dict_flatten(summary.loc[allowed_stats].to_dict())
        if self.bs_stats is not None:
            def format_stat(res):
                if isinstance(res, (int, float, bool, dict)):
                    return res
                elif isinstance(res, pd.Series):
                    return res.to_dict()
                else:
                    logging.error('Unrecognized return value for bs_stats', error_type=TypeError)

            stats = {key: format_stat(stat(records)) for key, stat in self.bs_stats.items()}
            summary = {**summary, **dict_flatten(stats)}

        return pd.Series(summary, name=self.estimator.name), records
