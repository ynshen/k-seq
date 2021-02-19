"""Module to access the convergence of fitting, e.g. model identifiability"""
from .least_squares import doc_helper
from yutility import logging
import numpy as np
import pandas as pd


@doc_helper.compose("""Apply repeated fitting on a Estimator with perturbed initial value to test empirical convergence
Store the convergence test results as these are separate tests from estimation

Attributes:
    <<conv_reps, estimator, conv_init_range, conv_stats>>

Methods:
    run: run converge test and return a summary and full records
""")
class ConvergenceTester:

    @doc_helper.compose("""Apply convergence test to given estimator
    
    Args:
        <<estimator, conv_reps, conv_init_range, conv_stats>>
    """)
    def __init__(self, estimator, conv_reps=10, conv_init_range=None, conv_stats=None):
        self.conv_reps = conv_reps
        self.estimator = estimator
        self.conv_init_range = conv_init_range
        self.conv_stats = conv_stats

    def _get_summary(self, records):
        """Utility to summarize multiple fitting result"""

        from ..utility.func_tools import dict_flatten
        report_data = records.describe()
        report_data.loc['range'] = report_data.loc['max'] - report_data.loc['min']
        summary = dict_flatten(report_data.loc[['mean', 'std', 'range']].to_dict())
        if self.conv_stats is not None:

            def format_stat(res):
                if isinstance(res, (int, float, bool, dict)):
                    return res
                elif isinstance(res, pd.Series):
                    return res.to_dict()
                else:
                    logging.error('Unrecognized return value for bs_stats', error_type=TypeError)

            stats = {key: format_stat(stat(records)) for key, stat in self.conv_stats.items()}
            summary = {**summary, **dict_flatten(stats)}

        def add_prefix(name):
            """Prefix 'conv_' is added to convergence test results"""
            return 'conv_' + name

        return pd.Series(summary, name=self.estimator.name).rename(add_prefix)

    def run(self):
        """Run convergence test, report a summary and full records

        Returns:
            summary: A pd.Series contains the `mean`, `sd`, `range` for each reported parameter, and conv_stats result
            records: A pd.Dataframe contains the full records
        """

        if not self.conv_init_range:
            init_range = [(0, 1) for _ in self.estimator.parameters]
        else:
            init_range = self.conv_init_range

        conv_test_res = [
            self.estimator.point_estimate(init_guess=[np.random.uniform(low, high) for (low, high) in init_range])
            for _ in range(self.conv_reps)
        ]

        def results_to_series(result):
            if result['metrics'] is not None:
                return result['params'].append(pd.Series(result['metrics']))
            else:
                return result['params']

        records = pd.DataFrame([results_to_series(result) for result in conv_test_res])

        return self._get_summary(records), records


