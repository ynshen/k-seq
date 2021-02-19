"""Uncertainty estimation using replicates"""
import pandas as pd
import numpy as np


class Replicates:

    def __init__(self, estimator, replicates):
        self.estimator = estimator
        self.replicates = replicates

    @property
    def n_replicates(self):
        return len(self.replicates)

    def run(self):
        """Perform fitting for replicates"""

        def run_replicates(samples):
            x_data, y_data = self.estimator.x_data[samples], self.estimator.y_data[samples]
            result = self.estimator.point_estimate(x_data=x_data, y_data=y_data)
            res_series = pd.Series(data=result['params'], index=self.estimator.parameters)
            if result['metrics'] is not None:
                for key, value in result['metrics'].items():
                    res_series[key] = value
            return res_series

        results = pd.DataFrame([run_replicates(samples) for samples in self.replicates])
        from ..utility.func_tools import dict_flatten
        summary = dict_flatten(results.describe(include=np.number).loc[['mean', 'std']].to_dict())

        def add_prefix(name):
            """Prefix 'rep_' is added to bootstrapping results"""
            return 'rep_' + name

        return pd.Series(summary, name=self.estimator.name).rename(add_prefix), results

