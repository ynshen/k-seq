"""Modules for model identifiability analysis"""

import os
import pandas as pd
import numpy as np
from .least_squares import doc_helper
from yutility import logging
logging.set_level('info')
from scipy import stats


# common for parameter correlation for model identifiability
def remove_nan(df):
    return df[~df.isna().any(axis=1)]


def spearman(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.spearmanr(records['k'], records['A']).correlation
    else:
        return np.nan


def pearson(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.pearsonr(records['k'], records['A'])[0]
    else:
        return np.nan


def spearman_log(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.spearmanr(np.log10(records['k']), np.log10(records['A'])).correlation
    else:
        return np.nan


def pearson_log(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.pearsonr(np.log10(records['k']), np.log10(records['A']))[0]
    else:
        return np.nan


def kendall_log(records):
    records = remove_nan(records)
    if records.shape[0] > 10:
        return stats.kendalltau(np.log10(records['k']), np.log10(records['A'])).correlation
    else:
        return np.nan


def gamma(df):
    """Get metric gamma = log_10{\sigma_k \mu_A} / {\sigma kA}"""
    if isinstance(df, (pd.Series, pd.DataFrame)):
        return np.log10(df['bs_k_std'] * df['bs_A_mean'] / df['bs_kA_std'])
    else:
        raise TypeError('Input should be a pd.Series or pd.DataFrame')


def _parameter_gen(param_range, log, size):
    from ..data.simu import DistGenerators

    if log:
        return 10 ** DistGenerators.uniform(low=np.log10(param_range[0]), high=np.log10(param_range[1]),
                                            size=size)
    else:
        return DistGenerators.uniform(low=param_range[0], high=param_range[1], size=size)





@doc_helper.compose("""Generate a 2d convergence map for randomly sampled data points from given parameter range
    it simulates sample_n sequence samples with random params selected from the range of (param1_range, param2_range),
    on the optional log scale
    
""")
class ParamMap:

    def __init__(self, model, sample_n, x_data, save_to,
                 param1_name, param1_range, param2_name, param2_range,
                 param1_log=False, param2_log=False, model_kwargs=None,
                 bootstrap_num=100, bs_record_num=50, bs_method='data', bs_stats=None, grouper=None, record_full=False,
                 conv_reps=20, conv_stats=None, conv_init_range=None,
                 fitting_kwargs=None, seed=23):
        from ..utility.func_tools import AttrScope
        from ..utility.file_tools import dump_json
        from pathlib import Path

        # assign model
        self.model = model
        self.parameters = None
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # assign parameter
        self.param1 = AttrScope(name=param1_name, range=param1_range, log=param1_log)
        self.param2 = AttrScope(name=param2_name, range=param2_range, log=param2_log)
        self.sample_n = sample_n

        # assign fitting specs
        fitting_kwargs = fitting_kwargs if fitting_kwargs is not None else {}
        self.batchfitter_kwargs = dict(
            bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method, bs_stats=bs_stats,
            grouper=grouper, record_full=record_full,
            conv_reps=conv_reps, conv_init_range=conv_init_range, conv_stats=conv_stats,
        )
        self.batchfitter_kwargs.update(fitting_kwargs)

        # assign x_data
        if not isinstance(x_data, pd.Series):
            x_data = pd.Series(x_data)
        self.x_data = x_data
        self.y_dataframe = None
        self.seed = seed
        self.results = None

        config = dict(
            sample_n=sample_n, x_data=list(x_data),
            param1_name=param1_name, param1_range=param1_range, param1_log=param1_log,
            param2_name=param2_name, param2_range=param2_range, param2_log=param2_log,
            bootstrap_num=bootstrap_num, bs_record_num=bs_record_num, bs_method=bs_method,
            bs_stats=list(bs_stats.keys()) if isinstance(bs_stats, dict) else bs_stats,
            conv_reps=conv_reps,
            conv_stats=list(conv_stats.keys()) if isinstance(conv_stats, dict) else conv_stats,
            conv_init_range=conv_init_range if conv_init_range is None else list(conv_init_range),
            seed=seed
        )

        # create saving path and save config
        self.save_to = Path(save_to)
        if not self.save_to.exists():
            self.save_to.mkdir(parents=True)
            logging.info(f'Output dir {str(self.save_to)} created')
        dump_json(config, path=self.save_to.joinpath('config.json'))

    def simulate_samples(self, grid=True, const_err=None, rel_err=None, y_enforce_positive=True):
        """Simulate a set of samples (param1 and param2)"""

        logging.info(f"Simulating dataset with const_error: {const_err}, pct_error: {rel_err}, "
                     f"y_enforce_positive: {y_enforce_positive}...")

        if self.seed is not None:
            np.random.seed(self.seed)

        if grid:
            n_cell = int(np.sqrt(self.sample_n)) + 1
            if self.param1.log:
                param1 = np.logspace(np.log10(self.param1.range[0]), np.log10(self.param1.range[1]), n_cell)
            else:
                param1 = np.linspace(self.param1.range[0], self.param1.range[1], n_cell)
            if self.param2.log:
                param2 = np.logspace(np.log10(self.param2.range[0]), np.log10(self.param2.range[1]), n_cell)
            else:
                param2 = np.linspace(self.param2.range[0], self.param2.range[1], n_cell)

            self.parameters = pd.DataFrame({
                self.param1.name: np.repeat(np.expand_dims(param1, -1), n_cell, -1).T.reshape(-1),
                self.param2.name: np.repeat(param2, n_cell)
            })
        else:
            self.parameters = pd.DataFrame({
                self.param1.name: _parameter_gen(self.param1.range, self.param1.log, size=self.sample_n),
                self.param2.name: _parameter_gen(self.param2.range, self.param2.log, size=self.sample_n)
            })

        def partial_model(param):

            y = self.model(self.x_data, **param.to_dict(), **self.model_kwargs)
            if not isinstance(y, pd.Series) and isinstance(self.x_data, pd.Series):
                y = pd.Series(y, index=self.x_data.index)
            return y

        self.y_dataframe = self.parameters.apply(partial_model, axis=1)

        if const_err is not None:
            self.y_dataframe += np.random.normal(loc=0, scale=const_err,
                                                 size=self.y_dataframe.shape)
        if rel_err is not None:
            self.y_dataframe += np.random.normal(loc=0, scale=self.y_dataframe * rel_err,
                                              size=self.y_dataframe.shape)

        if y_enforce_positive:
            self.y_dataframe[self.y_dataframe < 0] = 0

        logging.info('Simulation done.')

        self.save_to.joinpath('data').mkdir(exist_ok=True)
        self.parameters.to_csv(self.save_to.joinpath('data', 'parameters.csv'))
        self.y_dataframe.to_csv(self.save_to.joinpath('data', 'y.csv'))
        self.x_data.to_csv(self.save_to.joinpath('data', 'x.csv'))

    def fit(self, **kwargs):
        """Batch fit simulated result"""

        if self.seed:
            np.random.seed(self.seed)

        if self.y_dataframe is None:
            self.simulate_samples()

        from ..estimate.least_squares_batch import BatchFitter
        fitter = BatchFitter(y_dataframe=self.y_dataframe, x_data=self.x_data, model=self.model,
                             large_dataset=True, **self.batchfitter_kwargs)
        fitter.fit(convergence_test=self.batchfitter_kwargs['conv_reps'] > 0,
                   bootstrap=self.batchfitter_kwargs['bootstrap_num'] > 0,
                   point_estimate=True, stream_to=self.save_to, **kwargs)
        self.results = fitter.results
        self.results.summary.to_csv(self.save_to.joinpath('results.csv'))
        logging.info(f"Result saved to {self.save_to.joinpath('results.csv')}")
        os.system(f"cd {self.save_to} && tar -czf seqs.tar.gz seqs && rm -r seqs")

    @classmethod
    def load_result(cls, result_path, model=None):
        from pathlib import Path
        from ..utility.file_tools import read_json
        from .least_squares_batch import BatchFitResults

        result_path = Path(result_path)
        config = read_json(result_path.joinpath('config.json'))
        parameters = pd.read_csv(result_path.joinpath('data', 'parameters.csv'), index_col=0)
        parameters.index = parameters.index.astype('str')
        try:
            y_dataframe = pd.read_csv(result_path.joinpath('data', 'y.csv'), index_col=0)
        except FileNotFoundError:
            y_dataframe = pd.read_csv(result_path.joinpath('data', 'y_values.csv'), index_col=0)
        y_dataframe.index = y_dataframe.index.astype('str')
        try:
            x_data = pd.read_csv(result_path.joinpath('data', 'x.csv'), index_col=0, header=None, squeeze=True)
        except FileNotFoundError:
            x_data = pd.read_csv(result_path.joinpath('data', 'x_values.csv'), index_col=0, header=None, squeeze=True)

        x_data.index = x_data.index.astype('str')
        _ = config.pop('x_data', None)
        _ = config.pop('x_values', None)

        conv_map = cls(model=model, x_data=x_data, save_to=result_path, **config)
        conv_map.y_dataframe = y_dataframe
        conv_map.parameters = parameters
        conv_map.results = BatchFitResults.from_json(estimator=None,
                                                     path_to_folder=result_path,
                                                     x_data=x_data, y_dataframe=y_dataframe, sigma=None)

        return conv_map

    def get_metric_values(self, metric, finite_only=False):
        """Returns a pd.Series contains the metric value"""

        if isinstance(metric, str):
            metric_val = self.results.summary[metric]
        elif callable(metric):
            metric_val = self.results.summary.apply(metric, axis=1)
        else:
            raise TypeError()
        metric_val.index = metric_val.index.astype(str)
        if finite_only:
            metric_val = metric_val[np.isfinite(metric_val)]
        return metric_val

    def plot_map(self, metric=None, metric_label=None, scatter=False, gridsize=50, figsize=(5, 5),
                 ax=None, cax_pos=(0.91, 0.58, 0.03, 0.3), **plot_kwargs):
        import matplotlib.pyplot as plt
        from ..utility.plot_tools import ax_none

        if metric is not None:
            if isinstance(metric, str):
                metric = self.results.summary[metric]
            elif callable(metric):
                metric = self.results.summary.apply(metric, axis=1)
            else:
                logging.error('Unknown metric type, should be a column name or a callable', TypeError)

        ax = ax_none(ax, figsize=figsize)
        if scatter:
            im = ax.scatter(x=self.parameters[self.param1.name], y=self.parameters[self.param2.name],
                            c=metric, cmap='viridis', s=10, alpha=0.6, **plot_kwargs)
            if self.param1.log:
                ax.set_xscale('log')
            if self.param2.log:
                ax.set_yscale('log')
        else:
            xscale = 'log' if self.param1.log else 'linear'
            yscale = 'log' if self.param2.log else 'linear'

            im = ax.hexbin(x=self.parameters[self.param1.name], y=self.parameters[self.param2.name], gridsize=gridsize,
                           C=metric, reduce_C_function=np.mean, xscale=xscale, yscale=yscale,
                           cmap='viridis', **plot_kwargs)

        ax.set_xlim(*self.param1.range)
        ax.set_ylim(*self.param2.range)
        ax.tick_params(labelsize=12)
        ax.set_xlabel(f'True {self.param1.name}', fontsize=12)
        ax.set_ylabel(f'True {self.param2.name}', fontsize=12)

        fig = plt.gcf()
        if cax_pos is not None:
            cax = fig.add_axes(cax_pos)
            fig.colorbar(im, cax=cax, orientation='vertical')
            cax.set_ylabel(metric_label, fontsize=12)
            return ax, cax
        else:
            return ax



