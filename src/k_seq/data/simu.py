import logging

class DistGenerators:
    """A collection of random value generators from preset distributions

    Behavior:
        each distribution will return a generator if `return_gen` argument is True,
            else return a function take size value and return a generator

    Available distributions:
        lognormal
        uniform
        compo_lognormal
    """

    def __init__(self):
        pass

    @staticmethod
    def lognormal(size=None, loc=None, scale=None, c95=None, seed=None):
        """Sample from a log-normal distribution
        indicate with `loc` and `scale`, or `c95`

        Args:
            size (`int`): number of values to draw
            loc (`float`): center of log-normal distribution, default 0
            scale (`float`): log variance of the distribution, default 0
            c95 ([`float`, `float`]): 95% percentile of log-normal distribution
            seed: random seed

        Returns:
            a draw from distribution with given size
        """

        import numpy as np

        if c95 is None:
            if loc is None:
                loc = 0
            if scale is None:
                scale = 0
        else:
            c95 = np.log(np.array(c95))
            loc = (c95[0] + c95[1]) / 2
            scale = (c95[1] - c95[0]) / 3.92

        if seed is not None:
            np.random.seed(seed)

        if size is None:
            return np.exp(np.random.normal(loc=loc, scale=scale))
        else:
            return np.exp(np.random.normal(loc=loc, scale=scale, size=size))

    @staticmethod
    def uniform(low=None, high=None, size=None, seed=None):
        """Sample from a uniform distribution"""

        import numpy as np

        if seed is not None:
            np.random.seed(seed)
        if size is None:
            return np.random.uniform(low=low, high=high)
        else:
            return np.random.uniform(low=low, high=high, size=size)

    @staticmethod
    def compo_lognormal(size, loc=None, scale=None, c95=None, seed=None):
        """Sample a pool composition from a log-normal distribution
        indicate with `loc` and `scale`, or `c95`

        Example:
            scale = 0 means an evenly distributed pool with all components have relative abundance 1/size

        Args:
            size (`int`): size of the pool
            loc (`float`): center of log-normal distribution
            scale (`float`): log variance of the distribution
            c95 ([`float`, `float`]): 95% percentile of log-normal distribution
            seed: random seed

        """

        import numpy as np

        if c95 is None:
            if loc is None or scale is None:
                raise ValueError('Please indicate loc/scale or c95')
        else:
            c95 = np.log(np.array(c95))
            loc = (c95[0] + c95[1]) / 2
            scale = (c95[1] - c95[0]) / 3.92
        if seed is not None:
            np.random.seed(seed)

        q = np.exp(np.random.normal(loc=loc, scale=scale, size=size))
        return q / np.sum(q)


class PoolParamSimulator:
    """Collection of functions to simulate a set of parameters for a sequence pool

    Methods:

        - sample_from_ind_dist

        - sample_from_dataframe

    Returns:
        pd.DataFrame with columns of parameters and rows of simulated parameter for each sequence
    """

    @staticmethod
    def sample_from_ind_dist(p0, size=None, seed=None, **param_generators):
        """Simulate the pool parameter from individual draws of distribution

        Args:

            p0 (list-like, generator, or callable): at least need to indicate the initial pool composition

            size (int): number of unique sequences

            seed (int): global random seed to use in generation

            param_generators (kwargs): keyword generator to generate parameters

        """
        import numpy as np
        import pandas as pd

        def generate_params(param_input):
            from types import GeneratorType

            if isinstance(param_input, (list, np.ndarray, pd.Series)):
                return param_input
            elif isinstance(param_input, GeneratorType):
                # assume only generate one realization
                return [next(param_input) for _ in range(size)]
            elif callable(param_input):
                try:
                    # if there is a size parameter to pass
                    param_output = param_input(size=size)
                    if isinstance(param_output, (list, np.ndarray, pd.Series)):
                        return param_output
                    elif isinstance(param_output, GeneratorType):
                        return next(param_output)
                    else:
                        logging.error("Unknown input to draw a distribution value")
                        raise TypeError("Unknown input to draw a distribution value")
                except:
                    # if can not pass size, assume generate single samples
                    param_output = param_input()
                    if isinstance(param_output, GeneratorType):
                        return [next(param_input) for _ in range(size)]
                    elif isinstance(param_output, (float, int)):
                        return [param_input() for _ in range(size)]
            else:
                logging.error("Unknown input to draw a distribution value")
                raise TypeError('Unknown input to draw a distribution value')

        if seed is not None:
            np.random.seed(seed)

        results = pd.DataFrame(
            data={**{'p0': list(generate_params(p0))},
                  **{param: generate_params(gen) for param, gen in param_generators.items()}}
        )

        return results

    @classmethod
    def sample_from_dataframe(cls, df, size, replace=True, weights=None, seed=None):
        """Simulate parameter by resampling rows of a given data frame"""

        return df.sample(n=size, replace=replace, weights=weights, random_state=seed)


def simulate_counts(pool_size, c_list, N_list, p0=None,
                    kinetic_model=None, count_model=None, dna_amount_error=None,
                    sample_from_table=None, weights=None, replace=True,
                    reps=1, seed=None, note=None,
                    save_to=None, **param_generator):
    """Simulate a k-seq count data given kinetic and count model

    Args:
        pool_size (int): number of unique sequences
        c_list (list): a list of substrate concentrations for each sample, negative number means initial pool
        N_list (list): a list of total reads in for each sample
        p0 (list, generator, or callable returns generator): composition of initial pool
        kinetic_model(callable or ModelBase):
        count_model(callable or ModelBase):
        dna_amount_error (float or callable): a fixed Gaussian error with std. dev. as the float
            or any error function on the DNA amount
        sample_from_table (pd.DataFrame): optional to sample sequences from given table
        weights (list or str): weights/col of weight for sampling from table
        replace (bool): if sample with replacement
        reps (int): number of replicates for each c, N
        seed (int): global random seed to use
        save_to (path): path to the folder to save simulated results
        **param_generator: keyword arguments of list, generator or callable returns generator to draw parameters

    Returns:
        X (pd.DataFrame): c, n value for samples
        y (pd.DataFrame): sequence counts for sequence samples
        param_table (pd.DataFrame): table list the parameters of simulated pool
        seq_table (data.SeqTable): a SeqTable object stores all the data
    """

    from ..model import pool
    import pandas as pd
    if seed is not None:
        import numpy as np
        np.random.seed(seed)

    if kinetic_model is None:
        # default use BYO first-order model returns absolute amount
        from k_seq.model import kinetic
        kinetic_model = kinetic.BYOModel.amount_first_order
        logging.info('No kinetic model provided, use BYOModel.amount_first_order')
    if count_model is None:
        # default use MultiNomial
        from k_seq.model import count
        count_model = count.MultiNomial
        logging.info('No count model provided, use MultiNomial')

    if sample_from_table is None:
        param_table = PoolParamSimulator.sample_from_ind_dist(
            p0=p0,
            size=pool_size,
            **param_generator
        )
    else:
        param_table = PoolParamSimulator.sample_from_dataframe(
            df=sample_from_table,
            size=pool_size,
            replace=replace,
            weights=weights
        )

    param_table.index.name = 'seq'
    pool_model = pool.PoolModel(count_model=count_model,
                                kinetic_model=kinetic_model,
                                param_table=param_table)
    x = {}
    Y = {}
    dna_amount = {}
    for sample_ix, (c, n) in enumerate(zip(c_list, N_list)):
        if reps is None or reps == 1:
            dna_amount[f"s{sample_ix}"], Y[f"s{sample_ix}"] = pool_model.predict(c=c, N=n)
            x[f"s{sample_ix}"] = {'c': c, 'n': n}
        else:
            for rep in range(reps):
                dna_amount[f"s{sample_ix}-{rep}"], Y[f"s{sample_ix}-{rep}"] = pool_model.predict(c=c, N=n)
                x[f"s{sample_ix}-{rep}"] = {'c': c, 'n': n}
    # return x, Y, dna_amount, param_table
    x = pd.DataFrame.from_dict(x, orient='columns')
    Y = pd.DataFrame.from_dict(Y, orient='columns')
    dna_amount = pd.DataFrame.from_dict(dna_amount, orient='columns')
    dna_amount = dna_amount.sum(axis=0)
    if isinstance(dna_amount_error, float):
        import numpy as np
        dna_amount += np.random.normal(loc=0, scale=dna_amount_error, size=len(dna_amount))
    elif callable(dna_amount_error):
        dna_amount = dna_amount.apply(dna_amount_error)
    x.index.name = 'param'
    Y.index.name = 'seq'
    dna_amount.index.name = 'amount'

    from .seq_table import SeqTable

    seq_table = SeqTable(data_mtx=Y, x_values=x, note=note, grouper={'input': list(x.loc[x['c'] < 0].index),
                                                                     'reacted': list(x.loc[x['c'] < 0].index)})
    seq_table.add_total_dna_amount(dna_amount=dna_amount.to_dict())
    seq_table.table_abs_amnt = seq_table.dna_amount.apply(target=seq_table.table)
    from .transform import ReactedFractionNormalizer
    reacted_frac = ReactedFractionNormalizer(input_samples=list(x.loc[x['c'] < 0].index),
                                             reduce_method='median',
                                             remove_zero=True)
    seq_table.table_reacted_frac = reacted_frac.apply(seq_table.table_abs_amnt)
    from .filters import DetectedTimesFilter
    seq_table.table_seq_in_all_smpl_reacted_frac = DetectedTimesFilter(
        min_detected_times=seq_table.table_reacted_frac.shape[1]
    )(seq_table.table_reacted_frac)

    if save_to is not None:
        from pathlib import Path
        save_path = Path(save_to)
        if save_path.suffix == '':
            save_path.mkdir(parents=True, exist_ok=True)
            dna_amount.to_csv(f'{save_path}/dna_amount.csv')
            x.to_csv(f'{save_path}/x.csv')
            Y.to_csv(f'{save_path}/Y.csv')
            param_table.to_csv(f'{save_path}/truth.csv')
            seq_table.to_pickle(f"{save_path}/seq_table.pkl")
        else:
            logging.error('save_to should be a directory')
            raise TypeError('save_to should be a directory')

    return x, Y, dna_amount, param_table


# def reacted_frac_simulator(c_list, kinetic_model=None, percent_noise=0.2, pool_size=None,
#                            sample_from_table=None, weights=None, replace=True,
#                            reps=1, allow_zero=False, seed=None, save_to=None, **param_generator):
#     """Simulate reacted fraction of a pool of molecules (ribozymes), given substrate concentration, kinetic model
#
#     Args:
#         c_list (list): list of substrate concentration
#         kinetic_model (callable): kinetic model of the reaction
#         percent_noise (float or list): percent variance of Gaussian noise, if list, should have same shape as c_list,
#             default: 0.2
#         sample_from_table (pd.DataFrame): optional to sample sequences from given table
#         weights (list or str): weights/col of weight for sampling from table
#         replace (bool): if sample with replacement
#         reps (int): number of replicates for each c, N
#         allow_zero (bool): if allow reacted fraction to be zero after noise. If False, repeated sampling until a
#             positive reacted fraction is achieved, else bottomed by zero
#         seed (int): global random seed to use
#         save_to (path): path to the folder to save simulated results
#         **param_generator: keyword arguments of list, generator or callable returns generator to draw parameters
#
#     Returns:
#         x (pd.Series): c_list with replication
#         Y (pd.DataFrame): a table of reacted fraction
#         param_table (pd.DataFrame): a table of parameter truth
#     """
#
#     def add_noise(mean, noise, allow_zero=False):
#         """Add Gaussian noise on given mean
#         Note:
#             here is absolute noise, to inject percent noise, assign noise = mean * pct_noise
#
#         Args:
#             mean (float or list-like): mean values
#             noise (float or list-list): noise level, variance of Gaussian noise,
#                 if list-like should have same shape as mean
#             allow_zero (bool): if allow zero in noised inject data, if False, repeated sampling until non-negative value
#                 observed
#
#         Returns:
#             data_w_noise (float or np.ndarray): data with injected noise, same shape as mean
#         """
#         import numpy as np
#
#         y = np.random.normal(loc=mean, scale=noise)
#         if isinstance(mean, float):
#             while y < 0 and not allow_zero:
#                 y = np.random.normal(loc=y, scale=y_noise)
#             return max(y, 0)
#         else:
#             while (y < 0).any() and not allow_zero:
#                 y = np.random.normal(loc=y, scale=y_noise)
#             y[y < 0] = 0
#             return y
#
#     if kinetic_model is None:
#         # default use BYO first-order compositional model
#         from k_seq.model import kinetic
#         kinetic_model = kinetic.BYOModel.react_frac
#         print('No kinetic model provided, use BYOModel.amount_first_order')
#
#     if sample_from_table is None:
#
#         results = pd.DataFrame(
#             data={**{'p0': list(generate_params(p0))},
#                   **{param: generate_params(gen) for param, gen in param_generators.items()}}
#         )
#
#         return  results
#
#         param_table = sample_from_ind_dist(
#             seed=seed,
#             **param_generator
#         )
#     else:
#         param_table =
#             df=sample_from_table,
#             size=pool_size,
#             replace=replace,
#             weights=weights,
#             seed=seed
#         )
#
#     param_table.index.name = 'seq'
#
#     x_true = np.array(x_true)
#
#
#     if isinstance(params, list):
#         y_true = model(x_true, *params.values())
#     elif isinstance(params, dict):
#         y_true = model(x_true, **params)
#
#     if type(percent_noise) is float or type(percent_noise) is int:
#         y_noise = [percent_noise * y for y in y_true]
#     else:
#         y_noise = [tmp[0] * tmp[1] for tmp in zip(y_true, percent_noise)]
#
#     y_ = np.array([[add_noise(yt[0], yt[1], y_allow_zero) for yt in zip(y_true, y_noise)] for _ in range(replicates)])
#     if average:
#         return (x_true, np.mean(y_, axis=0))
#     else:
#         x_ = np.array([x_true for _ in range(replicates)])
#         return (x_.reshape(x_.shape[0] * x_.shape[1]), y_.reshape(y_.shape[0] * y_.shape[1]))
#
#
# def data_simulator_convergence_map(A_range, k_range, x, save_dir=None, percent_noise=0.0, func=func_default, replicates=5, A_res=100, k_res=100, A_log=False, k_log=True):
#     """
#
#     :param A_range: param k_range:
#     :param x: param save_dir:  (Default value = None)
#     :param percent_noise: Default value = 0.0)
#     :param func: Default value = func_default)
#     :param replicates: Default value = 5)
#     :param A_res: Default value = 100)
#     :param k_res: Default value = 100)
#     :param A_log: Default value = False)
#     :param k_log: Default value = True)
#     :param k_range:
#     :param save_dir:  (Default value = None)
#
#     """
#
#     if A_log:
#         A_values = np.logspace(np.log10(A_range[0]), np.log10(A_range[1]), A_res)
#     else:
#         A_values = np.linspace(A_range[0], A_range[1], A_res+1)[1:] # avoid A=0
#     if k_log:
#         k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[0]), k_res)
#     else:
#         k_values = np.linspace(k_range[0], k_range[1], k_res)
#
#     data_tensor = np.zeros((A_res, k_res, replicates, len(x)))
#     for A_ix in range(A_res):
#         for k_ix in range(k_res):
#             for rep in range(replicates):
#                 data_tensor[A_ix, k_ix, rep, :] = y_value_simulator(
#                     params=[A_values[A_ix], k_values[k_ix]],
#                     x_true=x,
#                     func=func,
#                     percent_noise=percent_noise,
#                     y_allow_zero=False,
#                     average=False
#                 )[1]
#     dataset_to_dump = {
#         'x': x,
#         'y_tensor': data_tensor
#     }
#     # if save_dir:
#     #     util.dump_pickle(dataset_to_dump, save_dir,
#     #                      log='Simulated dataset for convergence map of k ({}), A ({}), with x:{} and percent_noise:{}'.format(k_range, A_range, x, percent_noise),
#     #                      overwrite=True)
#     return dataset_to_dum

# def count_simulator(model, params, repeat=1, seed=None):
#     """Function to simulate a pool count with different parameters
#
#     Args:
#         model:
#         params:
#         repeat:
#         seed:
#
#     Returns:
#         pd.DataFrame contains counts table
#
#     """
#     import numpy as np
#     import pandas as pd
#
#     def run_model(param):
#         if isinstance(param, dict):
#             return model(**param)
#         elif isinstance(param, (list, tuple)):
#             return model(*param)
#         else:
#             return model(param)
#
#     # first parse params
#     if isinstance(params, list):
#         params = {f'sample_{int(idx)}': param for idx, param in enumerate(params)}
#
#     if seed is not None:
#         np.random.seed(seed)
#
#     result = {}   # {sample_name: counts}
#     for sample, param in params.items():
#         if repeat is None or repeat == 1:
#             result[sample] = run_model(param)
#         else:
#             for rep in range(repeat):
#                 result[f"{sample}-{rep}"] = run_model(param)
#     return pd.DataFrame.from_dict(result, orient='columns')
#
#
# class CountSimulator(object):
#     """Simulate pool counts given parameters (e.g. p0, k, A), and simulate counts for a given x values
#     """
#
#     def __init__(self, count_model, kinetic_model, count_params=None, kinetic_params=None,
#                  x_values=None, seed=None):
#         """Initialize a simulator with parameter table, count model, and parameters
#
#         Args:
#
#
#             kinetic_model (`Model` or callable): pool kinetic model, whose output is a list of abundance of pool members
#
#             kin_param (`dict`): contains parameter needed for
#
#             c_param:
#
#             c_model:
#             seed:
#         """
#         import numpy as np
#
#         self.k_model = k_model
#         self.k_param = k_param
#         self.c_model = c_model
#         self.c_param = c_param
#         self.seed = seed
#         if seed is not None:
#             np.random.seed(seed)
#
#     def get_data(self, k_param=None, c_param=None, N=1, seed=None):
#         """Return a size N data for each param set in k_param
#
#         Return: a dict or list of dict of
#             {'data': pd.DataFrame (sparse) of data,
#              'config':
#         """
#
#         def get_one_data(k_param, c_param):
#             comp = self.k_model(**k_param)
#             if np.sum(comp) != 1:
#                 comp = comp / np.sum(comp)
#             return self.c_model(comp, **c_param)
#
#         def get_one_config(k_param, c_param, N):
#             k_p = self.k_param.copy()
#             k_p.update(k_param)
#             c_p = self.c_param.copy()
#             c_p.update(c_param)
#
#             import pandas as pd
#
#             return {'data': pd.DataFrame(pd.SparseArray([get_one_data(k_p, c_p) for _ in range(N)]), dtype=int),
#                     'config': {'k_param': k_p, 'c_param': c_p}}
#
#         import numpy as np
#         if seed is not None:
#             np.random.seed(seed)
#
#         results = []
#
#     def to_pickle(self):
#         pass
#
#     def to_DataFrame(self):
#         pass
#
#     def to_csv(self):
#         pass



