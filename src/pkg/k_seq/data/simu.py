
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
                        raise TypeError("Unknown input to draw a distribution value")
                except:
                    # if can not pass size, assume generate single samples
                    param_output = param_input()
                    if isinstance(param_output, GeneratorType):
                        return [next(param_input) for _ in range(size)]
                    elif isinstance(param_output, (float, int)):
                        return [param_input() for _ in range(size)]
            else:
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

    # def generate(self, seed=None):
    #     """Return a generated parameter
    #
    #     Return: a dict or list of dict of
    #         {'data': pd.DataFrame (sparse) of data,
    #          'config':
    #     """
    #     import numpy as np
    #
    #     if seed is None:
    #         seed = self.seed
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     self.results = {sample: self.model(**param) for sample, param in self.parameters.items()}
    #
    # def to_pickle(self):
    #     pass
    #
    # def to_DataFrame(self, sparse=True, dtype='float', seed=None):
    #     import pandas as pd
    #     import numpy as np
    #
    #     if seed is None:
    #         seed = self.seed
    #
    #     if self.results is None:
    #         self.generate(seed=seed)
    #
    #     if sparse:
    #         if dtype.lower() in ['int', 'd']:
    #             dtype = pd.SparseDtype('int', fill_value=0)
    #         elif dtype.lower() in ['float', 'f']:
    #             dtype = pd.SparseDtype('float', fill_value=0.0)
    #         return pd.DataFrame(self.results).astype(dtype)
    #     else:
    #         if dtype.lower() in ['int', 'd']:
    #             dtype = np.int
    #         elif dtype.lower() in ['float', 'f']:
    #             dtype = np.float
    #         return pd.DataFrame(self.results, dtype=dtype)
    #
    # def to_numpy(self, seed=None):
    #     import numpy as np
    #
    #     if seed is None:
    #         seed = self.seed
    #     if self.results is None:
    #         self.generate(seed=seed)
    #     return np.array([value for value in self.results.values()])
    #
    # def to_csv(self):
    #     pass


def count_simulator(model_func, params, repeat=1, seed=None):
    """Function to simulate a pool count with different parameters

    Args:
        model_func:
        params:
        repeat:
        seed:

    Returns:
        pd.DataFrame contains counts table

    """
    import numpy as np
    import pandas as pd

    def run_model(param):
        if isinstance(param, dict):
            return model_func(**param)
        elif isinstance(param, (list, tuple)):
            return model_func(*param)
        else:
            return model_func(param)

    # first parse params
    if isinstance(params, list):
        params = {f'sample_{int(idx)}': param for idx, param in enumerate(params)}

    if seed is not None:
        np.random.seed(seed)

    result = {}   # {sample_name: counts}
    for sample, param in params.items():
        if repeat is None or repeat == 1:
            result[sample] = run_model(param)
        else:
            for rep in range(repeat):
                result[f"{sample}-{rep}"] = run_model(param)
    return pd.DataFrame.from_dict(result, orient='columns')


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



# ----------------------- Below from legacy ------------------------
#
# # import util
#
# def func_default(x, A, k):
#     """
#
#     :param x: param A:
#     :param k:
#     :param A:
#
#     """
#     return A * (1 - np.exp(-0.479 * 90 * k * x))
#
# def y_value_simulator(params, x_true, model=None, percent_noise=0.2,
#                       replicates=1, y_allow_zero=False, average=False):
#     """Simulator to simulate y value of a function, given x and noise level
#
#     :param params: a list of parameters used in the function
#     :param x_true: a list of true values for x
#     :param func: callable, function used to fit, default Abe's BYO estimator function
#     :param percent_noise: percent standard deviation of normal noise, real value or a list of real value with same order
#                           of x_true for its corresponding y (Default value = 0.2)
#     :param replicates: int, number of replicates for each x value (Default value = 1)
#     :param y_allow_zero: boolean, if True, 0 is allowed for y; if False, resample until y_value larger than 0 (Default value = False)
#     :param average: boolean, if doing average on each x_true point for simulated y (Default value = False)
#     :returns: x, y) two 1-d numpy array with same order and length
#
#     """
#     def add_noise(y, y_noise, y_allow_zero=False):
#         """
#
#         :param y: param y_noise:
#         :param y_allow_zero: Default value = False)
#         :param y_noise:
#
#         """
#         y = np.random.normal(loc=y, scale=y_noise)
#         while y < 0 and not y_allow_zero:
#             y = np.random.normal(loc=y, scale=y_noise)
#         return max(y, 0)
#
#     x_true = np.array(x_true)
#
#     if not model:
#         func = func_default
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
#     return dataset_to_dump
