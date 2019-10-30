
class DistGenerators:
    """A collection of random value generators from preset distributions


    Available distributions:

        - lognormal

        - uniform

        - compo_lognormal
    """

    def __init__(self):
        pass

    @staticmethod
    def lognormal(loc=None, scale=None, c95=None, seed=None, n=None):
        """Sample from a log-normal distribution
        indicate with `loc` and `scale`, or `c95`

        Args:

           n (`int`): number of values to draw

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

        if n is None:
            while True:
                yield np.exp(np.random.normal(loc=loc, scale=scale))
        else:
            while True:
                yield np.exp(np.random.normal(loc=loc, scale=scale, size=n))

    @staticmethod
    def uniform(low=None, high=None, n=None):
        """Sample from a uniform distribution"""

        import numpy as np

        if n is None:
            while True:
                yield np.random.uniform(low=low, high=high)
        else:
            while True:
                yield np.random.uniform(low=low, high=high, size=n)

    @staticmethod
    def compo_lognormal(size, loc=None, scale=None, c95=None, seed=None, n=None):
        """Sample a pool composition from a log-normal distribution

        indicate with `loc` and `scale`, or `c95`

        Example:

            scale = 0 means an evenly distributed pool with all components have relative abundance 1/size

        Args:

            size (`int`): size of the pool

            n (`int`): number of values to draw

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

        while True:
            if n is None:
                q = np.exp(np.random.normal(loc=loc, scale=scale))
            else:
                q = np.exp(np.random.normal(loc=loc, scale=scale, size=n))
            yield q / np.sum(q)


class ParamSimulator(object):
    """Simulate a set of parameters for a given model


    Attributes:
        todo: add attributes

    Methods:

        todo: add methods

    """
    def __init__(self, parameters, model, repeat=1, seed=23, **fixed_params):
        """
        Initialize the simulator with model and a list of parameter pass to model for simulation
        Args:
            parameters (`dict`): parameters should have one of following formats:
                [{param:value} for each sample]
                    Sample will be named as 'S{sample_num}-{replicate_num}'
                    any kwargs will be assigned to all samples
                {sample_name}:{param:value} for each sample}
                    any kwargs will be assigned to all samples

            model (`ModelBase` or callable): model to feed parameter
            repeat (`int`): number of replicates for each parameter set
            seed (`int`): random seed to fix in simulation for repeatability
        """
        import numpy as np

        from ..model import ModelBase

        if isinstance(model, ModelBase):
            self.model = model.func
        elif callable(model):
            self.model = model
        else:
            raise TypeError('model should be a subclass of ModelBase or a callable')

        # determine number of samples to simulate from parameters
        if isinstance(parameters, list):
            parameters = {f'S{ix}': parameter for ix, parameter in enumerate(parameters)}

        def expand_param(params, r):
            temp = {}
            for sample_key, sample_param in params.items():
                temp.update({f'{sample_key}-{ix}': sample_param for ix in range(r)})
            return temp

        if repeat > 1:
            parameters = expand_param(parameters, repeat)
        self.parameters = {sample_name: {**fixed_params, **parameter_dict}
                           for sample_name, parameter_dict in parameters.items()}
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.results = None

    @classmethod
    def from_dataframe(cls):
        pass

    def generate(self, seed=None):
        """Return a generated parameter

        Return: a dict or list of dict of
            {'data': pd.DataFrame (sparse) of data,
             'config':
        """
        import numpy as np

        if seed is None:
            seed = self.seed
        if seed is not None:
            np.random.seed(seed)

        self.results = {sample: self.model(**param) for sample, param in self.parameters.items()}

    def to_pickle(self):
        pass

    def to_DataFrame(self, sparse=True, dtype='float', seed=None):
        import pandas as pd
        import numpy as np

        if seed is None:
            seed = self.seed

        if self.results is None:
            self.generate(seed=seed)

        if sparse:
            if dtype.lower() in ['int', 'd']:
                dtype = pd.SparseDtype('int', fill_value=0)
            elif dtype.lower() in ['float', 'f']:
                dtype = pd.SparseDtype('float', fill_value=0.0)
            return pd.DataFrame(self.results).astype(dtype)
        else:
            if dtype.lower() in ['int', 'd']:
                dtype = np.int
            elif dtype.lower() in ['float', 'f']:
                dtype = np.float
            return pd.DataFrame(self.results, dtype=dtype)

    def to_numpy(self, seed=None):
        import numpy as np

        if seed is None:
            seed = self.seed
        if self.results is None:
            self.generate(seed=seed)
        return np.array([value for value in self.results.values()])

    def to_csv(self):
        pass


class CountSimulator(object):
    """Given pool initial relative abundance, kinetic model, and parameters to simulate a pool"""

    def __init__(self, seq_n=1e5, samplers=None, kin_model=None, kin_param=None, c_param=None, c_model=None, seed=23):
        """

        Args:
            kin_model (`Model` or callable): kinetic model, output should be a list of abundance of pool members
            kin_param (`dict`): contains parameter needed for
            c_param:
            c_model:
            seed:
        """
        import numpy as np

        self.k_model = k_model
        self.k_param = k_param
        self.c_model = c_model
        self.c_param = c_param
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def get_data(self, k_param=None, c_param=None, N=1, seed=None):
        """Return a size N data for each param set in k_param

        Return: a dict or list of dict of
            {'data': pd.DataFrame (sparse) of data,
             'config':
        """

        def get_one_data(k_param, c_param):
            comp = self.k_model(**k_param)
            if np.sum(comp) != 1:
                comp = comp / np.sum(comp)
            return self.c_model(comp, **c_param)

        def get_one_config(k_param, c_param, N):
            k_p = self.k_param.copy()
            k_p.update(k_param)
            c_p = self.c_param.copy()
            c_p.update(c_param)

            import pandas as pd

            return {'data': pd.DataFrame(pd.SparseArray([get_one_data(k_p, c_p) for _ in range(N)]), dtype=int),
                    'config': {'k_param': k_p, 'c_param': c_p}}

        import numpy as np
        if seed is not None:
            np.random.seed(seed)

        results = []

    def to_pickle(self):
        pass

    def to_DataFrame(self):
        pass

    def to_csv(self):
        pass



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
