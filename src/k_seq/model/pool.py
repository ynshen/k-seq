from . import ModelBase


class PoolModel(ModelBase):
    """Model of a kinetic pool

    Attributes:
        kinetic_model (`callable`): input initial pool with parameter and return a reacted pool composition
        count_model (`callable`): input pool composition and return a list of counts given total counts or params
        kinetic_params (list of str): list of parameter names for kinetic model
        count_params (list of str): list of parameter names for count model
        note (str): note for the model
    """

    def __repr__(self):
        return f"Model for a kinetic pool with\n" \
               f"\tkinetic model:{self.kinetic_model}\n" \
               f"\tcount model:{self.count_model}\n" \
               f"\tnote: {self.note}"

    def __init__(self, count_model, kinetic_model=None, param_table=None, note=None, **params):
        """Construct a pool model with given kinetic models and count_model
        Args:
            count_model (`ModelBase` or `callable`): model for sequencing counts
            kinetic_model (`ModalBase` or `callable`): model for pool kinetics, no react if not given
            **params:
        """

        def _static_pool(p0):
            """Static pool with no reaction"""
            return p0

        from ..utility.func_tools import get_func_params
        import pandas as pd

        super().__init__()
        if kinetic_model is None:
            self.kinetic_model = _static_pool
        else:
            try:
                if issubclass(kinetic_model, ModelBase):
                    kinetic_model = kinetic_model.func
            except:
                pass
            if callable(kinetic_model):
                self.kinetic_model = kinetic_model
            else:
                raise TypeError('model should be a ModelBase subclass or a callable')
        try:
            if issubclass(count_model, ModelBase):
                count_model = count_model.func
        except:
            pass
        if callable(count_model):
            self.count_model = count_model
        else:
            raise TypeError('model should be a ModelBase subclass or a callable')

        self.kinetic_params = get_func_params(self.kinetic_model, exclude_x=False)
        self.count_params = get_func_params(self.count_model, exclude_x=False)
        if param_table is not None:
            params.update({col: param_table[col] for col in param_table.columns})
        self.params = pd.DataFrame.from_dict(params, orient='columns')
        self.note = note

    def func(self, **params):
        """Draw counts from given parameters

        Returns:
            output sum from kinetic model: reacted amount if not normalized
            counts

        """
        import numpy as np

        kinetic_params = {key: item for key, item in params.items() if key in self.kinetic_params}
        count_params = {key: item for key, item in params.items() if key in self.count_params}

        pt = self.kinetic_model(**kinetic_params)
        return np.sum(pt), self.count_model(pt / np.sum(pt), **count_params)

    def predict(self, **params):
        """Wrapper over func, can accept parameters to overwrite current ones if exist"""
        params = {**self.params, **params}
        return self.func(**params)

    def __call__(self, **params):
        return self.predict(**params)

######################## Belows are from legacy ####################################
# def pool_count_models(p, k_model, k_param, c_model, c_param):
#     import numpy as np
#
#     if not isinstance(p, np.ndarray):
#         init_p = np.array(p)
#
#     if np.sum(init_p) != 1:
#         init_p = init_p / np.sum(init_p)
#
#     if isinstance(k_model, str):
#         from . import kinetic
#
#         if k_model in kinetic.__dict__.keys():
#             k_model = kinetic.__dict__[k_model]
#         else:
#             raise ValueError('Model {} not implemented'.format(k_model))
#     elif not callable(k_model):
#         raise TypeError('Custom model should be a callable')
#
#     if isinstance(c_model, str):
#         from . import count
#
#         if c_model in count.__dict__.keys():
#             c_model = count.__dict__[c_model]
#         else:
#             raise ValueError('Model {} not implemented'.format(c_model))
#     elif not callable(c_model):
#         raise TypeError('Custom model should be a callable')
#
#     reacted_frac = k_model(**k_param)
#     post_p = init_p * reacted_frac
#
#     return c_model(post_p, **c_param)


# class PoolModel(object):
#
#     def __init__(self, kinetic_model, count_model, x=None, y=None, p0=None, k_params=None, c_params=None):
#         """A pool model integrate kinetic model and count model
#         if x, y are given, it is saved as data
#           x: x values for each 1-dim of y value
#           y: count value with shape (sample_num, uniq_seq_num)
#         if p0, k_params, c_params are given, they are used for simulation
#
#         Args:
#             kinetic_model (`Model` or callable): return the relative abundance given
#         """
#
#         import numpy as np
#
#         self.y = np.array(y)
#         self._param_name = ['p', 'N']
#         self.p = p
#         self.N = N
#         self.mle_estimator = None
#         self.mle_res = None
#
#     @property
#     def params(self):
#         return {
#             param: getattr(self, param, None) for param in self._param_name
#         }
#
#     @property
#     def p(self):
#         return self._p
#
#     @p.setter
#     def p(self, value):
#         import numpy as np
#         if value is None:
#             pass
#         else:
#             if np.sum(value) != 1:
#                 value = np.array(value) / np.sum(value)
#             self._p = value
#
#     @property
#     def X(self):
#         return self.y.sum(-1)
#
#     @X.setter
#     def X(self, value):
#         raise AttributeError('X is the number of trials, can be only inferred from y')
#
#     def predict(self, N=None, p=None, uniq_seq_num=1, seed=None):
#         """Method as prediction model, single observation"""
#         import numpy as np
#
#         if seed is not None:
#             np.random.seed(seed)
#
#         N = N if N is not None else self.N
#         p = p if p is not None else self.p
#         if uniq_seq_num == 1:
#             return self.func(p=p, N=N)
#         else:
#             return np.array([self.func(p=p, N=N) for _ in range(uniq_seq_num)])
#
#     @staticmethod
#     def func(p, N):
#         from scipy.stats import multinomial
#         return multinomial.rvs(n=N, p=p)
#
#     @staticmethod
#     def negloglikelihood(y, N, p):
#         """Method to calculate likelihood
#         defined on p, given data
#
#         Return:
#             a list of negative log-likelilhood value for each input
#         """
#         from scipy.stats import multinomial
#         import numpy as np
#         if np.sum(p) != 1:
#             p = np.array(p)/np.sum(p)
#         logpmf = multinomial.logpmf(x=y, n=N, p=p)
#         if np.inf in logpmf or -np.inf in logpmf or np.nan in logpmf:
#             print('Inf observed in logpmf with\nN={}\ny={}\np={}'.format(N, y, p))
#             print(logpmf)
#             return np.ones()
#         else:
#             return -logpmf
#
#     def mle_fit(self, y=None, start_param=None, maxiter=10000, maxfun=500, inplace=True, **kwargs):
#         import numpy as np
#
#         if y is None:
#             y = self.y
#         else:
#             y = np.array(y)
#
#         def nll_fn(p):
#             return self.negloglikelihood(p=p, y=y, N=y.sum(-1))
#
#         mle_estimator = _get_mle_estimator(self, nll_func=nll_fn)(endog=y)
#
#         if start_param is None:
#             start_param = np.ones(y.shape[1])/y.shape[1]
#
#         if inplace:
#             self.mle_res = mle_estimator.fit(start_params=start_param, maxiter=maxiter, maxfun=maxfun, **kwargs)
#             self.mle_estimator = mle_estimator
#         else:
#             return mle_estimator.fit(start_params=start_param, maxiter=maxiter, maxfun=maxfun, **kwargs)
#
#     def mle_fit_scipy(self, y=None, start_param=None, maxiter=10000, method=None):
#         """Directly use `scipy.optimize` as backend to minimize the negative log-likelihood function
#
#         It is constrained minimization at large scale > 10^5 parameters, methods to choose
#         - trust-constr: Trust-Region Constrained Algorithm, based on EQSQP and TRIP
#
#         """
#         import numpy as np
#
#         if y is None:
#             y = self.y
#         else:
#             y = np.array(y)
#
#         def nll_fn(p):
#             import numpy as np
#             nll = self.negloglikelihood(p=p, y=y, N=y.sum(-1))
#             return np.sum(nll)/len(nll)
#
#         from scipy.optimize import Bounds, LinearConstraint, minimize
#         lb = 0.01 / np.max(y.sum(-1))
#         bounds = Bounds(np.ones(shape=y.shape[-1]) * lb, np.ones(shape=y.shape[-1]))
#
#         simplex = LinearConstraint(np.ones(shape=y.shape[-1]), 1, 1)
#         if start_param is None:
#             start_param = np.mean(y, axis=0)
#             start_param = start_param/start_param.sum()
#         result = minimize(nll_fn, x0=start_param, method='slsqp', constraints=simplex,
#                           options={'verbose':1}, bounds=bounds)
#         return result