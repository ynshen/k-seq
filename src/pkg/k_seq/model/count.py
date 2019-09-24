
from . import ModelBase


def _get_mle_estimator(model, nll_func):
    from statsmodels.base.model import GenericLikelihoodModel

    class CustomizedEstimator(GenericLikelihoodModel):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.nparams = len(model._param_name)

        def nloglikeobs(self, params):
            return nll_func(params)

    return CustomizedEstimator


class MultiNomial(object):

    def __init__(self, y=None, p=None, N=None):
        """MultiNomial model of pool counts
        if y is given, it is saved as data
        if p and N are given, they are used for simulation
        """

        import numpy as np

        self.y = np.array(y)
        self._param_name = ['p', 'N']
        self.p = p
        self.N = N
        self.mle_estimator = None
        self.mle_res = None

    @property
    def params(self):
        return {
            param: getattr(self, param, None) for param in self._param_name
        }

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        import numpy as np
        if value is None:
            pass
        else:
            if np.sum(value) != 1:
                value = np.array(value) / np.sum(value)
            self._p = value

    @property
    def X(self):
        return self.y.sum(-1)

    @X.setter
    def X(self, value):
        raise AttributeError('X is the number of trials, can be only inferred from y')

    def predict(self, N=None, p=None, size=1, seed=None):
        """Method as prediction model, single observation"""
        import numpy as np

        if seed is not None:
            np.random.seed(seed)

        N = N if N is not None else self.N
        p = p if p is not None else self.p
        if size == 1:
            return self.func(p=p, N=N)
        else:
            return np.array([self.func(p=p, N=N) for _ in range(size)])

    @staticmethod
    def func(p, N):
        from scipy.stats import multinomial
        return multinomial.rvs(n=N, p=p)

    @staticmethod
    def negloglikelihood(y, N, p):
        """Method to calculate likelihood
        defined on p, given data

        Return:
            a list of negative log-likelilhood value for each input
        """
        from scipy.stats import multinomial
        import numpy as np
        if np.sum(p) != 1:
            p = np.array(p)/np.sum(p)
        logpmf = multinomial.logpmf(x=y, n=N, p=p)
        if np.inf in logpmf or -np.inf in logpmf or np.nan in logpmf:
            print('Inf observed in logpmf with\nN={}\ny={}\np={}'.format(N, y, p))
            print(logpmf)
            return np.ones()
        else:
            return -logpmf

    def mle_fit(self, y=None, start_param=None, maxiter=10000, maxfun=500, inplace=True, **kwargs):
        import numpy as np

        if y is None:
            y = self.y
        else:
            y = np.array(y)

        def nll_fn(p):
            return self.negloglikelihood(p=p, y=y, N=y.sum(-1))

        mle_estimator = _get_mle_estimator(self, nll_func=nll_fn)(endog=y)

        if start_param is None:
            start_param = np.ones(y.shape[1])/y.shape[1]

        if inplace:
            self.mle_res = mle_estimator.fit(start_params=start_param, maxiter=maxiter, maxfun=maxfun, **kwargs)
            self.mle_estimator = mle_estimator
        else:
            return mle_estimator.fit(start_params=start_param, maxiter=maxiter, maxfun=maxfun, **kwargs)

    def mle_fit_scipy(self, y=None, start_param=None, maxiter=10000, method=None):
        """Directly use `scipy.optimize` as backend to minimize the negative log-likelihood function

        It is constrained minimization at large scale > 10^5 parameters, methods to choose
        - trust-constr: Trust-Region Constrained Algorithm, based on EQSQP and TRIP

        """
        import numpy as np

        if y is None:
            y = self.y
        else:
            y = np.array(y)

        def nll_fn(p):
            import numpy as np
            nll = self.negloglikelihood(p=p, y=y, N=y.sum(-1))
            return np.sum(nll)/len(nll)

        from scipy.optimize import Bounds, LinearConstraint, minimize
        lb = 0.01 / np.max(y.sum(-1))
        bounds = Bounds(np.ones(shape=y.shape[-1]) * lb, np.ones(shape=y.shape[-1]))

        simplex = LinearConstraint(np.ones(shape=y.shape[-1]), 1, 1)
        if start_param is None:
            start_param = np.mean(y, axis=0)
            start_param = start_param/start_param.sum()
        result = minimize(nll_fn, x0=start_param, method='slsqp', constraints=simplex,
                          options={'verbose':1}, bounds=bounds)
        return result



def count_pois(p, N):
    from scipy.stats import poisson
    import numpy as np
    p = np.array(p)
    if np.sum(p) != 1:
        p = p / np.sum(p)
    return poisson.rvs(p * N)


def zero_inflated_pois(p, N, alpha):
    pass


def count_nbin(mu, s):

    def convert_params(mu, s):
        """
        Convert mean/dispersion parameterization (mu, s) of a negative binomial that
          mean = mu
          var = mu + s * mu ** 2
        to the ones scipy supports NB(n, p) ~ number of success, where
          n: num. of failure
          p: prob of single failure
         where:
           mean: (1-p)n/p
           var: (1-p)n/(p**2)

        See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        """
        var = mu + s * mu ** 2
        p = mu / var
        n = 1 / s
        return n, p

    from scipy.stats import nbinom

    return nbinom.rvs(*convert_params(mu, s))



