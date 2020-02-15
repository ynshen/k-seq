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


class MultiNomial(ModelBase):
    """todo: many redundant function, clean it up, estimator should be removed"""

    def __init__(self, y=None, p=None, N=None, **params):
        """MultiNomial model of pool counts
        if y is given, it is saved as data
        if p and N are given, they are used for simulation
        """
        super().__init__()
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
        try:
            return multinomial.rvs(n=N, p=p)
        except ValueError:
            print('ValueError observed for:')
            print(N)
            print(p)

    def __call__(self, N=None, p=None, size=1, seed=None):
        return self.predict(N=N, p=p, size=seed, seed=seed)



