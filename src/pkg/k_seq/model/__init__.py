
class ResultBase(object):
    """base structure for a model estimation results
    For now, use Results model in `statsmodel`
    """


class ModelBase(object):
    """base structure for a model
    A model contains following aspects:
      - model param estimation: need data, func, params, seed,
      - random variable generator, need params, seed, func
    """

    def __init__(self, X=None, y=None, seed=23, **params):
        self.params = params    # a dictionary of parameters
        self.seed = seed    # fix seed for repeatability
        self.result = None    # space holder for mle estimation results
        self.X = X
        self.y = y

    def data(self, y, X=None):
        self.X = X
        self.y = y

    def func(self):
        """model function: params --> prediction"""
        pass

    def predict(self, **params):
        """use given parameter or predicted parameter
        todo: to fill
        """
        if params == {}:
            if self.result is None:
                return ValueError('No estimated parameters, please indicate parameter')
            else:
                pass

    def nloglikelihood(self):
        """negative log-likelihood value for each observation, parameter --> nloglikelihood"""
        pass

