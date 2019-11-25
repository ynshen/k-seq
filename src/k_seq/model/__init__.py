
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

    def __init__(self, **params):
        pass

    def func(self, **params):
        """model function: params --> prediction"""
        pass

    def predict(self, **params):
        """Function to predict model output"""
        pass

    def nloglikelihood(self):
        """negative log-likelihood value for each observation, parameter --> nloglikelihood"""
        pass

