from abc import ABC, abstractmethod

class ModelBase(ABC):
    """base structure for a model
    A model contains following aspects:
      - model param estimation: need data, func, params, seed,
      - random variable generator, need params, seed, func
    """

    def __init__(self, **params):
        pass
    
    @abstractmethod
    def func(self, **params):
        """model function: params --> prediction"""
        pass
    
    @abstractmethod
    def predict(self, **params):
        """Function to predict model output"""
        pass

