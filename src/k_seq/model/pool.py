"""Model a kinetic reaction pool"""

from . import ModelBase
from ..utility.log import logging
from ..utility.func_tools import get_func_params
from inspect import isclass
import numpy as np


class PoolModel(ModelBase):
    """Model of a kinetic pool,

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
        """Initialize a pool model with given kinetic models and count_model
        Args:
            count_model (`ModelBase` or `callable`): model for sequencing counts
            kinetic_model (`ModalBase` or `callable`): model for pool kinetics, no react if not given
            **params:
        """

        def _static_pool(p0):
            """Static pool with no reaction"""
            return p0

        super().__init__()
        if kinetic_model is None:
            self.kinetic_model = _static_pool
        elif isclass(kinetic_model) and issubclass(kinetic_model, ModelBase):
            self.kinetic_model = kinetic_model.func
        elif callable(kinetic_model):
            self.kinetic_model = kinetic_model
        else:
            logging.error('model should be a ModelBase subclass or a callable', error_type=TypeError)

        if isclass(count_model) and issubclass(count_model, ModelBase):
            self.count_model = count_model.func
        elif callable(count_model):
            self.count_model = count_model
        else:
            logging.error('model should be a ModelBase subclass or a callable', error_type=TypeError)

        self.kinetic_params = get_func_params(self.kinetic_model, exclude_x=False)
        self.count_params = get_func_params(self.count_model, exclude_x=False)
        if param_table is not None:
            params.update({col: param_table[col] for col in param_table.columns})
        self.params = params
        self.note = note

    def func(self, **params):
        """Draw counts from given parameters

        Returns:
            output sum from kinetic model: reacted amount if not normalized
            counts for each sequence
        """

        kinetic_params = {key: item for key, item in params.items() if key in self.kinetic_params}
        count_params = {key: item for key, item in params.items() if key in self.count_params}

        pt = self.kinetic_model(**kinetic_params)
        return np.sum(pt), self.count_model(pt / np.sum(pt), **count_params)

    def predict(self, **params):
        """Wrapper over _get_mask, can accept parameters to overwrite current ones if exist"""
        params = {**self.params, **params}
        return self.func(**params)

    def __call__(self, **params):
        return self.predict(**params)
