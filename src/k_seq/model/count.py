"""Module model the counts

Notes:
    kinetic models were currently implemented as callable function. It might migrate to `Model` subclass for
      storing parameters, set required parameters, etc
"""

import numpy as np
import pandas as pd
from ..utility.log import logging
from ..utility.func_tools import is_numeric


def multinomial(p, N, seed=None):
    """Multinomial distribution for a given probability p and total number of draws"""
    if seed is not None:
        np.random.seed(seed)

    if np.sum(p) != 1:
        p = np.array(p) / np.sum(p)

    from scipy.stats import multinomial
    if isinstance(N, (list, np.ndarray, pd.Series)):
        return np.array([multinomial.rvs(n=int(n), p=p) for n in N])
    elif is_numeric(N):
        return multinomial.rvs(n=int(N), p=p)
    else:
        logging.error("Unknown N type", error_type=TypeError)
