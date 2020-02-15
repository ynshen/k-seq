"""Module model the counts"""

import numpy as np
import pandas as pd
from ..utility.log import logging


def multinomial(p, N, seed=None):
    """Multinomial distribution for a given probability p and total number of draws"""
    if seed is not None:
        np.random.seed(seed)

    if np.sum(p) != 1:
        p = np.array(p) / np.sum(p)

    from scipy.stats import multinomial
    if isinstance(N, (list, np.ndarray, pd.Series)):
        return np.array([multinomial.rvs(n=n, p=p) for n in N])
    elif isinstance(N, int):
        return multinomial.rvs(n=N, p=p)
    else:
        logging.error("Unknown N type", error_type=TypeError)
