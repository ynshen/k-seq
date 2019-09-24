
def first_order(x, k, A, alpha, t):
    """Base first order kinetic model
    Args:
        x (`float` or np.array): initial amount(s), could be composition of a pool
        k (`float` or np.array): kinetic coefficient(s)
        A (`float` or np.array): ratio of active molecules
        alpha (`float`): degradation parameter for substrates
        t (`float`): reaction time
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * x))


def first_order_w_slope(x, k, A, alpha, t, b):
    """Base first order kinetic model
    Args:
        x (`float` or np.array): initial amount(s), could be composition of a pool
        k (`float` or np.array): kinetic coefficient(s)
        A (`float` or np.array): ratio of active molecules
        alpha (`float`): degradation parameter for substrates
        t (`float`): reaction time
        b (`float` or np.array): slopes
    """
    import numpy as np

    return A * (1 - np.exp(- alpha * t * k * x)) + b


def byo_model(x, k, A, b=None, slope=False):
    if slope:
        return first_order_w_slope(x=x, k=k, A=A, alpha=0.479, t=90, b=b)
    else:
        return first_order(x=x, k=k, A=A, alpha=0.479, t=90)