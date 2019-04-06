import numpy as np

def func_default(x, params):
    A, k = params
    return A * (1 - np.exp(-0.479 * 90 * k * x))

def y_value_simulator(params, x_true, func=None, percent_noise=0.1,
                      replicates=1, y_allow_zero=False, average=False):
    """
    Simulator to simulate y value of a function, given x and noise level
    :param params: a list of parameters used in the function
    :param x_true: a list of true values for x
    :param func: callable, function used to fit, default Abe's BYO fitting function
    :param percent_noise: percent standard deviation of normal noise, real value or a list of real value with same order
                          of x_true for its corresponding y
    :param replicates: int, number of replicates for each x value
    :param y_allow_zero: boolean, if True, 0 is allowed for y; if False, resample until y_value larger than 0
    :param average: boolean, if doing average on each x_true point for simulated y
    :return: (x, y) two 1-d numpy array with same order and length
    """
    def add_noise(y, y_noise, y_allow_zero=False):
        y = np.random.normal(loc=y, scale=y_noise)
        while y < 0 and not y_allow_zero:
            y = np.random.normal(loc=y, scale=y_noise)
        return max(y, 0)

    np.random.seed(23)
    x_true = np.array(x_true)

    if not func:
        func = func_default
    y_true = func(x_true, params)
    if type(noise) is float or type(noise) is int:
        y_noise = [noise * y for y in y_true]
    else:
        y_noise = [tmp[0] * tmp[1] for tmp in zip(y_true, percent_noise)]

    y_ = np.array([[add_noise(yt[0], yt[1], y_allow_zero) for yt in zip(y_true, y_noise)] for _ in range(replicates)])
    if average:
        return (x_true, np.mean(y_, axis=0))
    else:
        x_ = np.array([x_true for _ in range(replicates)])
        return (x_.reshape(x_.shape[0] * x_.shape[1]), y_.reshape(y_.shape[0] * y_.shape[1]))



