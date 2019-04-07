import numpy as np
import util

def func_default(x, A, k):
    return A * (1 - np.exp(-0.479 * 90 * k * x))

def y_value_simulator(params, x_true, func=None, percent_noise=0.2,
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

    x_true = np.array(x_true)

    if not func:
        func = func_default
    y_true = func(x_true, *params)
    if type(percent_noise) is float or type(percent_noise) is int:
        y_noise = [percent_noise * y for y in y_true]
    else:
        y_noise = [tmp[0] * tmp[1] for tmp in zip(y_true, percent_noise)]

    y_ = np.array([[add_noise(yt[0], yt[1], y_allow_zero) for yt in zip(y_true, y_noise)] for _ in range(replicates)])
    if average:
        return (x_true, np.mean(y_, axis=0))
    else:
        x_ = np.array([x_true for _ in range(replicates)])
        return (x_.reshape(x_.shape[0] * x_.shape[1]), y_.reshape(y_.shape[0] * y_.shape[1]))


def data_simulator_convergence_map(A_range, k_range, x, save_dir=None, percent_noise=0.0, func=func_default, replicates=5, A_res=100, k_res=100, A_log=False, k_log=True):

    if A_log:
        A_values = np.logspace(np.log10(A_range[0]), np.log10(A_range[1]), A_res)
    else:
        A_values = np.linspace(A_range[0], A_range[1], A_res+1)[1:] # avoid A=0
    if k_log:
        k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[0]), k_res)
    else:
        k_values = np.linspace(k_range[0], k_range[1], k_res)

    data_tensor = np.zeros((A_res, k_res, replicates, len(x)))
    for A_ix in range(A_res):
        for k_ix in range(k_res):
            for rep in range(replicates):
                data_tensor[A_ix, k_ix, rep, :] = y_value_simulator(
                    params=[A_values[A_ix], k_values[k_ix]],
                    x_true=x,
                    func=func,
                    percent_noise=percent_noise,
                    y_allow_zero=False,
                    average=False
                )[1]
    dataset_to_dump = {
        'x': x,
        'y_tensor': data_tensor
    }
    if save_dir:
        util.dump_pickle(dataset_to_dump, save_dir,
                         log='Simulated dataset for convergence map of k ({}), A ({}), with x:{} and percent_noise:{}'.format(k_range, A_range, x, percent_noise),
                         overwrite=True)
    return dataset_to_dump
