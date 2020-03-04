"""
This module contains the methods used for bootstrap estimator
"""


def fitting(ydata, xdata, maxFold=None, fitMtd='trf', ciEst=True, alpha=0.479, func=None, bsDepth=2000):
    """
    :param ydata: the y-value (reacted fraction), 1-D array
    :param xdata: the x-value, same shape as ydata
    :param maxFold: if the maximum reacted fraction can exceed the indicated value, default None
    :param fitMtd: estimator method to use
    :param ciEst: If the confidence interval will be estimated by bootstrapping
    :param alpha: degradation factor, default is measured factor for BYO
    :param func: _get_mask to fit, if None will fit to default exponential function
    :return: seq with estimator results
    """
    from scipy.optimize import curve_fit
    import numpy as np

    def exp_func(x, A, k):
        return A * (1 - np.exp(- alpha * 90 * k * x))  # BYO degradation adjustment and 90 minutes

    if func is None:
        func = exp_func
    if maxFold is not None:
        ydata = np.array([min(yi, maxFold) for yi in ydata])
    valid = ~np.isnan(ydata)  # exclude missing data
    stat = {}
    try:
        initGuess = [np.random.random(), np.random.random()]
        params, pcov = curve_fit(func, xdata=xdata[valid], ydata=ydata[valid],
                                 method=fitMtd, bounds=([0, 0], [1., np.inf]), p0=initGuess)
        x_ = xdata[valid]
        y_ = ydata[valid]
        yHat = func(x_, params[0], params[1])
        res = y_ - yHat
        pctRes = res / yHat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((ydata[valid] - np.mean(ydata[valid])) ** 2)
        stat['r2'] = (1 - ss_res / ss_tot)
    except RuntimeError:
        params = [np.nan, np.nan]
    stat['params'] = [params[0], params[1], params[0]*params[1]]
    if ciEst:
        if (len(xdata[valid])>1)and(~np.isnan(params[0])): # bootstrap can be performed
            paramList = []
            for _ in range(bsDepth):
                pctResResampled = np.random.choice(pctRes, replace=True, size=len(pctRes))
                yToFit = yHat + yHat * pctResResampled
                try:
                    initGuess = (np.random.random(), np.random.random())
                    params, pcov = curve_fit(func, x_, yToFit,
                                             method=fitMtd, bounds=([0, 0], [1., np.inf]), p0=initGuess)
                except:
                    params = [np.nan, np.nan]
                    print(x_, yToFit)
                paramList.append([params[0], params[1], params[0] * params[1]])
            stat['stdevs'] = np.nanstd(paramList, axis=0, ddof=1)
            stat['ci95'] = np.array([np.nanpercentile(paramList, 2.5, axis=0), np.nanpercentile(paramList, 50, axis=0),
                                    np.nanpercentile(paramList, 97.5, axis=0), np.nanmean(paramList, axis=0)]).T
        else:
            stat['stdevs'] = [np.nan]
            stat['ci95'] = [np.nan]
    return stat