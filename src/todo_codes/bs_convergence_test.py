"""Legacy code of bootstrap method convergence"""

import util
import time
import multiprocessing as mp
import numpy as np


def fitting(seq, xdata, maxFold=None, fitMtd='trf', ciEst=True, func=None, bsDepth=2000):
    """
    :param seq: the sequence to fit
    :param xdata: the x-value, corresponding to the list of ALL samples
    :param maxFold: if the maximum reacted fraction can exceed the indicated value, default None
    :param fitMtd: fitting method to use
    :param ciEst: If the confidence interval will be estimated by bootstrapping
    :param func: _get_mask to fit, if None will fit to default exponential function
    :return: seq with fitting results
    """
    from scipy.optimize import curve_fit
    import numpy as np

    def exp_func(x, A, k):
        return A * (1 - np.exp(- 0.479 * 90 * k * x))  # BTO degradation adjustment and 90 minutes

    if func is None:
        func = exp_func
    ydata = np.array(seq['reactedFrac'][1:])
    if not(maxFold is None):
        ydata = np.array([min(yi, maxFold) for yi in ydata])
    valid = ~np.isnan(ydata)
    try:
        x_ = xdata[valid]
        y_ = ydata[valid]
        initGuess = [np.random.random(), np.random.random()]
        params, pcov = curve_fit(func, xdata=x_, ydata=y_,
                                 method=fitMtd, bounds=([0, 0], [1., np.inf]), p0=initGuess)
        yHat = func(x_, params[0], params[1])
        res = y_ - yHat
        pctRes = res / yHat
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((ydata[valid] - np.mean(ydata[valid])) ** 2)
        seq['r2'] = (1 - ss_res / ss_tot)
    except RuntimeError:
        params = [np.nan, np.nan]
    seq['params'] = [params[0], params[1], params[0]*params[1]]
    if ciEst:
        if (len(xdata[valid])>1)and(~np.isnan(params[0])):
            seq['bsRes'] = []
            for ix in range(201):
                bsDepth = ix * 10
                paramList = []
                for i in range(bsDepth):
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
                seq['bsRes'].append([np.nanpercentile(paramList, 2.5, axis=0),
                                     np.nanpercentile(paramList, 50, axis=0),
                                     np.nanpercentile(paramList, 97.5, axis=0),
                                     np.nanmean(paramList, axis=0),
                                     np.nanstd(paramList, axis=0, ddof=1)])
            seq['paramList'] = paramList
        else:
            seq['stdevs'] = [np.nan]
            seq['ci95'] = [np.nan]

    return seq


def work_fn(seq):
    xdata = np.array([0.00025 for i in range(6)] +
                     [0.00005 for i in range(6)] +
                     [0.00001 for i in range(6)] +
                     [0.000002 for i in range(6)])
    seq = fitting(seq, xdata, fitMtd='trf', ciEst=True)
    return seq

def main():
    timeInit = time.time()
    seqToFit = util.load_pickle('./selectedSeq_abe.pkl')
    seqToFit = np.random.choice(seqToFit, size=1, replace=False)
    print(len(seqToFit))
    pool = mp.Pool(processes=40)
    seqToFit = pool.map(work_fn, seqToFit)
    util.dump_pickle(data=seqToFit,
                     dirc='./bsConvergenceTestRes.pkl',
                     log='Bootstrapping convergence test results of selected Abe\'s data, the bootstrap is practice separately on 10, 20, ..., 2000 depth, on Pod (40 cores)',
                     overwrite=True)
    timeEnd = time.time()
    print('Process finished in %i s' % (timeEnd - timeInit))

if __name__ == '__main__':
    main()
