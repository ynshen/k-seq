'''
This module contain methods to estimate the confidence interval/standard deviation in k-seq analysis
'''
__author__ = "Yuning Shen"
__email__ = "yuningshen@ucsb.edu"

import util
import numpy as np

def method_1(sample):
    from scipy.optimize import curve_fit

    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])
    valid = ~np.isnan(y_)
    if len(y_[valid]) > 0:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]),
                                   p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
            pcov = [[np.nan, np.nan],
                    [np.nan, np.nan]]
    else:
        popt = [np.nan, np.nan]
        pcov = [[np.nan, np.nan],
                [np.nan, np.nan]]

    return ([popt[0], popt[1], popt[0] * popt[1]],
            [np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]),
             popt[0] * popt[1] * np.sqrt(pcov[0][0] / (popt[0] ** 2) + pcov[1][1] / (popt[1] ** 2) + 2 * pcov[0][1] / (popt[0] * popt[1]))])

def method_2(sample):
    from scipy.optimize import curve_fit
    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])

    valid = ~np.isnan(y_)
    if len(x_[valid]) > 0:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]),
                                   p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
    else:
        popt = [np.nan, np.nan]
    var = [popt[0], popt[1], popt[0] * popt[1]]

    varList = []
    for i in range(3):
        valid = ~np.isnan(y[i])
        if len(x[i][valid]) > 0:
            try:
                initGuess = (np.random.random(), np.random.random())
                popt, pcov = curve_fit(func, x[i][valid], y[i][valid], method='trf', bounds=([0, 0], [1., np.inf]),
                                       p0=initGuess)
            except RuntimeError:
                popt = [np.nan, np.nan]
        else:
            popt = [np.nan, np.nan]
        varList.append([popt[0], popt[1], popt[0] * popt[1]])
    return (var, np.nanstd(varList, axis=0, ddof=1))


def method_3(sample):
    from scipy.optimize import curve_fit
    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])
    valid = ~np.isnan(y_)
    if len(y_[valid])>0:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
    else:
        popt = [np.nan, np.nan]
    var = [popt[0], popt[1], popt[0] * popt[1]]

    varList = []
    yResampled = np.array([np.random.choice(y.T[k], replace=True, size=3) for k in range(4)]).T
    for i in range(3):
        valid = ~np.isnan(yResampled[i])
        if len(x[i][valid]) > 0:
            try:
                initGuess = (np.random.random(), np.random.random())
                popt, pcov = curve_fit(func, x[i][valid], yResampled[i][valid], method='trf', bounds=([0, 0], [1., np.inf]),
                                       p0=initGuess)
            except RuntimeError:
                popt = [np.nan, np.nan]
        else:
            popt = [np.nan, np.nan]
        varList.append([popt[0], popt[1], popt[0] * popt[1]])
    return (var, np.nanstd(varList, axis=0, ddof=1))


def method_4(sample):
    from scipy.optimize import curve_fit
    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])
    valid = ~np.isnan(y_)
    if len(y_[valid]) > 0:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
    else:
        popt = [np.nan, np.nan]
    var = [popt[0], popt[1], popt[0] * popt[1]]

    # Jackknife delete-1
    varList = []
    x_ = x_[valid]
    y_ = y_[valid]
    for i in range(len(x_)):
        mask = [True for i in x_]
        mask[i] = False
        if len(x_[mask]) > 0:
            try:
                initGuess = (np.random.random(), np.random.random())
                popt, pcov = curve_fit(func, x_[mask], y_[mask], method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
            except RuntimeError:
                popt = [np.nan, np.nan]
        else:
            popt = [np.nan, np.nan]
        varList.append([popt[0], popt[1], popt[0] * popt[1]])
    return (var, np.nanstd(varList, axis=0, ddof=1))


def method_5_single(sample, bsDepth=500):
    from scipy.optimize import curve_fit
    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])
    valid = ~np.isnan(y_)
    if len(y_[valid]) > 0:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
    else:
        popt = [np.nan, np.nan]
    var = [popt[0], popt[1], popt[0] * popt[1]]

    # Bootstrap
    varList = []
    x_ = x_[valid]
    y_ = y_[valid]
    yPredicted = func(x_, popt[0], popt[1])
    pctRes = (y_ - yPredicted) / yPredicted
    for i in range(bsDepth):
        pctResResampled = np.random.choice(pctRes, replace=True, size=len(pctRes))
        yToFit = yPredicted + yPredicted * pctResResampled
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_, yToFit, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [0, 0]
        except:
            print(sample)
        varList.append([popt[0], popt[1], popt[0] * popt[1]])
    return (var, varList)


def method_5_multi(sample):
    from scipy.optimize import curve_fit
    def func(x, A, k):
        return A * (1 - np.exp(-0.479 * 90 * k * x))

    bsDepth = 500
    x = np.array([sample['x'] for i in range(3)])
    y = sample['data']
    x_ = np.reshape(x, x.shape[0] * x.shape[1])
    y_ = np.reshape(y, y.shape[0] * y.shape[1])
    valid = ~np.isnan(y_)
    if len(y_[valid]) > 1:
        try:
            initGuess = (np.random.random(), np.random.random())
            popt, pcov = curve_fit(func, x_[valid], y_[valid], method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
        except RuntimeError:
            popt = [np.nan, np.nan]
        varList = []
        x_ = x_[valid]
        y_ = y_[valid]
        yPredicted = func(x_, popt[0], popt[1])
        pctRes = (y_ - yPredicted) / yPredicted
        for i in range(bsDepth):
            pctResResampled = np.random.choice(pctRes, replace=True, size=len(pctRes))
            yToFit = yPredicted + yPredicted * pctResResampled
            try:
                initGuess = (np.random.random(), np.random.random())
                popt, pcov = curve_fit(func, x_, yToFit, method='trf', bounds=([0, 0], [1., np.inf]), p0=initGuess)
            except:
                popt = [np.nan, np.nan]
                print(x, yToFit)
            varList.append([popt[0], popt[1], popt[0] * popt[1]])
    else:
        popt = [np.nan, np.nan]
        varList = [np.nan]
    var = [popt[0], popt[1], popt[0] * popt[1]]
    sample['estVar'] = var
    sample['estStd'] = varList

    return sample


def method_5_multi_main(simuSet=None):
    import time
    import multiprocessing as mp
    timeInit = time.time()

    pool = mp.Pool(processes=8)
    if simuSet is None:
        simuSet = util.load_pickle('/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_2_full.pkl')
    simuSet = pool.map(method_5_multi, simuSet)
    timeEnd = time.time()
    print('Process finished in %i s' %(timeEnd - timeInit))

    if not(simuSet is None):
        return simuSet
    else:
        pass
        # util.dump_pickle(simuSet,
        #              '/mnt/storage/projects/ribozyme_predict/data/k_seq/fit_simu/simuSet_1_method_4_parallel_res.pkl',
        #              log='10000 simulated data on [3,3,3,3] using method 5: bootstrap residues for 500 times, parallelized',
        #              overwrite=True)


def results_survey(simuSet):
    def check(sample):
        if str(list(sample['config'])) == str([0, 0, 0, 0]):
            return 0
        elif np.isnan(sample['estStd'][2]) or np.isinf(sample['estStd'][2]):
            return 0
        else:
            return 1

    pctSuccess = sum([check(sample) for sample in simuSet])/ len(simuSet)
    results = [[sample['estStd'][2] / sample['estVar'][2],
                int(sample['estVar'][2] - 2 * sample['estStd'][2] < sample['kA'] < sample['estVar'][2] + 2 * sample['estStd'][2])]
                for sample in simuSet if check(sample)==1]
    return pctSuccess, results

def results_survey_mtd45(simuSet, percentile=False):
    pctSuccess = len([1 for sample in simuSet if len(sample['estStd']) != 1])/ len(simuSet)
    if percentile:
        results = [[np.std(sample['estStd'], axis=0, ddof=1)[2] / sample['estVar'][2],
                    int(np.percentile(sample['estStd'], 2.5, axis=0)[2] < sample['kA'] < np.percentile(sample['estStd'], 97.5, axis=0)[2])]
                   for sample in simuSet if len(sample['estStd']) > 1]
    else:
        results = [[np.std(sample['estStd'], axis=0, ddof=1)[2]/sample['estVar'][2],
                    int(sample['estVar'][2] - 2* np.std(sample['estStd'], axis=0, ddof=1)[2] < sample['kA'] < sample['estVar'][2] + 2* np.std(sample['estStd'], axis=0, ddof=1)[2])]
                   for sample in simuSet if len(sample['estStd']) > 1]
    return pctSuccess, results
